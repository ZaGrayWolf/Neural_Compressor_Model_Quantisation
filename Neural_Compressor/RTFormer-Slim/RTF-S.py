#!/usr/bin/env python3
"""
RTFormer-Slim Quantization Benchmark Script
Compatible with Intel Neural Compressor (INC).
"""

import os
import sys
import time
import warnings
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# --- 1. CONFIGURATION & MACOS BACKEND ---
if sys.platform == "darwin":
    torch.backends.quantized.engine = "qnnpack"
    os.environ["PYTORCH_JIT"] = "0"
else:
    torch.backends.quantized.engine = "fbgemm"

# Suppress logs
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" 

INC_AVAILABLE = False
try:
    from neural_compressor.quantization import fit
    from neural_compressor.config import PostTrainingQuantConfig
    from neural_compressor import set_random_seed
    INC_AVAILABLE = True
except ImportError:
    pass

# ------------------------------------------------------------------------------
# 2. Model Definition (RTFormer-Slim)
# ------------------------------------------------------------------------------

class GFA(nn.Module):
    def __init__(self, dim, reduction=1):
        super().__init__()
        self.reduction = reduction
        if reduction > 1:
            self.sr = nn.Conv2d(dim, dim, reduction, reduction)
            self.norm = nn.LayerNorm(dim)
        else:
            self.sr = None
        self.q   = nn.Linear(dim, dim)
        self.kv  = nn.Linear(dim, dim * 2)
        self.proj= nn.Linear(dim, dim)
        self.scale = (dim) ** -0.5

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, C).unsqueeze(1)
        if self.sr:
            x2 = x.transpose(1,2).view(B, C, H, W)
            x2 = self.sr(x2).view(B, C, -1).transpose(1,2)
            x2 = self.norm(x2)
        else:
            x2 = x
        kv = self.kv(x2).reshape(B, -1, 2, C).permute(2,0,1,3)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2,-1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).squeeze(1)

        return self.proj(out)

class RTFormerBlock(nn.Module):
    def __init__(self, dim, reduction):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = GFA(dim, reduction)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn   = nn.Sequential(
            nn.Linear(dim, dim*4),
            nn.GELU(),
            nn.Linear(dim*4, dim)
        )

    def forward(self, x, H, W):
        x = x + self.attn(self.norm1(x), H, W)
        x = x + self.ffn(self.norm2(x))
        return x

class OverlapPatchMerging(nn.Module):
    def __init__(self, in_ch, out_ch, patch_size, stride):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, patch_size, stride, patch_size//2, bias=False)
        self.norm = nn.LayerNorm(out_ch)

    def forward(self, x):
        x = self.conv(x)
        B, C2, H2, W2 = x.shape
        x = x.flatten(2).transpose(1,2)
        x = self.norm(x)
        return x, H2, W2

class RTFormerBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        widths    = [32, 64, 128, 256]
        depths    = [2, 2, 2, 2]
        reductions= [8, 4, 2, 1]
        patch_sz  = [3, 3, 3, 3]
        strides   = [1, 2, 2, 2]

        self.stages = nn.ModuleList()
        self.norms  = nn.ModuleList()
        in_ch = 3
        for w, d, r, ps, st in zip(widths, depths, reductions, patch_sz, strides):
            layers = [OverlapPatchMerging(in_ch, w, ps, st)]
            for _ in range(d):
                layers.append(RTFormerBlock(w, r))
            self.stages.append(nn.Sequential(*layers))
            self.norms.append(nn.LayerNorm(w))
            in_ch = w

    def forward(self, x):
        feats = []
        for stage, norm in zip(self.stages, self.norms):
            # Manually unpack tuple from OverlapPatchMerging
            # This is critical for FX tracing compatibility
            merged, H, W = stage[0](x)
            
            x_seq = merged
            for i in range(1, len(stage)):
                x_seq = stage[i](x_seq, H, W)
            
            x_seq = norm(x_seq)
            B, N, C = x_seq.shape
            x_sp = x_seq.transpose(1,2).view(B, C, H, W)
            feats.append(x_sp)
            x = x_sp
        return feats

class DAPPM(nn.Module):
    def __init__(self, in_ch, mid_ch):
        super().__init__()
        self.scale1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, mid_ch, 1, bias=False),
            nn.BatchNorm2d(mid_ch), nn.ReLU()
        )
        self.scale2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(2),
            nn.Conv2d(in_ch, mid_ch, 1, bias=False),
            nn.BatchNorm2d(mid_ch), nn.ReLU()
        )
        self.scale3 = nn.Sequential(
            nn.AdaptiveAvgPool2d(4),
            nn.Conv2d(in_ch, mid_ch, 1, bias=False),
            nn.BatchNorm2d(mid_ch), nn.ReLU()
        )
        self.scale4 = nn.Sequential(
            nn.AdaptiveAvgPool2d(8),
            nn.Conv2d(in_ch, mid_ch, 1, bias=False),
            nn.BatchNorm2d(mid_ch), nn.ReLU()
        )
        self.process = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 1, bias=False),
            nn.BatchNorm2d(mid_ch), nn.ReLU()
        )
        self.compress = nn.Sequential(
            nn.Conv2d(mid_ch*5, mid_ch, 1, bias=False),
            nn.BatchNorm2d(mid_ch), nn.ReLU()
        )

    def forward(self, x):
        size = x.shape[2:]
        x0 = self.process(x)
        x1 = F.interpolate(self.scale1(x), size, mode='bilinear', align_corners=False)
        x2 = F.interpolate(self.scale2(x), size, mode='bilinear', align_corners=False)
        x3 = F.interpolate(self.scale3(x), size, mode='bilinear', align_corners=False)
        x4 = F.interpolate(self.scale4(x), size, mode='bilinear', align_corners=False)
        return self.compress(torch.cat([x0, x1, x2, x3, x4], 1))

class RTFormerHead(nn.Module):
    def __init__(self, in_chs=[32,64,128,256], mid_ch=128, num_classes=19):
        super().__init__()
        self.mlps = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ic, mid_ch, 1, bias=False),
                nn.BatchNorm2d(mid_ch), nn.ReLU()
            ) for ic in in_chs
        ])
        self.dappm = DAPPM(mid_ch * len(in_chs), mid_ch)
        self.final = nn.Sequential(
            nn.Conv2d(mid_ch, mid_ch, 3, 1, 1, bias=False),
            nn.BatchNorm2d(mid_ch), nn.ReLU(),
            nn.Dropout2d(0.1),
            nn.Conv2d(mid_ch, num_classes, 1)
        )

    def forward(self, feats):
        size = feats[0].shape[2:]
        outs = []
        for mlp, f in zip(self.mlps, feats):
            y = mlp(f)
            if y.shape[2:] != size:
                y = F.interpolate(y, size, mode='bilinear', align_corners=False)
            outs.append(y)
        x = torch.cat(outs, 1)
        x = self.dappm(x)
        return self.final(x)

class RTFormerSlim(nn.Module):
    def __init__(self, num_classes=19):
        super().__init__()
        self.backbone = RTFormerBackbone()
        self.head     = RTFormerHead(num_classes=num_classes)

    def forward(self, x):
        feats = self.backbone(x)
        return self.head(feats)

# ------------------------------------------------------------------------------
# 3. Helpers & Benchmarking
# ------------------------------------------------------------------------------

class DummyCalibrationDataset(Dataset):
    def __init__(self, input_shape, num_samples=10):
        self.num_samples = num_samples
        self.input_shape = input_shape
    def __len__(self):
        return self.num_samples
    def __getitem__(self, idx):
        return torch.randn(self.input_shape)

def measure_latency(model, input_tensor, model_name="Model", verbose=True):
    device = torch.device('cpu')
    model.to(device)
    model.eval()
    
    try:
        with torch.no_grad():
            for _ in range(3): 
                model(input_tensor)
    except Exception as e:
        if verbose: print(f"Warmup failed: {e}")
        return 0.0

    num_runs = 10 
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            start = time.perf_counter()
            model(input_tensor)
            end = time.perf_counter()
            times.append((end - start)*1000)
    
    avg_latency = sum(times) / len(times)
    if verbose:
        print(f"[{model_name}] Average Latency: {avg_latency:.2f} ms")
    return avg_latency

# ------------------------------------------------------------------------------
# 4. Main Execution
# ------------------------------------------------------------------------------

if __name__=='__main__':
    if not INC_AVAILABLE:
        print("FATAL ERROR: 'neural-compressor' is not installed.")
        sys.exit(1)
        
    print(f"Running on Platform: {sys.platform}")
    print(f"Quantization Engine: {torch.backends.quantized.engine}")
    
    input_list = [
        (3, 640, 360), (3, 1280, 720), (3, 1360, 760),
        (3, 1600, 900), (3, 1920, 1080), (3, 2048, 1152),
        (3, 2560, 1440), (3, 3840, 2160)
    ]
    
    NUM_CLASSES = 19
    BATCH_SIZE = 1
    benchmark_results = []
    
    print("\n" + "="*80)
    print("RTFORMER-SLIM MULTI-RESOLUTION BENCHMARK")
    print("="*80)

    for i, shape in enumerate(input_list):
        c, w, h = shape
        tensor_shape = (c, h, w)
        res_label = f"{w}x{h}"
        
        print(f"\nProcessing Resolution {i+1}/{len(input_list)}: {res_label} (Tensor: {tensor_shape})")
        
        if w * h > 2560 * 1440:
             print("  [Warning] Very high resolution. Memory usage will be high.")

        dummy_input = torch.randn(BATCH_SIZE, *tensor_shape)
        
        # 1. Setup FP32
        model_fp32 = RTFormerSlim(num_classes=NUM_CLASSES).eval().cpu()
        
        # 2. Measure FP32 Latency
        print("  -> Measuring FP32 Latency...")
        try:
            fp32_lat = measure_latency(model_fp32, dummy_input, model_name="FP32", verbose=True)
        except Exception as e:
            print(f"  -> FP32 Failed (likely OOM): {e}")
            fp32_lat = 0.0

        # 3. Quantize
        int8_lat = 0.0
        if fp32_lat > 0:
            print("  -> Quantizing...")
            calib_dataset = DummyCalibrationDataset(input_shape=tensor_shape, num_samples=5)
            calib_dataloader = DataLoader(calib_dataset, batch_size=BATCH_SIZE)
            set_random_seed(42)
            
            config = PostTrainingQuantConfig(
                approach="static",
                calibration_sampling_size=5,
                reduce_range=True,
                example_inputs=dummy_input # Critical for Transformer logic
            )
            
            try:
                # Suppress INC logs
                quantized_model = fit(
                    model=model_fp32,
                    conf=config,
                    calib_dataloader=calib_dataloader
                )
                
                if quantized_model:
                     print("  -> Measuring INT8 Latency...")
                     int8_lat = measure_latency(quantized_model, dummy_input, model_name="INT8", verbose=True)
                else:
                    print("  -> Quantization returned None.")
                    
            except Exception as e:
                print(f"  -> Quantization Failed: {e}")
                int8_lat = 0.0
        
        # 4. Record Results
        benchmark_results.append({
            "size": res_label,
            "fp32": fp32_lat,
            "int8": int8_lat
        })
        
        # 5. Cleanup
        del model_fp32
        if 'quantized_model' in locals(): del quantized_model
        del dummy_input
        gc.collect()

    # -- Print Final Table --
    print("\n\nSUMMARY - RTFORMER-SLIM")
    print("="*105)
    w_size = 20
    w_fp32 = 25
    w_int8 = 25
    w_speed = 15
    
    header = f" {'Input Size':<{w_size}} | {'Latency FP32 (ms)':<{w_fp32}} | {'Latency INT8 (ms)':<{w_int8}} | {'Speedup':<{w_speed}}"
    print(header)
    print("-" * 105)
    
    for res in benchmark_results:
        fp32_str = f"{res['fp32']:.2f}" if res['fp32'] > 0 else "N/A"
        int8_str = f"{res['int8']:.2f}" if res['int8'] > 0 else "N/A"
        
        speedup_str = "N/A"
        if res['fp32'] > 0 and res['int8'] > 0:
            speedup = res['fp32'] / res['int8']
            speedup_str = f"{speedup:.2f}x"
            
        row = f" {res['size']:<{w_size}} | {fp32_str:<{w_fp32}} | {int8_str:<{w_int8}} | {speedup_str:<{w_speed}}"
        print(row)
    print("="*105 + "\n")