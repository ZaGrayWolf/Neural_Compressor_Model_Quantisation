#!/usr/bin/env python3
"""
SegFormer-B5 Quantization Benchmark Script
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
# Critical for running on Mac without crashing
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
# 2. Model Definition (SegFormer-B5)
# ------------------------------------------------------------------------------

class OverlapPatchMerging(nn.Module):
    def __init__(self, in_ch, out_ch, patch_size, stride):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, patch_size, stride, patch_size//2, bias=False)
        self.norm = nn.LayerNorm(out_ch)
    def forward(self, x):
        # x: [B, C, H, W]
        x = self.conv(x)
        _, C2, H2, W2 = x.shape
        x = x.flatten(2).transpose(1,2) # [B, N, C]
        x = self.norm(x)
        return x, H2, W2

class EfficientSelfAttention(nn.Module):
    def __init__(self, C, heads, reduction):
        super().__init__()
        self.heads = heads
        self.scale = (C//heads)**-0.5
        self.q = nn.Linear(C, C)
        self.kv = nn.Linear(C, 2*C)
        self.proj = nn.Linear(C, C)
        if reduction > 1:
            self.sr = nn.Conv2d(C, C, reduction, reduction)
            self.norm = nn.LayerNorm(C)
        else:
            self.sr = None
            
    def forward(self, x, H, W):
        B, N, C = x.shape
        # Reshape for multi-head attention
        q = self.q(x).view(B, N, self.heads, C//self.heads).permute(0, 2, 1, 3)
        
        if self.sr:
            x2 = x.permute(0, 2, 1).view(B, C, H, W)
            x2 = self.sr(x2).view(B, C, -1).permute(0, 2, 1)
            x2 = self.norm(x2)
        else:
            x2 = x
            
        kv = self.kv(x2).view(B, -1, 2, self.heads, C//self.heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)

class MixFFN(nn.Module):
    def __init__(self, C, hidden):
        super().__init__()
        self.fc1 = nn.Linear(C, hidden)
        self.dw = nn.Conv2d(hidden, hidden, 3, 1, 1, groups=hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, C)
        
    def forward(self, x, H, W):
        B, N, C = x.shape
        x = self.fc1(x)
        x = x.transpose(1, 2).view(B, -1, H, W)
        x = self.dw(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.act(x)
        return self.fc2(x)

class SegFormerBlock(nn.Module):
    def __init__(self, C, heads, mlp_ratio, reduction):
        super().__init__()
        self.norm1 = nn.LayerNorm(C)
        self.attn = EfficientSelfAttention(C, heads, reduction)
        self.norm2 = nn.LayerNorm(C)
        self.ffn = MixFFN(C, int(C*mlp_ratio))
        
    def forward(self, x, H, W):
        x = x + self.attn(self.norm1(x), H, W)
        x = x + self.ffn(self.norm2(x), H, W)
        return x

class SegFormerB5(nn.Module):
    def __init__(self, num_classes=19):
        super().__init__()
        # B5 Configuration
        ws = [64, 128, 320, 512]
        ds = [3, 6, 40, 3]
        heads = [1, 2, 5, 8]
        rr = [8, 4, 2, 1]
        ps = [7, 3, 3, 3]
        strides = [4, 2, 2, 2]
        mlp_ratio = 4
        dec_ch = 768

        self.stages = nn.ModuleList()
        for i in range(4):
            layers = []
            layers.append(OverlapPatchMerging(
                in_ch=3 if i == 0 else ws[i-1],
                out_ch=ws[i],
                patch_size=ps[i],
                stride=strides[i]
            ))
            for _ in range(ds[i]):
                layers.append(SegFormerBlock(ws[i], heads[i], mlp_ratio, rr[i]))
            self.stages.append(nn.Sequential(*layers))

        self.norms = nn.ModuleList([nn.LayerNorm(w) for w in ws])
        self.decoder_mlps = nn.ModuleList([nn.Conv2d(w, dec_ch, 1, bias=False) for w in ws])
        self.head = nn.Conv2d(dec_ch*4, num_classes, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        feats = []
        for i, stage in enumerate(self.stages):
            # Explicit tuple unpacking for FX compatibility
            x, H2, W2 = stage[0](x)
            
            # Process blocks
            for blk in stage[1:]:
                x = blk(x, H2, W2)
            
            x = self.norms[i](x)
            
            # Reshape back to image format
            x_sp = x.transpose(1, 2).reshape(B, H2, W2, -1).permute(0, 3, 1, 2)
            feats.append(x_sp)
            x = x_sp

        # Decoder Head
        outs = []
        tgt = feats[0].shape[2:]
        for mlp, feat in zip(self.decoder_mlps, feats):
            y = mlp(feat)
            if y.shape[2:] != tgt:
                y = F.interpolate(y, size=tgt, mode='bilinear', align_corners=False)
            outs.append(y)

        x = torch.cat(outs, 1)
        x = self.head(x)
        if x.shape[2:] != (H, W):
            x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False)
        return x

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
    
    # Reduced warmup/runs for SegFormer because it is heavy
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
    
    # Input format: (Channels, Width, Height)
    input_list = [
        (3, 640, 360), (3, 1280, 720), (3, 1360, 760),
        (3, 1600, 900), (3, 1920, 1080), (3, 2048, 1152),
        (3, 2560, 1440), (3, 3840, 2160)
    ]
    
    NUM_CLASSES = 19
    BATCH_SIZE = 1
    benchmark_results = []
    
    print("\n" + "="*80)
    print("SEGFORMER-B5 MULTI-RESOLUTION BENCHMARK")
    print("="*80)

    for i, shape in enumerate(input_list):
        c, w, h = shape
        # PyTorch uses (Batch, Channels, Height, Width)
        tensor_shape = (c, h, w)
        res_label = f"{w}x{h}"
        
        print(f"\nProcessing Resolution {i+1}/{len(input_list)}: {res_label} (Tensor: {tensor_shape})")
        
        # Check for potential OOM on large resolutions before alloc
        if w * h > 2560 * 1440:
             print("  [Warning] Very high resolution. Memory usage will be high.")

        dummy_input = torch.randn(BATCH_SIZE, *tensor_shape)
        
        # 1. Setup FP32
        model_fp32 = SegFormerB5(num_classes=NUM_CLASSES).eval().cpu()
        
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
                example_inputs=dummy_input # Critical for Transformer tracing
            )
            
            try:
                # Suppress logs
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
        
        # 5. Aggressive Cleanup
        del model_fp32
        if 'quantized_model' in locals(): del quantized_model
        del dummy_input
        gc.collect()

    # -- Print Final Table --
    print("\n\nSUMMARY - SEGFORMER-B5")
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