#!/usr/bin/env python3
"""
UNEXT Quantization Benchmark Script
Compatible with Intel Neural Compressor (INC).
"""

import os
import sys
import time
import warnings
import math
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

# Check for timm
try:
    from timm.layers import to_2tuple, DropPath
except ImportError:
    print("FATAL ERROR: 'timm' library not found. Please run: pip install timm")
    sys.exit(1)

INC_AVAILABLE = False
try:
    from neural_compressor.quantization import fit
    from neural_compressor.config import PostTrainingQuantConfig
    from neural_compressor import set_random_seed
    INC_AVAILABLE = True
except ImportError:
    pass

# ------------------------------------------------------------------------------
# 2. Model Definition (UNEXT)
# ------------------------------------------------------------------------------

class DWConv(nn.Module):
    """ Depth-wise convolution """
    def __init__(self, dim=768):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).reshape(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        return x

class shiftmlp(nn.Module):
    """ Shifted MLP """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.fc2(x)
        return x

class shiftedBlock(nn.Module):
    """ Block combining LayerNorm, MLP, and residual connection """
    def __init__(self, dim, mlp_ratio=4., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = shiftmlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, H, W):
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x

class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding with overlapping patches """
    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        stride = to_2tuple(stride)
        padding = (patch_size[0] // 2, patch_size[1] // 2) if patch_size[0] > 1 else (0,0)

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride, padding=padding)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        H_out, W_out = x.shape[-2:]
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H_out, W_out

class UNext(nn.Module):
    def __init__(self, num_classes=4, input_channels=3, depths=[1, 1], mlp_ratios=[4., 4.], drop_path_rate=0.1, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dims = [16, 32, 128, 160, 256]
        self.decoder_dims = [160, 128, 32, 16, 16]
        self.depths = depths
        self.mlp_ratios = mlp_ratios

        # Encoder
        self.encoder1 = nn.Conv2d(input_channels, self.dims[0], 3, padding=1)
        self.ebn1 = nn.BatchNorm2d(self.dims[0])
        self.encoder2 = nn.Conv2d(self.dims[0], self.dims[1], 3, padding=1)
        self.ebn2 = nn.BatchNorm2d(self.dims[1])
        self.encoder3 = nn.Conv2d(self.dims[1], self.dims[2], 3, padding=1)
        self.ebn3 = nn.BatchNorm2d(self.dims[2])
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(2, 2)

        # Transformer Encoder Stages
        self.patch_embed3 = OverlapPatchEmbed(in_chans=self.dims[2], embed_dim=self.dims[3], patch_size=3, stride=2)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths) * 2)]
        cur = 0
        self.block1 = nn.ModuleList([shiftedBlock(dim=self.dims[3], mlp_ratio=mlp_ratios[0], drop_path=dpr[cur + i], norm_layer=norm_layer)
                                     for i in range(depths[0])])
        self.norm3 = norm_layer(self.dims[3])
        cur += depths[0]

        self.patch_embed4 = OverlapPatchEmbed(in_chans=self.dims[3], embed_dim=self.dims[4], patch_size=3, stride=2)
        self.block2 = nn.ModuleList([shiftedBlock(dim=self.dims[4], mlp_ratio=mlp_ratios[1], drop_path=dpr[cur + i], norm_layer=norm_layer)
                                     for i in range(depths[1])])
        self.norm4 = norm_layer(self.dims[4])
        cur += depths[1]

        # Decoder
        self.decoder1 = nn.Conv2d(self.dims[4] + self.dims[3], self.decoder_dims[0], 3, padding=1)
        self.dbn1 = nn.BatchNorm2d(self.decoder_dims[0])
        self.dblock1 = nn.ModuleList([shiftedBlock(dim=self.decoder_dims[0], mlp_ratio=mlp_ratios[0], drop_path=dpr[cur + i], norm_layer=norm_layer)
                                     for i in range(depths[0])])
        self.dnorm3 = norm_layer(self.decoder_dims[0])
        cur += depths[0]
        
        self.decoder2 = nn.Conv2d(self.decoder_dims[0] + self.dims[2], self.decoder_dims[1], 3, padding=1)
        self.dbn2 = nn.BatchNorm2d(self.decoder_dims[1])
        self.dblock2 = nn.ModuleList([shiftedBlock(dim=self.decoder_dims[1], mlp_ratio=mlp_ratios[1], drop_path=dpr[cur + i], norm_layer=norm_layer)
                                     for i in range(depths[1])])
        self.dnorm4 = norm_layer(self.decoder_dims[1])
        
        self.decoder3 = nn.Conv2d(self.decoder_dims[1] + self.dims[1], self.decoder_dims[2], 3, padding=1)
        self.dbn3 = nn.BatchNorm2d(self.decoder_dims[2])
        self.decoder4 = nn.Conv2d(self.decoder_dims[2] + self.dims[0], self.decoder_dims[3], 3, padding=1)
        self.dbn4 = nn.BatchNorm2d(self.decoder_dims[3])
        self.decoder5 = nn.Conv2d(self.decoder_dims[3], self.decoder_dims[4], 3, padding=1)
        self.dbn5 = nn.BatchNorm2d(self.decoder_dims[4])
        self.final = nn.Conv2d(self.decoder_dims[4], num_classes, 1)

    def _decode_block(self, x, skip, conv, bn, blocks, norm):
        x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.relu(bn(conv(x)))
        B_dec, C_dec, H_dec, W_dec = x.shape
        x_flat = x.flatten(2).transpose(1, 2)
        for blk in blocks:
            x_flat = blk(x_flat, H_dec, W_dec)
        x = norm(x_flat).transpose(1, 2).reshape(B_dec, C_dec, H_dec, W_dec)
        return x

    def _upsample(self, x, skip, conv, bn, target_size_override=None):
        if target_size_override:
            size = target_size_override
        else:
            size = skip.shape[2:] if skip is not None else None
        
        if size is None:
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        else:
            x = F.interpolate(x, size=size, mode='bilinear', align_corners=False)
        
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        return self.relu(bn(conv(x)))

    def forward(self, x):
        input_size = x.shape[2:]

        # Encoder 
        e1 = self.relu(self.ebn1(self.encoder1(x)))
        e2_in = self.maxpool(e1)
        e2 = self.relu(self.ebn2(self.encoder2(e2_in)))
        e3_in = self.maxpool(e2)
        e3 = self.relu(self.ebn3(self.encoder3(e3_in)))
        
        # Transformer Stage 1
        t4_in = self.maxpool(e3)
        t4, h4, w4 = self.patch_embed3(t4_in)
        for blk in self.block1:
            t4 = blk(t4, h4, w4)
        t4 = self.norm3(t4)
        t4_spatial = t4.transpose(1, 2).reshape(t4.shape[0], self.dims[3], h4, w4)
        
        # Transformer Stage 2
        t5, h5, w5 = self.patch_embed4(t4_spatial)
        for blk in self.block2:
            t5 = blk(t5, h5, w5)
        t5 = self.norm4(t5)
        t5_spatial = t5.transpose(1, 2).reshape(t5.shape[0], self.dims[4], h5, w5)
        
        # Decoder 
        d1 = self._decode_block(t5_spatial, t4_spatial, self.decoder1, self.dbn1, self.dblock1, self.dnorm3)
        d2 = self._decode_block(d1, e3, self.decoder2, self.dbn2, self.dblock2, self.dnorm4)
        d3 = self._upsample(d2, e2, self.decoder3, self.dbn3)
        d4 = self._upsample(d3, e1, self.decoder4, self.dbn4)
        d5 = self._upsample(d4, None, self.decoder5, self.dbn5, target_size_override=input_size)
        
        out = self.final(d5)
        out = F.interpolate(out, size=input_size, mode='bilinear', align_corners=False)
        return out

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
            for _ in range(5): 
                model(input_tensor)
    except Exception as e:
        if verbose: print(f"Warmup failed: {e}")
        return 0.0

    num_runs = 20 
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
    
    NUM_CLASSES = 4
    BATCH_SIZE = 1
    fixed_depths = [1, 1]
    fixed_mlp_ratios = [4., 4.]
    
    benchmark_results = []
    
    print("\n" + "="*80)
    print("UNEXT MULTI-RESOLUTION BENCHMARK")
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
        model_fp32 = UNext(
            num_classes=NUM_CLASSES, 
            input_channels=3, 
            depths=fixed_depths, 
            mlp_ratios=fixed_mlp_ratios
        ).eval().cpu()
        
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
    print("\n\nSUMMARY - UNEXT")
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