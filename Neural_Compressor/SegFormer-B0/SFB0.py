#!/usr/bin/env python3
"""
SegFormer-B0 Quantization Benchmark Script
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
from typing import List, Tuple

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
# 2. Model Definition (SegFormer-B0)
# ------------------------------------------------------------------------------

class OverlapPatchMerging(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, patch_size: int, stride: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=patch_size, stride=stride, 
                              padding=patch_size // 2, bias=False)
        self.norm = nn.LayerNorm(out_channels)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        x = self.conv(x)
        _, _, H_new, W_new = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H_new, W_new

class EfficientSelfAttention(nn.Module):
    def __init__(self, channels: int, num_heads: int, reduction_ratio: int = 1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5
        self.reduction_ratio = reduction_ratio
        
        self.q = nn.Linear(channels, channels, bias=True)
        self.kv = nn.Linear(channels, channels * 2, bias=True)
        self.proj = nn.Linear(channels, channels)
        
        self.sr = None
        self.norm = None
        if reduction_ratio > 1:
            self.sr = nn.Conv2d(channels, channels, kernel_size=reduction_ratio, stride=reduction_ratio)
            self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        if self.sr is not None and self.reduction_ratio > 1:
            x_spatial = x.transpose(1, 2).reshape(B, C, H, W)
            x_reduced = self.sr(x_spatial)
            x_reduced = x_reduced.reshape(B, C, -1).transpose(1, 2)
            x_reduced = self.norm(x_reduced)
            x_kv = x_reduced
        else:
            x_kv = x
            
        kv = self.kv(x_kv).reshape(B, -1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x

class MixFFN(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int):
        super().__init__()
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.dwconv = nn.Conv2d(hidden_channels, hidden_channels, 3, 1, 1, bias=True, groups=hidden_channels)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_channels, in_channels)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, N, C = x.shape
        x = self.fc1(x)
        C_hidden = x.shape[-1]
        x = x.transpose(1, 2).reshape(B, C_hidden, H, W)
        x = self.dwconv(x)
        x = x.reshape(B, C_hidden, -1).transpose(1, 2)
        x = self.act(x)
        x = self.fc2(x)
        return x

class SegFormerEncoderBlock(nn.Module):
    def __init__(self, channels: int, num_heads: int, mlp_ratio: int = 4, reduction_ratio: int = 1):
        super().__init__()
        self.norm1 = nn.LayerNorm(channels)
        self.attn = EfficientSelfAttention(channels, num_heads, reduction_ratio)
        self.norm2 = nn.LayerNorm(channels)
        self.ffn = MixFFN(channels, channels * mlp_ratio)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), H, W)
        x = x + self.ffn(self.norm2(x), H, W)
        return x

class MiTEncoder(nn.Module):
    def __init__(self, in_channels: int, widths: List[int], depths: List[int], 
                 num_heads: List[int], reduction_ratios: List[int], 
                 patch_sizes: List[int], strides: List[int], mlp_ratio: int):
        super().__init__()
        
        self.num_stages = len(depths)
        self.patch_merging_layers = nn.ModuleList()
        self.transformer_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        for i in range(self.num_stages):
            in_ch = in_channels if i == 0 else widths[i-1]
            patch_merge = OverlapPatchMerging(
                in_channels=in_ch,
                out_channels=widths[i],
                patch_size=patch_sizes[i],
                stride=strides[i]
            )
            self.patch_merging_layers.append(patch_merge)
            
            transformer_blocks = nn.ModuleList([
                SegFormerEncoderBlock(widths[i], num_heads[i], mlp_ratio, reduction_ratios[i])
                for _ in range(depths[i])
            ])
            self.transformer_layers.append(transformer_blocks)
            self.norms.append(nn.LayerNorm(widths[i]))

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        features = []
        for i in range(self.num_stages):
            x, H, W = self.patch_merging_layers[i](x)
            for block in self.transformer_layers[i]:
                x = block(x, H, W)
            x = self.norms[i](x)
            B, N, C = x.shape
            x_spatial = x.transpose(1, 2).reshape(B, C, H, W)
            features.append(x_spatial)
            x = x_spatial
        return features

class SegFormerDecoder(nn.Module):
    def __init__(self, in_channels_list: List[int], out_channels: int):
        super().__init__()
        self.mlps = nn.ModuleList([
            nn.Conv2d(in_ch, out_channels, 1, bias=False) for in_ch in in_channels_list
        ])

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        target_size = features[0].shape[2:]
        outs = []
        for i, (feature, mlp) in enumerate(zip(features, self.mlps)):
            x = mlp(feature)
            if x.shape[2:] != target_size:
                x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
            outs.append(x)
        return torch.cat(outs, dim=1)

class SegFormer(nn.Module):
    def __init__(self, in_channels: int, widths: List[int], depths: List[int],
                 num_heads: List[int], reduction_ratios: List[int],
                 patch_sizes: List[int], strides: List[int], mlp_ratio: int,
                 decoder_channels: int, num_classes: int):
        super().__init__()
        self.encoder = MiTEncoder(
            in_channels, widths, depths, num_heads, reduction_ratios,
            patch_sizes, strides, mlp_ratio
        )
        self.decoder = SegFormerDecoder(widths, decoder_channels)
        self.head = nn.Conv2d(decoder_channels * len(widths), num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_shape = x.shape[-2:]
        features = self.encoder(x)
        x = self.decoder(features)
        x = self.head(x)
        if x.shape[2:] != input_shape:
            x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        return x

def segformer_b0(num_classes: int = 19):
    return SegFormer(
        in_channels=3,
        widths=[32, 64, 160, 256],      # C1, C2, C3, C4
        depths=[2, 2, 2, 2],            # L1, L2, L3, L4
        num_heads=[1, 2, 5, 8],         # H1, H2, H3, H4
        reduction_ratios=[8, 4, 2, 1],  # R1, R2, R3, R4
        patch_sizes=[7, 3, 3, 3],
        strides=[4, 2, 2, 2],
        mlp_ratio=4,
        decoder_channels=256,
        num_classes=num_classes
    )

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
    print("SEGFORMER-B0 MULTI-RESOLUTION BENCHMARK")
    print("="*80)

    for i, shape in enumerate(input_list):
        c, w, h = shape
        tensor_shape = (c, h, w)
        res_label = f"{w}x{h}"
        
        print(f"\nProcessing Resolution {i+1}/{len(input_list)}: {res_label} (Tensor: {tensor_shape})")
        
        # OOM warning
        if w * h > 2560 * 1440:
             print("  [Warning] Very high resolution. Memory usage will be high.")

        dummy_input = torch.randn(BATCH_SIZE, *tensor_shape)
        
        # 1. Setup FP32
        model_fp32 = segformer_b0(num_classes=NUM_CLASSES).eval().cpu()
        
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
    print("\n\nSUMMARY - SEGFORMER-B0")
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