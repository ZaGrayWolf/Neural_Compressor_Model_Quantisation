#!/usr/bin/env python3
"""
ERFNet Quantization Benchmark Script
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
# 2. Model Definition (ERFNet)
# ------------------------------------------------------------------------------

class ConvBNAct(nn.Sequential):
    """Convolution -> BatchNorm -> ReLU"""
    def __init__(self, in_c, out_c, k, s=1, ph=0, pw=0, dil=1):
        super().__init__(
            nn.Conv2d(in_c, out_c, k, stride=s, padding=(ph, pw), dilation=dil, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

class DeConvBNAct(nn.Sequential):
    """Transposed Convolution -> BatchNorm -> ReLU"""
    def __init__(self, in_c, out_c):
        super().__init__(
            nn.ConvTranspose2d(in_c, out_c, 3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

class InitialBlock(nn.Module):
    """Initial downsampling block with CORRECT BatchNorm channel calculation."""
    def __init__(self, in_c, out_c):
        super().__init__()
        # The conv branch outputs (out_c - in_c) channels.
        # This way, (out_c - in_c) from conv + in_c from pool = out_c channels for BN.
        self.conv = nn.Conv2d(in_c, out_c - in_c, 3, stride=2, padding=1, bias=False)
        self.pool = nn.MaxPool2d(3, stride=2, padding=1)
        self.bn   = nn.BatchNorm2d(out_c) 
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        c = self.conv(x)
        p = self.pool(x)
        out = torch.cat([c, p], dim=1)
        return self.relu(self.bn(out))

class NonBt1DBlock(nn.Module):
    """Non-bottleneck 1D block with residual connection"""
    def __init__(self, ch, dilation=1):
        super().__init__()
        self.conv = nn.Sequential(
            ConvBNAct(ch, ch, (3,1), ph=1, pw=0),
            ConvBNAct(ch, ch, (1,3), ph=0, pw=1),
            ConvBNAct(ch, ch, (3,1), ph=dilation, pw=0, dil=dilation),
            ConvBNAct(ch, ch, (1,3), ph=0, pw=dilation, dil=dilation)
        )
        self.bnact = nn.Sequential(nn.BatchNorm2d(ch), nn.ReLU(True))

    def forward(self, x):
        res = x
        x = self.conv(x)
        return self.bnact(x + res)

class ERFNet(nn.Module):
    def __init__(self, num_class=19):
        super().__init__()
        self.l1  = InitialBlock(3, 16) 
        self.l2  = InitialBlock(16, 64)
        self.l3  = nn.Sequential(*[NonBt1DBlock(64) for _ in range(5)])
        self.l8  = InitialBlock(64, 128)
        self.l9  = nn.Sequential(*[NonBt1DBlock(128, d) for d in [2,4,8,16,2,4,8,16]])
        self.l17 = DeConvBNAct(128, 64)
        self.l18 = nn.Sequential(*[NonBt1DBlock(64) for _ in range(2)])
        self.l20 = DeConvBNAct(64, 16)
        self.l21 = nn.Sequential(*[NonBt1DBlock(16) for _ in range(2)])
        self.l23 = DeConvBNAct(16, num_class)

    def forward(self, x):
        x = self.l1(x); x = self.l2(x)
        x = self.l3(x); x = self.l8(x)
        x = self.l9(x); x = self.l17(x)
        x = self.l18(x); x = self.l20(x)
        x = self.l21(x); return self.l23(x)

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
    
    NUM_CLASSES = 19
    BATCH_SIZE = 1
    benchmark_results = []
    
    print("\n" + "="*80)
    print("ERFNET MULTI-RESOLUTION BENCHMARK")
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
        model_fp32 = ERFNet(num_class=NUM_CLASSES).eval().cpu()
        
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
                example_inputs=dummy_input 
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
    print("\n\nSUMMARY - ERFNET")
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