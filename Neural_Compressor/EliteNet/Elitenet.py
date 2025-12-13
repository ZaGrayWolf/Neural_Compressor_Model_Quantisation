#!/usr/bin/env python3
"""
Elitenet Quantization Benchmark Script
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
# 2. Model Definition (Elitenel)
# ------------------------------------------------------------------------------

class ConvBlockNoSkip(nn.Module):
    def __init__(self, in_c, out_c, k_sz=3, pool=True):
        super().__init__()
        pad = (k_sz - 1) // 2
        self.pool = nn.MaxPool2d(kernel_size=2) if pool else None

        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=k_sz, padding=pad, bias=True),
            nn.ReLU(),
            nn.BatchNorm2d(out_c),
            nn.Conv2d(out_c, out_c, kernel_size=k_sz, padding=pad, bias=True),
            nn.ReLU(),
            nn.BatchNorm2d(out_c)
        )

    def forward(self, x):
        if self.pool:
            x = self.pool(x)
        return self.block(x)

class UNetEncoderWithSkip(nn.Module):
    def __init__(self, in_c, layers, k_sz=3):
        super().__init__()
        self.layers = layers
        self.first = ConvBlockNoSkip(in_c, layers[0], k_sz, pool=False)

        self.down_path = nn.ModuleList()
        self.up_path = nn.ModuleList()

        for i in range(len(layers) - 1):
            self.down_path.append(ConvBlockNoSkip(layers[i], layers[i+1], k_sz, pool=True))
        for i in range(len(layers) - 1):
            self.up_path.append(nn.ConvTranspose2d(layers[i+1], layers[i], kernel_size=2, stride=2, bias=True))

        self.final_conv = nn.Conv2d(layers[0], in_c, kernel_size=1, bias=True)

    def forward(self, x):
        # We need to store skip connections for the decoder
        activations = []

        # Encoder
        x1 = self.first(x)
        activations.append(x1)
        x = x1
        
        for down in self.down_path:
            x = down(x)
            activations.append(x)

        # Decoder
        # Reverse iterate up_path
        for i, up in enumerate(reversed(self.up_path)):
            x = up(x)
            # Retrieve corresponding encoder feature map
            skip_connection = activations[len(self.layers)-2-i]

            # Fix mismatched spatial size by cropping/padding
            # This logic is captured by the tracer via example_inputs
            diff_h = skip_connection.size(2) - x.size(2)
            diff_w = skip_connection.size(3) - x.size(3)
            
            if diff_h != 0 or diff_w != 0:
                pad_left = diff_w // 2
                pad_right = diff_w - pad_left
                pad_top = diff_h // 2
                pad_bottom = diff_h - pad_top
                x = F.pad(x, [pad_left, pad_right, pad_top, pad_bottom])

            x = x + skip_connection

        # Final conv
        x = self.final_conv(x)
        
        # Modified: Return only the output tensor for quantization compatibility
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
    
    IN_CHANNELS = 3
    LAYERS = [4, 8, 16, 32] # As per your config
    BATCH_SIZE = 1
    
    benchmark_results = []
    
    print("\n" + "="*80)
    print("ELITENEL MULTI-RESOLUTION BENCHMARK")
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
        model_fp32 = UNetEncoderWithSkip(in_c=IN_CHANNELS, layers=LAYERS, k_sz=3).eval().cpu()
        
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
                example_inputs=dummy_input # Critical for FX tracing of F.pad logic
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
    print("\n\nSUMMARY - ELITENEL")
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