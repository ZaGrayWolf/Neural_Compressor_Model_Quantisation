#!/usr/bin/env python3
"""
CompNet (Autoencoder) Quantization Benchmark Script
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
# 2. Model Definition (CompNet)
# ------------------------------------------------------------------------------

class encoder(nn.Module):
    def __init__(self, n_downconv=3, in_chn=3):
        super().__init__()
        self.n_downconv = n_downconv
        layer_list = [
            nn.Conv2d(in_channels=in_chn, out_channels=64, kernel_size=3, stride=1, padding=1), 
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        ]
        for i in range(self.n_downconv):
            layer_list.extend([
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
            ])
        layer_list.append(
            nn.Conv2d(in_channels=64, out_channels=16, kernel_size=3, stride=1, padding=1),
        )
        self.encoder = nn.Sequential(*layer_list)

    def forward(self, x):
        return torch.clamp(self.encoder(x), 0, 1)

class decoder(nn.Module):
    def __init__(self, n_upconv=3, out_chn=3):
        super().__init__()
        self.n_upconv = n_upconv
        layer_list = [
            nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        ]
        for i in range(self.n_upconv):
            layer_list.extend([
                nn.Conv2d(in_channels=64, out_channels=64*4, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.PixelShuffle(2),
            ])
        layer_list.extend([
            nn.Conv2d(in_channels=64, out_channels=out_chn*4, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(2)
        ])
        self.decoder = nn.Sequential(*layer_list)

    def forward(self, x):
        return torch.clamp(self.decoder(x), 0, 1)

class autoencoder(nn.Module):
    def __init__(self, n_updownconv=3, in_chn=3, out_chn=3):
        super().__init__()
        self.n_updownconv = n_updownconv
        self.in_chn = in_chn
        self.out_chn = out_chn
        self.encoder = encoder(n_downconv=self.n_updownconv, in_chn=self.in_chn)
        self.decoder = decoder(n_upconv=self.n_updownconv, out_chn=self.out_chn)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
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
    
    # Warmup
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
    
    # Input format from your code was (H, W). Converting to full list here.
    input_hw_sizes = [
        (360, 640),
        (720, 1280),
        (760, 1360),
        (900, 1600),
        (1080, 1920),
        (1152, 2048),
        (1440, 2560),
        (2160, 3840),
    ]
    
    BATCH_SIZE = 1
    CHANNELS = 3
    n_updownconv = 3
    benchmark_results = []
    
    print("\n" + "="*80)
    print("COMPNET (AUTOENCODER) MULTI-RESOLUTION BENCHMARK")
    print("="*80)

    for i, (h, w) in enumerate(input_hw_sizes):
        # PyTorch expects (Batch, Channels, Height, Width)
        tensor_shape = (CHANNELS, h, w)
        res_label = f"{w}x{h}" # Width x Height usually standard for labeling
        
        print(f"\nProcessing Resolution {i+1}/{len(input_hw_sizes)}: {res_label} (Tensor: {tensor_shape})")
        
        # OOM warning for high res
        if w * h > 2560 * 1440:
             print("  [Warning] Very high resolution. Memory usage will be high.")

        dummy_input = torch.randn(BATCH_SIZE, *tensor_shape)
        
        # 1. Setup FP32
        model_fp32 = autoencoder(n_updownconv=n_updownconv, in_chn=CHANNELS, out_chn=CHANNELS).eval().cpu()
        
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
                example_inputs=dummy_input # Critical for FX tracing of clamp/custom layers
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
    print("\n\nSUMMARY - COMPNET")
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