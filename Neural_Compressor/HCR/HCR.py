#!/usr/bin/env python3
"""
HRNet-W48 + OCR Quantization Benchmark Script
Compatible with Intel Neural Compressor (INC).
"""

import os
import sys
import time
import warnings
import gc
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np

# --- 0. PATCHES & CONFIGURATION ---

# Patch numpy for legacy HRNet code
if not hasattr(np, 'int'):
    np.int = int

# HRNet Configuration Data
EXTRA = {
    "STAGE1": {
        "NUM_MODULES": 1, "NUM_BRANCHES": 1,
        "NUM_BLOCKS": [4], "NUM_CHANNELS": [64],
        "BLOCK": "BOTTLENECK", "FUSE_METHOD": "SUM"
    },
    "STAGE2": {
        "NUM_MODULES": 1, "NUM_BRANCHES": 2,
        "NUM_BLOCKS": [4, 4], "NUM_CHANNELS": [48, 96],
        "BLOCK": "BASIC", "FUSE_METHOD": "SUM"
    },
    "STAGE3": {
        "NUM_MODULES": 4, "NUM_BRANCHES": 3,
        "NUM_BLOCKS": [4, 4, 4], "NUM_CHANNELS": [48, 96, 192],
        "BLOCK": "BASIC", "FUSE_METHOD": "SUM"
    },
    "STAGE4": {
        "NUM_MODULES": 3, "NUM_BRANCHES": 4,
        "NUM_BLOCKS": [4, 4, 4, 4], "NUM_CHANNELS": [48, 96, 192, 384],
        "BLOCK": "BASIC", "FUSE_METHOD": "SUM"
    },
    "FINAL_CONV_KERNEL": 1,
    "WITH_HEAD": True,
    "OCR": {"MID_CHANNELS": 512, "KEY_CHANNELS": 256, "DROPOUT": 0.05},
}

class Config:
    class MODEL:
        EXTRA = EXTRA
        ALIGN_CORNERS = False
        OCR = SimpleNamespace(**EXTRA["OCR"])
    class DATASET:
        NUM_CLASSES = 19

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
# 2. Model Loading (Dynamic Import)
# ------------------------------------------------------------------------------

MODULE_FILE = "hrnet_w48_ocr_full.py"
if not Path(MODULE_FILE).is_file():
    print(f"\n[Error] Expected {MODULE_FILE} in the current directory.")
    sys.exit(1)

spec = importlib.util.spec_from_file_location("hrnet_w48_ocr", MODULE_FILE)
hrnet_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hrnet_module)

def load_model():
    config = Config()
    model = hrnet_module.HighResolutionNet(config, num_classes=19)
    return model

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
    
    # HRNet is very heavy, minimal warmup to save time
    try:
        with torch.no_grad():
            for _ in range(2): 
                model(input_tensor)
    except Exception as e:
        if verbose: print(f"Warmup failed: {e}")
        return 0.0

    num_runs = 5 # Reduced runs for HRNet due to high compute cost
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
    
    BATCH_SIZE = 1
    benchmark_results = []
    
    print("\n" + "="*80)
    print("HRNET-W48 + OCR MULTI-RESOLUTION BENCHMARK")
    print("="*80)

    for i, shape in enumerate(input_list):
        c, w, h = shape
        tensor_shape = (c, h, w)
        res_label = f"{w}x{h}"
        
        print(f"\nProcessing Resolution {i+1}/{len(input_list)}: {res_label} (Tensor: {tensor_shape})")
        
        # HRNet W48 consumes massive memory at high res
        if w * h > 1920 * 1080:
             print("  [Warning] Very high resolution. High risk of OOM on standard machines.")

        dummy_input = torch.randn(BATCH_SIZE, *tensor_shape)
        
        # 1. Setup FP32
        try:
            model_fp32 = load_model().eval().cpu()
        except Exception as e:
            print(f"  -> Model Load Failed: {e}")
            continue

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
            calib_dataset = DummyCalibrationDataset(input_shape=tensor_shape, num_samples=3) # Reduced samples
            calib_dataloader = DataLoader(calib_dataset, batch_size=BATCH_SIZE)
            set_random_seed(42)
            
            config = PostTrainingQuantConfig(
                approach="static",
                calibration_sampling_size=3,
                reduce_range=True,
                example_inputs=dummy_input # Critical for tracing HRNet branches
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
        
        # 5. Aggressive Cleanup
        del model_fp32
        if 'quantized_model' in locals(): del quantized_model
        del dummy_input
        gc.collect()

    # -- Print Final Table --
    print("\n\nSUMMARY - HRNET-W48 + OCR")
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