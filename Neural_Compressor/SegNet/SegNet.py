#!/usr/bin/env python3
"""
SegNet Quantization Benchmark Script (Fixed)
Fixes: Replaces MaxUnpool2d with Interpolate to remove tuple outputs,
       fixing the 'tuple object has no attribute numel' crash in INC.
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
# 2. Model Definition (SegNet - Quantization Friendly)
# ------------------------------------------------------------------------------

class SegNet(nn.Module):
    def __init__(self, input_channels=3, n_labels=21, kernel_size=3):
        super(SegNet, self).__init__()
        
        # Encoder blocks
        self.conv1_1 = nn.Conv2d(input_channels, 64, kernel_size, padding=1)
        self.bn1_1 = nn.BatchNorm2d(64)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size, padding=1)
        self.bn1_2 = nn.BatchNorm2d(64)
        
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size, padding=1)
        self.bn2_1 = nn.BatchNorm2d(128)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size, padding=1)
        self.bn2_2 = nn.BatchNorm2d(128)
        
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size, padding=1)
        self.bn3_1 = nn.BatchNorm2d(256)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size, padding=1)
        self.bn3_2 = nn.BatchNorm2d(256)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size, padding=1)
        self.bn3_3 = nn.BatchNorm2d(256)
        
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size, padding=1)
        self.bn4_1 = nn.BatchNorm2d(512)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size, padding=1)
        self.bn4_2 = nn.BatchNorm2d(512)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size, padding=1)
        self.bn4_3 = nn.BatchNorm2d(512)
        
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size, padding=1)
        self.bn5_1 = nn.BatchNorm2d(512)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size, padding=1)
        self.bn5_2 = nn.BatchNorm2d(512)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size, padding=1)
        self.bn5_3 = nn.BatchNorm2d(512)
        
        # Decoder blocks
        self.conv5_3_D = nn.Conv2d(512, 512, kernel_size, padding=1)
        self.bn5_3_D = nn.BatchNorm2d(512)
        self.conv5_2_D = nn.Conv2d(512, 512, kernel_size, padding=1)
        self.bn5_2_D = nn.BatchNorm2d(512)
        self.conv5_1_D = nn.Conv2d(512, 512, kernel_size, padding=1)
        self.bn5_1_D = nn.BatchNorm2d(512)
        
        self.conv4_3_D = nn.Conv2d(512, 512, kernel_size, padding=1)
        self.bn4_3_D = nn.BatchNorm2d(512)
        self.conv4_2_D = nn.Conv2d(512, 512, kernel_size, padding=1)
        self.bn4_2_D = nn.BatchNorm2d(512)
        self.conv4_1_D = nn.Conv2d(512, 256, kernel_size, padding=1)
        self.bn4_1_D = nn.BatchNorm2d(256)
        
        self.conv3_3_D = nn.Conv2d(256, 256, kernel_size, padding=1)
        self.bn3_3_D = nn.BatchNorm2d(256)
        self.conv3_2_D = nn.Conv2d(256, 256, kernel_size, padding=1)
        self.bn3_2_D = nn.BatchNorm2d(256)
        self.conv3_1_D = nn.Conv2d(256, 128, kernel_size, padding=1)
        self.bn3_1_D = nn.BatchNorm2d(128)
        
        self.conv2_2_D = nn.Conv2d(128, 128, kernel_size, padding=1)
        self.bn2_2_D = nn.BatchNorm2d(128)
        self.conv2_1_D = nn.Conv2d(128, 64, kernel_size, padding=1)
        self.bn2_1_D = nn.BatchNorm2d(64)
        
        self.conv1_2_D = nn.Conv2d(64, 64, kernel_size, padding=1)
        self.bn1_2_D = nn.BatchNorm2d(64)
        
        self.classifier = nn.Conv2d(64, n_labels, 1)
        
        # FIX: return_indices=False to return a Tensor, not a tuple
        self.pool = nn.MaxPool2d(2, 2, return_indices=False)
    
    def forward(self, x):
        # Encoder
        x = F.relu(self.bn1_1(self.conv1_1(x)))
        x = F.relu(self.bn1_2(self.conv1_2(x)))
        size1 = x.size()
        x = self.pool(x) # Returns just tensor
        
        x = F.relu(self.bn2_1(self.conv2_1(x)))
        x = F.relu(self.bn2_2(self.conv2_2(x)))
        size2 = x.size()
        x = self.pool(x)
        
        x = F.relu(self.bn3_1(self.conv3_1(x)))
        x = F.relu(self.bn3_2(self.conv3_2(x)))
        x = F.relu(self.bn3_3(self.conv3_3(x)))
        size3 = x.size()
        x = self.pool(x)
        
        x = F.relu(self.bn4_1(self.conv4_1(x)))
        x = F.relu(self.bn4_2(self.conv4_2(x)))
        x = F.relu(self.bn4_3(self.conv4_3(x)))
        size4 = x.size()
        x = self.pool(x)
        
        x = F.relu(self.bn5_1(self.conv5_1(x)))
        x = F.relu(self.bn5_2(self.conv5_2(x)))
        x = F.relu(self.bn5_3(self.conv5_3(x)))
        size5 = x.size()
        x = self.pool(x)
        
        # Decoder - Replace Unpool with Interpolate (Nearest)
        # Block 5
        x = F.interpolate(x, size=size5[2:], mode='nearest')
        x = F.relu(self.bn5_3_D(self.conv5_3_D(x)))
        x = F.relu(self.bn5_2_D(self.conv5_2_D(x)))
        x = F.relu(self.bn5_1_D(self.conv5_1_D(x)))
        
        # Block 4
        x = F.interpolate(x, size=size4[2:], mode='nearest')
        x = F.relu(self.bn4_3_D(self.conv4_3_D(x)))
        x = F.relu(self.bn4_2_D(self.conv4_2_D(x)))
        x = F.relu(self.bn4_1_D(self.conv4_1_D(x)))
        
        # Block 3
        x = F.interpolate(x, size=size3[2:], mode='nearest')
        x = F.relu(self.bn3_3_D(self.conv3_3_D(x)))
        x = F.relu(self.bn3_2_D(self.conv3_2_D(x)))
        x = F.relu(self.bn3_1_D(self.conv3_1_D(x)))
        
        # Block 2
        x = F.interpolate(x, size=size2[2:], mode='nearest')
        x = F.relu(self.bn2_2_D(self.conv2_2_D(x)))
        x = F.relu(self.bn2_1_D(self.conv2_1_D(x)))
        
        # Block 1
        x = F.interpolate(x, size=size1[2:], mode='nearest')
        x = F.relu(self.bn1_2_D(self.conv1_2_D(x)))
        
        x = self.classifier(x)
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
    
    NUM_CLASSES = 21
    BATCH_SIZE = 1
    benchmark_results = []
    
    print("\n" + "="*80)
    print("SEGNET MULTI-RESOLUTION BENCHMARK (TENSOR-ONLY)")
    print("="*80)

    for i, shape in enumerate(input_list):
        c, w, h = shape
        tensor_shape = (c, h, w)
        res_label = f"{w}x{h}"
        
        print(f"\nProcessing Resolution {i+1}/{len(input_list)}: {res_label} (Tensor: {tensor_shape})")
        
        if w * h > 1920 * 1080:
             print("  [Warning] Very high resolution. Memory usage will be high.")

        dummy_input = torch.randn(BATCH_SIZE, *tensor_shape)
        
        # 1. Setup FP32
        model_fp32 = SegNet(input_channels=c, n_labels=NUM_CLASSES).eval().cpu()
        
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
            
            # Static quantization is safe now because inputs/outputs are pure Tensors
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
    print("\n\nSUMMARY - SEGNET (FIXED)")
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
