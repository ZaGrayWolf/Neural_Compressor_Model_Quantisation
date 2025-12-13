#!/usr/bin/env python3
"""
SUMMNet Quantization Benchmark Script
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
# 2. Model Definition (SUMMNet)
# ------------------------------------------------------------------------------

class SUMNet_all_bn(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super(SUMNet_all_bn, self).__init__()

        # Encoder
        self.conv1     = nn.Conv2d(in_ch, 64, 3, padding=1)
        self.bn1       = nn.BatchNorm2d(64, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2     = nn.Conv2d(64, 128, 3, padding=1)
        self.bn2       = nn.BatchNorm2d(128, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
        self.pool1     = nn.MaxPool2d(2, 2, return_indices = True)
        self.conv3a    = nn.Conv2d(128, 256, 3, padding=1)
        self.bn3a      = nn.BatchNorm2d(256, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
        self.conv3b    = nn.Conv2d(256, 256, 3, padding=1)
        self.bn3b      = nn.BatchNorm2d(256, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
        self.pool2     = nn.MaxPool2d(2, 2, return_indices = True)
        self.conv4a    = nn.Conv2d(256, 512, 3, padding=1)
        self.bn4a      = nn.BatchNorm2d(512, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
        self.conv4b    = nn.Conv2d(512, 512, 3, padding=1)
        self.bn4b      = nn.BatchNorm2d(512, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
        self.pool3     = nn.MaxPool2d(2, 2, return_indices = True)
        self.conv5a    = nn.Conv2d(512, 512, 3, padding=1)
        self.bn5a      = nn.BatchNorm2d(512, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
        self.conv5b    = nn.Conv2d(512, 512, 3, padding=1)
        self.bn5b      = nn.BatchNorm2d(512, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
        self.pool4     = nn.MaxPool2d(2, 2, return_indices = True)

        # Decoder
        self.unpool4   = nn.MaxUnpool2d(2, 2)
        self.donv5b    = nn.Conv2d(1024, 512, 3, padding = 1)
        self.dbn5b     = nn.BatchNorm2d(512, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
        self.donv5a    = nn.Conv2d(512, 512, 3, padding = 1)
        self.dbn5a     = nn.BatchNorm2d(512, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
        self.unpool3   = nn.MaxUnpool2d(2, 2)
        self.donv4b    = nn.Conv2d(1024, 512, 3, padding = 1)
        self.dbn4b     = nn.BatchNorm2d(512, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
        self.donv4a    = nn.Conv2d(512, 256, 3, padding = 1)
        self.dbn4a     = nn.BatchNorm2d(256, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
        self.unpool2   = nn.MaxUnpool2d(2, 2)
        self.donv3b    = nn.Conv2d(512, 256, 3, padding = 1)
        self.dbn3b     = nn.BatchNorm2d(256, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
        self.donv3a    = nn.Conv2d(256, 128, 3, padding = 1)
        self.dbn3a     = nn.BatchNorm2d(128, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
        self.unpool1   = nn.MaxUnpool2d(2, 2)
        self.donv2     = nn.Conv2d(128, 64, 3, padding = 1)
        self.dbn2      = nn.BatchNorm2d(64, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
        self.donv1     = nn.Conv2d(128, 32, 3, padding = 1)
        self.dbn1      = nn.BatchNorm2d(32, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
        self.output    = nn.Conv2d(32, out_ch, 1)

    def forward(self, x):
        # Encoder
        conv1_out = self.conv1(x)
        conv1_bn = self.bn1(conv1_out)
        conv1 = F.relu(conv1_bn, inplace=True)
        
        conv2_out = self.conv2(conv1)
        conv2_bn = self.bn2(conv2_out)
        conv2 = F.relu(conv2_bn, inplace=True)
        
        pool1, idxs1 = self.pool1(conv2)

        conv3a_out = self.conv3a(pool1)
        conv3a_bn = self.bn3a(conv3a_out)
        conv3a = F.relu(conv3a_bn, inplace=True)
        
        conv3b_out = self.conv3b(conv3a)
        conv3b_bn = self.bn3b(conv3b_out)
        conv3b = F.relu(conv3b_bn, inplace=True)
        
        pool2, idxs2 = self.pool2(conv3b)

        conv4a_out = self.conv4a(pool2)
        conv4a_bn = self.bn4a(conv4a_out)
        conv4a = F.relu(conv4a_bn, inplace=True)
        
        conv4b_out = self.conv4b(conv4a)
        conv4b_bn = self.bn4b(conv4b_out)
        conv4b = F.relu(conv4b_bn, inplace=True)
        
        pool3, idxs3 = self.pool3(conv4b)

        conv5a_out = self.conv5a(pool3)
        conv5a_bn = self.bn5a(conv5a_out)
        conv5a = F.relu(conv5a_bn, inplace=True)
        
        conv5b_out = self.conv5b(conv5a)
        conv5b_bn = self.bn5b(conv5b_out)
        conv5b = F.relu(conv5b_bn, inplace=True)
        
        pool4, idxs4 = self.pool4(conv5b)

        # Decoder
        # The tracer needs example_inputs to handle .size() dynamic calls
        unpool4 = self.unpool4(pool4, idxs4, output_size=conv5b.size())
        
        donv5b_in = torch.cat([unpool4, conv5b], 1)
        donv5b_out = self.donv5b(donv5b_in)
        donv5b_bn = self.dbn5b(donv5b_out)
        donv5b = F.relu(donv5b_bn, inplace=True)
        
        donv5a_out = self.donv5a(donv5b)
        donv5a_bn = self.dbn5a(donv5a_out)
        donv5a = F.relu(donv5a_bn, inplace=True)

        unpool3 = self.unpool3(donv5a, idxs3, output_size=conv4b.size())
        
        donv4b_in = torch.cat([unpool3, conv4b], 1)
        donv4b_out = self.donv4b(donv4b_in)
        donv4b_bn = self.dbn4b(donv4b_out)
        donv4b = F.relu(donv4b_bn, inplace=True)
        
        donv4a_out = self.donv4a(donv4b)
        donv4a_bn = self.dbn4a(donv4a_out)
        donv4a = F.relu(donv4a_bn, inplace=True)

        unpool2 = self.unpool2(donv4a, idxs2, output_size=conv3b.size())
        
        donv3b_in = torch.cat([unpool2, conv3b], 1)
        donv3b_out = self.donv3b(donv3b_in)
        donv3b_bn = self.dbn3b(donv3b_out)
        donv3b = F.relu(donv3b_bn, inplace=True)
        
        donv3a_out = self.donv3a(donv3b)
        donv3a_bn = self.dbn3a(donv3a_out)
        donv3a = F.relu(donv3a_bn, inplace=True)

        unpool1 = self.unpool1(donv3a, idxs1, output_size=conv2.size())
        
        donv2_out = self.donv2(unpool1)
        donv2_bn = self.dbn2(donv2_out)
        donv2 = F.relu(donv2_bn, inplace=True)
        
        donv1_in = torch.cat([donv2, conv1], 1)
        
        donv1_out = self.donv1(donv1_in)
        donv1_bn = self.dbn1(donv1_out)
        donv1 = F.relu(donv1_bn, inplace=True)

        output = self.output(donv1)
        return output

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
    OUT_CHANNELS = 1 # Segmentation mask
    BATCH_SIZE = 1
    benchmark_results = []
    
    print("\n" + "="*80)
    print("SUMMNET MULTI-RESOLUTION BENCHMARK")
    print("="*80)

    for i, shape in enumerate(input_list):
        c, w, h = shape
        tensor_shape = (c, h, w)
        res_label = f"{w}x{h}"
        
        print(f"\nProcessing Resolution {i+1}/{len(input_list)}: {res_label} (Tensor: {tensor_shape})")
        
        # OOM warning
        if w * h > 1920 * 1080:
             print("  [Warning] Very high resolution. Memory usage will be high.")

        dummy_input = torch.randn(BATCH_SIZE, *tensor_shape)
        
        # 1. Setup FP32
        model_fp32 = SUMNet_all_bn(in_ch=IN_CHANNELS, out_ch=OUT_CHANNELS).eval().cpu()
        
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
                example_inputs=dummy_input # Critical for MaxUnpool2d logic
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
    print("\n\nSUMMARY - SUMMNET")
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