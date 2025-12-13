#!/usr/bin/env python3
"""
LinkNet Quantization Benchmark Script
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
from torchvision.models import resnet18, ResNet18_Weights

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
# 2. Model Definition (LinkNet)
# ------------------------------------------------------------------------------

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.deconv = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 
                                         kernel_size=3, stride=2, padding=1, 
                                         output_padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nn.ReLU(inplace=True)
        
        self.conv3 = nn.Conv2d(in_channels // 4, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu3 = nn.ReLU(inplace=True)

    def forward(self, x, target_size=None):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        
        x = self.deconv(x)
        x = self.bn2(x)
        x = self.relu2(x)
        
        if target_size is not None:
            x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        return x

class LinkNet(nn.Module):
    def __init__(self, num_classes=19):
        super(LinkNet, self).__init__()
        
        # Encoder (ResNet18)
        # Using weights=None to speed up instantiation for profiling
        resnet = resnet18(weights=None)
        
        self.encoder_conv1 = resnet.conv1
        self.encoder_bn1 = resnet.bn1
        self.encoder_relu = resnet.relu
        self.encoder_maxpool = resnet.maxpool
        
        self.encoder_layer1 = resnet.layer1
        self.encoder_layer2 = resnet.layer2
        self.encoder_layer3 = resnet.layer3
        self.encoder_layer4 = resnet.layer4
        
        # Decoder
        self.decoder_layer4 = DecoderBlock(512, 256)
        self.decoder_layer3 = DecoderBlock(256, 128)
        self.decoder_layer2 = DecoderBlock(128, 64)
        self.decoder_layer1 = DecoderBlock(64, 64)
        
        # Final Upsampling
        self.final_deconv1 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        self.final_bn1 = nn.BatchNorm2d(32)
        self.final_relu1 = nn.ReLU(inplace=True)
        
        self.final_conv = nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False)
        self.final_bn2 = nn.BatchNorm2d(32)
        self.final_relu2 = nn.ReLU(inplace=True)
        
        self.final_deconv2 = nn.ConvTranspose2d(32, num_classes, kernel_size=2, stride=2, bias=False)

    def forward(self, x):
        input_size = x.shape[2:]
        
        # Encoder
        e0 = self.encoder_conv1(x)
        e0 = self.encoder_bn1(e0)
        e0 = self.encoder_relu(e0)
        e0_pool = self.encoder_maxpool(e0)
        
        e1 = self.encoder_layer1(e0_pool)
        e2 = self.encoder_layer2(e1)
        e3 = self.encoder_layer3(e2)
        e4 = self.encoder_layer4(e3)
        
        # Decoder with Additive Skip Connections
        
        d4 = self.decoder_layer4(e4, target_size=e3.shape[2:]) + e3
        d3 = self.decoder_layer3(d4, target_size=e2.shape[2:]) + e2
        d2 = self.decoder_layer2(d3, target_size=e1.shape[2:]) + e1
        d1 = self.decoder_layer1(d2, target_size=e0.shape[2:]) + e0
        
        # Final
        f1 = self.final_deconv1(d1)
        f1 = self.final_bn1(f1)
        f1 = self.final_relu1(f1)
        
        f2 = self.final_conv(f1)
        f2 = self.final_bn2(f2)
        f2 = self.final_relu2(f2)
        
        out = self.final_deconv2(f2)
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
    
    NUM_CLASSES = 19
    BATCH_SIZE = 1
    benchmark_results = []
    
    print("\n" + "="*80)
    print("LINKNET MULTI-RESOLUTION BENCHMARK")
    print("="*80)

    for i, shape in enumerate(input_list):
        c, w, h = shape
        tensor_shape = (c, h, w)
        res_label = f"{w}x{h}"
        
        print(f"\nProcessing Resolution {i+1}/{len(input_list)}: {res_label} (Tensor: {tensor_shape})")
        
        # OOM warning for high res
        if w * h > 2560 * 1440:
             print("  [Warning] Very high resolution. Memory usage will be high.")

        dummy_input = torch.randn(BATCH_SIZE, *tensor_shape)
        
        # 1. Setup FP32
        model_fp32 = LinkNet(num_classes=NUM_CLASSES).eval().cpu()
        
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
                example_inputs=dummy_input # Critical for FX tracing of LinkNet Add nodes
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
    print("\n\nSUMMARY - LINKNET")
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