#!/usr/bin/env python3
"""
ENet model with bilinear upsampling replacing MaxUnpool2d and ConvTranspose2d,
compatible with Intel Neural Compressor post-training quantization.
"""

import os
import sys
import time
import warnings
import platform
import gc

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# --- CONFIGURATION & MACOS BACKEND ---
if sys.platform == "darwin":
    torch.backends.quantized.engine = "qnnpack"
    # Prevent JIT fusion errors on ARM
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

# --- Model Definition (UNCHANGED) ---

class InitialBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bias=False, relu=True):
        super().__init__()
        activation = nn.ReLU if relu else nn.PReLU
        self.main_branch = nn.Conv2d(in_channels, out_channels - 3, 3, 2, 1, bias=bias)
        self.ext_branch = nn.MaxPool2d(3, 2, 1)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.out_activation = activation()

    def forward(self, x):
        main = self.main_branch(x)
        ext = self.ext_branch(x)
        out = torch.cat((main, ext), 1)
        out = self.batch_norm(out)
        return self.out_activation(out)

class RegularBottleneck(nn.Module):
    def __init__(self, channels, internal_ratio=4, kernel_size=3, padding=0,
                 dilation=1, asymmetric=False, dropout_prob=0, bias=False, relu=True):
        super().__init__()
        internal_channels = channels // internal_ratio
        activation = nn.ReLU if relu else nn.PReLU
        
        self.ext_conv1 = nn.Sequential(
            nn.Conv2d(channels, internal_channels, 1, 1, bias=bias),
            nn.BatchNorm2d(internal_channels), activation())
        
        if asymmetric:
            self.ext_conv2 = nn.Sequential(
                nn.Conv2d(internal_channels, internal_channels, (kernel_size, 1), 1, (padding, 0), dilation=dilation, bias=bias),
                nn.BatchNorm2d(internal_channels), activation(),
                nn.Conv2d(internal_channels, internal_channels, (1, kernel_size), 1, (0, padding), dilation=dilation, bias=bias),
                nn.BatchNorm2d(internal_channels), activation())
        else:
            self.ext_conv2 = nn.Sequential(
                nn.Conv2d(internal_channels, internal_channels, kernel_size, 1, padding, dilation=dilation, bias=bias),
                nn.BatchNorm2d(internal_channels), activation())
        
        self.ext_conv3 = nn.Sequential(
            nn.Conv2d(internal_channels, channels, 1, 1, bias=bias),
            nn.BatchNorm2d(channels), activation())
        
        self.ext_regul = nn.Dropout2d(p=dropout_prob)
        self.out_activation = activation()

    def forward(self, x):
        main = x
        ext = self.ext_conv1(x)
        ext = self.ext_conv2(ext)
        ext = self.ext_conv3(ext)
        ext = self.ext_regul(ext)
        out = main + ext
        return self.out_activation(out)

class DownsamplingBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, internal_ratio=4,
                 dropout_prob=0, bias=False, relu=True):
        super().__init__()
        internal_channels = in_channels // internal_ratio
        activation = nn.ReLU if relu else nn.PReLU
        
        self.main_max1 = nn.MaxPool2d(2, 2)
        self.ext_conv1 = nn.Sequential(
            nn.Conv2d(in_channels, internal_channels, 2, 2, bias=bias),
            nn.BatchNorm2d(internal_channels), activation())
        self.ext_conv2 = nn.Sequential(
            nn.Conv2d(internal_channels, internal_channels, 3, 1, 1, bias=bias),
            nn.BatchNorm2d(internal_channels), activation())
        self.ext_conv3 = nn.Sequential(
            nn.Conv2d(internal_channels, out_channels, 1, 1, bias=bias),
            nn.BatchNorm2d(out_channels), activation())
        self.ext_regul = nn.Dropout2d(p=dropout_prob)
        self.out_activation = activation()

    def forward(self, x):
        main = self.main_max1(x)
        ext = self.ext_conv1(x)
        ext = self.ext_conv2(ext)
        ext = self.ext_conv3(ext)
        ext = self.ext_regul(ext)
        
        # Use F.pad for FX Tracing compatibility
        ch_main = main.shape[1]
        ch_ext = ext.shape[1]
        pad_channels = ch_ext - ch_main
        
        if pad_channels > 0:
            main = F.pad(main, (0, 0, 0, 0, 0, pad_channels))
        
        out = main + ext
        return self.out_activation(out)

class UpsamplingBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, internal_ratio=4,
                 dropout_prob=0, bias=False, relu=True):
        super().__init__()
        internal_channels = in_channels // internal_ratio
        activation = nn.ReLU if relu else nn.PReLU
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.ext_conv1 = nn.Sequential(
            nn.Conv2d(in_channels, internal_channels, 1, bias=bias),
            nn.BatchNorm2d(internal_channels), activation())
        self.ext_conv2 = nn.Sequential(
            nn.Conv2d(internal_channels, out_channels, 3, 1, 1, bias=bias),
            nn.BatchNorm2d(out_channels), activation())
        self.ext_regul = nn.Dropout2d(p=dropout_prob)
        self.out_activation = activation()
    
    def forward(self, x):
        x = self.upsample(x)
        x = self.ext_conv1(x)
        x = self.ext_conv2(x)
        x = self.ext_regul(x)
        return self.out_activation(x)

class ENet(nn.Module):
    def __init__(self, num_classes, encoder_relu=True, decoder_relu=True):
        super().__init__()
        self.initial_block = InitialBlock(3, 16, relu=encoder_relu)
        self.downsample1_0 = DownsamplingBottleneck(16, 64, dropout_prob=0.01, relu=encoder_relu)
        self.regular1_1 = RegularBottleneck(64, padding=1, dropout_prob=0.01, relu=encoder_relu)
        self.regular1_2 = RegularBottleneck(64, padding=1, dropout_prob=0.01, relu=encoder_relu)
        self.regular1_3 = RegularBottleneck(64, padding=1, dropout_prob=0.01, relu=encoder_relu)
        self.regular1_4 = RegularBottleneck(64, padding=1, dropout_prob=0.01, relu=encoder_relu)
        self.downsample2_0 = DownsamplingBottleneck(64, 128, dropout_prob=0.1, relu=encoder_relu)
        self.regular2_1 = RegularBottleneck(128, padding=1, dropout_prob=0.1, relu=encoder_relu)
        self.dilated2_2 = RegularBottleneck(128, dilation=2, padding=2, dropout_prob=0.1, relu=encoder_relu)
        self.asymmetric2_3 = RegularBottleneck(128, kernel_size=5, padding=2, asymmetric=True, dropout_prob=0.1, relu=encoder_relu)
        self.dilated2_4 = RegularBottleneck(128, dilation=4, padding=4, dropout_prob=0.1, relu=encoder_relu)
        self.regular2_5 = RegularBottleneck(128, padding=1, dropout_prob=0.1, relu=encoder_relu)
        self.dilated2_6 = RegularBottleneck(128, dilation=8, padding=8, dropout_prob=0.1, relu=encoder_relu)
        self.asymmetric2_7 = RegularBottleneck(128, kernel_size=5, asymmetric=True, padding=2, dropout_prob=0.1, relu=encoder_relu)
        self.dilated2_8 = RegularBottleneck(128, dilation=16, padding=16, dropout_prob=0.1, relu=encoder_relu)
        self.regular3_0 = self.regular2_1
        self.dilated3_1 = self.dilated2_2
        self.asymmetric3_2 = self.asymmetric2_3
        self.dilated3_3 = self.dilated2_4
        self.regular3_4 = self.regular2_5
        self.dilated3_5 = self.dilated2_6
        self.asymmetric3_6 = self.asymmetric2_7
        self.dilated3_7 = self.dilated2_8
        self.upsample4_0 = UpsamplingBottleneck(128, 64, dropout_prob=0.1, relu=decoder_relu)
        self.regular4_1 = RegularBottleneck(64, padding=1, dropout_prob=0.1, relu=decoder_relu)
        self.regular4_2 = RegularBottleneck(64, padding=1, dropout_prob=0.1, relu=decoder_relu)
        self.upsample5_0 = UpsamplingBottleneck(64, 16, dropout_prob=0.1, relu=decoder_relu)
        self.regular5_1 = RegularBottleneck(16, padding=1, dropout_prob=0.01, relu=decoder_relu)
        self.final_conv = nn.Conv2d(16, num_classes, kernel_size=1)
    
    def forward(self, x):
        x = self.initial_block(x)
        x = self.downsample1_0(x)
        x = self.regular1_1(x); x = self.regular1_2(x); x = self.regular1_3(x); x = self.regular1_4(x)
        x = self.downsample2_0(x)
        x = self.regular2_1(x); x = self.dilated2_2(x); x = self.asymmetric2_3(x); x = self.dilated2_4(x)
        x = self.regular2_5(x); x = self.dilated2_6(x); x = self.asymmetric2_7(x); x = self.dilated2_8(x)
        x = self.regular3_0(x); x = self.dilated3_1(x); x = self.asymmetric3_2(x); x = self.dilated3_3(x)
        x = self.regular3_4(x); x = self.dilated3_5(x); x = self.asymmetric3_6(x); x = self.dilated3_7(x)
        x = self.upsample4_0(x)
        x = self.regular4_1(x); x = self.regular4_2(x)
        x = self.upsample5_0(x)
        x = self.regular5_1(x)
        x = self.final_conv(x)
        return x

# -- Dummy calibration dataset --

class DummyCalibrationDataset(Dataset):
    def __init__(self, input_shape, num_samples=10):
        self.num_samples = num_samples
        self.input_shape = input_shape
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return torch.randn(self.input_shape)

# -- Profiling and Latency Measurement --

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

    num_runs = 15
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

# -- Main Benchmark Loop --

if __name__ == "__main__":
    if not INC_AVAILABLE:
        print("FATAL ERROR: 'neural-compressor' is not installed.")
        sys.exit(1)
        
    print(f"Running on Platform: {sys.platform}")
    print(f"Quantization Engine: {torch.backends.quantized.engine}")
    
    # Provided Input Sizes format: (Channels, Width, Height)
    input_list = [
        (3, 640, 360), 
        (3, 1280, 720), 
        (3, 1360, 760),
        (3, 1600, 900), 
        (3, 1920, 1080), 
        (3, 2048, 1152),
        (3, 2560, 1440), 
        (3, 3840, 2160)
    ]
    
    NUM_CLASSES = 21
    BATCH_SIZE = 1
    benchmark_results = []
    
    print("\n" + "="*80)
    print("STARTING MULTI-RESOLUTION BENCHMARK")
    print("="*80)

    for i, shape in enumerate(input_list):
        c, w, h = shape
        # PyTorch expects (Batch, Channels, Height, Width)
        # So we swap h and w from the tuple for the tensor creation
        tensor_shape = (c, h, w) 
        res_label = f"{w}x{h}"
        
        print(f"\nProcessing Resolution {i+1}/{len(input_list)}: {res_label} (Tensor: {tensor_shape})")
        
        dummy_input = torch.randn(BATCH_SIZE, *tensor_shape)
        
        # 1. Setup FP32 Model
        model_fp32 = ENet(num_classes=NUM_CLASSES, encoder_relu=True).eval().cpu()
        
        # 2. Measure FP32 Latency
        print("  -> Measuring FP32 Latency...")
        try:
            fp32_lat = measure_latency(model_fp32, dummy_input, model_name="FP32", verbose=True)
        except Exception as e:
            print(f"  -> FP32 Failed: {e}")
            fp32_lat = 0.0

        # 3. Quantize
        print("  -> Quantizing...")
        int8_lat = 0.0
        
        # Only proceed to quantize if FP32 worked
        if fp32_lat > 0:
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
                # Suppress INC logs for cleaner output
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
                print(f"  -> Quantization/Inference Failed (likely OOM for large res): {e}")
                int8_lat = 0.0
        
        # 4. Record Results
        benchmark_results.append({
            "size": res_label,
            "fp32": fp32_lat,
            "int8": int8_lat
        })
        
        # 5. Cleanup to free memory
        del model_fp32
        if 'quantized_model' in locals(): del quantized_model
        del dummy_input
        gc.collect()

    # -- Print Final Table --
    print("\n\nSUMMARY - ALL RESOLUTIONS")
    print("="*105)
    # Define column widths
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