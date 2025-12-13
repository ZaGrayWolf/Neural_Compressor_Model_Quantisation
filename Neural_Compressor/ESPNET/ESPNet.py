#!/usr/bin/env python3
"""
ESPNet Quantization Benchmark Script
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
# 2. Model Definition (ESPNet)
# ------------------------------------------------------------------------------

def conv1x1(in_c, out_c, stride=1):
    return nn.Conv2d(in_c, out_c, 1, stride, bias=False)

class Activation(nn.Module):
    def __init__(self, act_type='prelu'):
        super().__init__()
        self.act = nn.PReLU() if act_type == 'prelu' else nn.ReLU(inplace=True)
    def forward(self, x): return self.act(x)

class ConvBNAct(nn.Sequential):
    def __init__(self, in_c, out_c, k, s=1, d=1, act_type='prelu'):
        padding = d * (k - 1) // 2
        super().__init__(
            nn.Conv2d(in_c, out_c, k, s, padding, dilation=d, bias=False),
            nn.BatchNorm2d(out_c),
            Activation(act_type)
        )

class DeConvBNAct(nn.Sequential):
    def __init__(self, in_c, out_c, act_type='prelu'):
        super().__init__(
            nn.ConvTranspose2d(in_c, out_c, 3, 2, 1, 1, bias=False),
            nn.BatchNorm2d(out_c),
            Activation(act_type)
        )

class ESPModule(nn.Module):
    def __init__(self, in_c, out_c, K=5, ks=3, stride=1, act_type='prelu'):
        super().__init__()
        self.K = K
        self.use_skip = in_c == out_c and stride == 1
        kn = out_c // K
        k1 = out_c - kn * (K - 1)
        self.is_perfect = (k1 == kn)

        if self.is_perfect:
            self.reduce = conv1x1(in_c, kn, stride)
        else:
            self.reduce = nn.ModuleList([conv1x1(in_c, k1, stride),
                                         conv1x1(in_c, kn, stride)])

        self.layers = nn.ModuleList()
        for i in range(K):
            ch = kn if self.is_perfect else (k1 if i == 0 else kn)
            self.layers.append(ConvBNAct(ch, ch, ks, 1, 2**i, act_type))

    def forward(self, x):
        res = x if self.use_skip else None
        if self.is_perfect:
            x_r = self.reduce(x)
            feats = [layer(x_r) for layer in self.layers]
            # Accumulate features
            for i in range(1, len(feats)): 
                feats[i] = feats[i] + feats[i-1]
        else:
            x_r1 = self.reduce[0](x); x_rn = self.reduce[1](x)
            feats = [self.layers[0](x_r1)] + [layer(x_rn) for layer in self.layers[1:]]
            for i in range(2, len(feats)): 
                feats[i] = feats[i] + feats[i-1]

        out = torch.cat(feats, 1)
        if res is not None: out = out + res
        return out

class L2Block(nn.Module):
    def __init__(self, in_c, hid_c, alpha, use_skip, reinforce, act_type):
        super().__init__()
        self.use_skip, self.reinforce = use_skip, reinforce
        ic = in_c + 3 if reinforce else in_c
        self.down = ESPModule(ic, hid_c, stride=2, act_type=act_type)
        self.layers = nn.Sequential(*[ESPModule(hid_c, hid_c, act_type=act_type)
                                      for _ in range(alpha)])

    def forward(self, x, x_input=None):
        x = self.down(x); skip = x if self.use_skip else None
        x = self.layers(x)
        if self.use_skip: x = torch.cat([x, skip], 1)
        if self.reinforce and x_input is not None:
            size = x.shape[2:]
            q = F.interpolate(x_input, size, mode='bilinear', align_corners=False)
            x = torch.cat([x, q], 1)
        return x

class L3Block(nn.Module):
    def __init__(self, in_c, hid_c, out_c, alpha, use_skip, reinforce, use_decoder, act_type):
        super().__init__()
        self.use_skip, self.reinforce = use_skip, reinforce
        ic = in_c + 3 if reinforce else in_c
        self.down = ESPModule(ic, hid_c, stride=2, act_type=act_type)
        self.layers = nn.Sequential(*[ESPModule(hid_c, hid_c, act_type=act_type)
                                      for _ in range(alpha)])
        out_ch = hid_c * 2 if use_skip else hid_c
        self.out_conv = (ConvBNAct(out_ch, out_c, 1, act_type=act_type)
                         if use_decoder else conv1x1(out_ch, out_c))

    def forward(self, x):
        x = self.down(x); skip = x if self.use_skip else None
        x = self.layers(x)
        if self.use_skip: x = torch.cat([x, skip], 1)
        return self.out_conv(x)

class Decoder(nn.Module):
    def __init__(self, num_class, l1_c, l2_c, act_type='prelu'):
        super().__init__()
        self.up3 = DeConvBNAct(num_class, num_class, act_type)
        self.cat2 = ConvBNAct(l2_c, num_class, 1, act_type=act_type)
        self.conv2 = ESPModule(2*num_class, num_class, act_type=act_type)
        self.up2 = DeConvBNAct(num_class, num_class, act_type)
        self.cat1 = ConvBNAct(l1_c, num_class, 1, act_type=act_type)
        self.conv1 = ESPModule(2*num_class, num_class, act_type=act_type)
        self.up1 = DeConvBNAct(num_class, num_class, act_type)

    def forward(self, x, x_l1, x_l2):
        x = self.up3(x); x_l2 = self.cat2(x_l2)
        x = torch.cat([x, x_l2], 1); x = self.conv2(x)
        x = self.up2(x); x_l1 = self.cat1(x_l1)
        x = torch.cat([x, x_l1], 1); x = self.conv1(x)
        return self.up1(x)

class ESPNet(nn.Module):
    def __init__(self, num_class=1, n_channel=3, arch_type='espnet',
                 K=5, alpha2=2, alpha3=8, block_ch=[16,64,128],
                 act_type='prelu'):
        super().__init__()
        # Modified to remove ActivationTrackerMixin for cleaner quantization
        types = ['espnet','espnet-a','espnet-b','espnet-c']
        if arch_type not in types: raise ValueError("Unsupported arch")
        self.arch_type = arch_type
        use_skip = arch_type in ['espnet','espnet-b','espnet-c']
        reinforce = arch_type in ['espnet','espnet-c']
        use_decoder = arch_type=='espnet'
        if arch_type=='espnet-a': block_ch[2]=block_ch[1]

        self.l1 = ConvBNAct(n_channel, block_ch[0], 3, 2, act_type=act_type)
        self.l2 = L2Block(block_ch[0], block_ch[1], alpha2,
                          use_skip, reinforce, act_type)
        l2_out_c = block_ch[1]*2 if use_skip else block_ch[1]
        if reinforce: l2_out_c += 3
        self.l3 = L3Block(l2_out_c, block_ch[2], num_class, alpha3,
                          use_skip, reinforce, use_decoder, act_type)
        self.dec = Decoder(num_class, l1_c=block_ch[0]+n_channel,
                           l2_c=l2_out_c, act_type=act_type) if use_decoder else None

    def forward(self, x):
        inp = x
        x = self.l1(x)

        # L2 (with optional reinforcement)
        x_l1_for_decoder = None
        if self.l2.reinforce:
            size = x.shape[2:]
            q = F.interpolate(inp, size, mode='bilinear', align_corners=False)
            x_cat = torch.cat([x, q], 1)
            x_l1_for_decoder = x_cat 
            x = self.l2(x_cat, inp)
        else:
            x = self.l2(x)

        x_l2_for_decoder = None
        if self.dec:
            x_l2_for_decoder = x 

        # L3 (with optional reinforcement)
        if self.l3.reinforce:
             size = x.shape[2:]
             q = F.interpolate(inp, size, mode='bilinear', align_corners=False)
             x = torch.cat([x, q], 1)

        x = self.l3(x)

        # Decoder or final upsample
        if self.dec:
            x = self.dec(x, x_l1_for_decoder, x_l2_for_decoder)
        else:
            x = F.interpolate(x, inp.shape[2:], mode='bilinear', align_corners=True)
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
    
    NUM_CLASSES = 21 # Matching standard segmentation tasks
    BATCH_SIZE = 1
    benchmark_results = []
    
    print("\n" + "="*80)
    print("ESPNET MULTI-RESOLUTION BENCHMARK")
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
        model_fp32 = ESPNet(num_class=NUM_CLASSES, arch_type='espnet').eval().cpu()
        
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
                example_inputs=dummy_input # Critical for tracing branching logic
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
    print("\n\nSUMMARY - ESPNET")
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