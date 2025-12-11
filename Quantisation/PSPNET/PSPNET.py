#!/usr/bin/env python3
"""
PSPNet ONNX/INT8 Quantisation & Benchmark Pipeline (CPU-optimised)
This script integrates the PSPNet model (ResNet50 backbone) into the 
standard profiling and quantization pipeline.
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore")

# ----------  ONNX / ORT imports  ----------
try:
    import onnxruntime as ort
    ORT_AVAILABLE = True
except Exception:
    ort = None
    ORT_AVAILABLE = False

try:
    import onnx
    ONNX_AVAILABLE = True
except Exception:
    onnx = None
    ONNX_AVAILABLE = False

try:
    from onnxruntime.quantization import (
        quantize_static, CalibrationDataReader, QuantType, QuantFormat,
        quantize_dynamic
    )
    ORT_QUANT_AVAILABLE = True
except Exception:
    ORT_QUANT_AVAILABLE = False

# ===================================================================
#  PSPNet Model Definition (ResNet Backbone + PSP Head)
# ===================================================================

# --- ResNet Backbone Implementation ---
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=dilation, bias=False, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=dilation, bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
    def forward(self, x):
        identity = x
        out = self.conv1(x); out = self.bn1(out); out = self.relu(out)
        out = self.conv2(out); out = self.bn2(out)
        if self.downsample is not None: identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=dilation, bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
    def forward(self, x):
        identity = x
        out = self.conv1(x); out = self.bn1(out); out = self.relu(out)
        out = self.conv2(out); out = self.bn2(out); out = self.relu(out)
        out = self.conv3(out); out = self.bn3(out)
        if self.downsample is not None: identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    # Simplified ResNet for PSPNet feature extraction
    def __init__(self, block, layers):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # Standard ResNet layers
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        # Modified layers for PSPNet: stride=1, dilation=2/4 in layer3/4
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4)

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, dilation=dilation)) # Use dilation here
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            # Subsequent blocks in the layer also need dilation if the layer has it
            current_dilation = dilation if stride == 1 else 1 
            layers.append(block(self.inplanes, planes, dilation=current_dilation))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x); x = self.bn1(x); x = self.relu(x); x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x_layer3 = self.layer3(x) # Output for Aux loss
        x = self.layer4(x_layer3) # Main feature output
        return x, x_layer3 # Return both for main and aux branches

def resnet50_pspnet(pretrained=False):
    # Uses Bottleneck and ResNet50 layer configuration [3, 4, 6, 3]
    # Note: Dilations are applied within the ResNet class for PSPNet
    model = ResNet(Bottleneck, [3, 4, 6, 3]) 
    # Load pretrained weights if needed (omitted here)
    return model

# --- PSPNet Components ---
class ConvBlock(nn.Module):
    # Basic Conv -> BN -> ReLU block
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=False):
        super(ConvBlock, self).__init__()
        # Recalculate padding based on kernel and dilation for 'same' padding effect
        padding = (kernel_size + (kernel_size - 1) * (dilation - 1) -1) // 2 
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True) # Use inplace ReLU
        )
    def forward(self, x):
        return self.conv(x)

def upsample(input, size=None, scale_factor=None, align_corners=False):
    # Wrapper for F.interpolate
    return F.interpolate(input, size=size, scale_factor=scale_factor, mode='bilinear', align_corners=align_corners)

class PyramidPooling(nn.Module):
    def __init__(self, in_channels, pool_sizes=[1, 2, 3, 6]):
        super(PyramidPooling, self).__init__()
        self.pool_sizes = pool_sizes
        self.out_channels_per_pool = in_channels // len(pool_sizes)
        
        self.pools = nn.ModuleList()
        for size in pool_sizes:
            self.pools.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(size),
                ConvBlock(in_channels, self.out_channels_per_pool, kernel_size=1) # 1x1 Conv
            ))

    def forward(self, x):
        features = [x] # Start with original features
        input_size = x.size()[-2:]
        for pool_module in self.pools:
            pooled_features = pool_module(x)
            upsampled_features = upsample(pooled_features, size=input_size, align_corners=False) # Use consistent align_corners
            features.append(upsampled_features)
        
        # Concatenate original features with upsampled pooled features
        out = torch.cat(features, dim=1)
        return out

class PSPNet(nn.Module):
    def __init__(self, n_classes=21, backbone_channels=2048, aux=True):
        super(PSPNet, self).__init__()
        self.aux = aux
        
        self.backbone = resnet50_pspnet(pretrained=False) # Use PSPNet variant
        
        # Pyramid Pooling Module
        self.pyramid_pooling = PyramidPooling(backbone_channels)
        # Input to decoder is backbone_channels + 4 * (backbone_channels // 4) = 2 * backbone_channels
        decoder_in_channels = backbone_channels * 2 
        
        # Final Decoder Convolution
        self.decoder = nn.Sequential(
            ConvBlock(decoder_in_channels, 512, kernel_size=3), # Conv 3x3
            nn.Dropout(0.1),
            nn.Conv2d(512, n_classes, kernel_size=1) # Final 1x1 Conv
        )

        # Auxiliary Branch (optional)
        if self.aux:
            self.aux_branch = nn.Sequential(
                ConvBlock(backbone_channels // 2, 256, kernel_size=3), # Input from layer3 (1024 channels)
                nn.Dropout(0.1),
                nn.Conv2d(256, n_classes, kernel_size=1)
            )

    def forward(self, x):
        original_size = x.size()[-2:]
        
        # Backbone feature extraction
        features, aux_features = self.backbone(x) # features=layer4, aux_features=layer3
        
        # Main Branch
        x = self.pyramid_pooling(features)
        x = self.decoder(x)
        x = upsample(x, size=original_size, align_corners=False) # Upsample final output

        # Auxiliary Branch (if enabled) - Not returned for ONNX export
        if self.aux and self.training: # Aux branch only used during training
            aux_out = self.aux_branch(aux_features)
            aux_out = upsample(aux_out, size=original_size, align_corners=False)
            # In a typical training setup, you'd return both: return x, aux_out
            # For inference/ONNX, we only need the main output 'x'
            return x # Modified for ONNX compatibility
        else:
            return x # Return only main output for inference/ONNX

# ===================================================================
# 3. ONNX/ORT Pipeline Utilities (Standard)
# ===================================================================

# --- Profilers ---
class ActivationProfiler:
    def __init__(self): self.activations = []; self.hooks = []
    def register_hooks(self, model):
        def hook_fn(module, input, output):
            if isinstance(output, torch.Tensor):
                size_bytes = output.element_size() * output.nelement()
                self.activations.append({'module': module.__class__.__name__, 'shape': tuple(output.shape), 'size_mb': size_bytes / (1024**2)})
        for name, module in model.named_modules():
            if len(list(module.children())) == 0: self.hooks.append(module.register_forward_hook(hook_fn))
    def clear(self): self.activations = []
    def remove_hooks(self): [h.remove() for h in self.hooks]; self.hooks = []
    def get_stats(self):
        if not self.activations: return {'total_memory_mb': 0.0, 'peak_activation_mb': 0.0}
        total_mb = sum(a['size_mb'] for a in self.activations)
        peak_mb = max(a['size_mb'] for a in self.activations) 
        return {'total_memory_mb': total_mb, 'peak_activation_mb': peak_mb}

class FlopProfiler:
    def __init__(self): self.hooks = []; self.layer_flops = []
    def _conv_flops(self, module, inp, out):
        try:
            x = inp[0]; out_t = out
            batch, Cout, H_out, W_out = out_t.shape
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                Cin, kh, kw, groups = module.in_channels, module.kernel_size[0], module.kernel_size[1], module.groups
                macs = batch * Cout * H_out * W_out * (Cin // groups) * kh * kw
                self.layer_flops.append((module.__class__.__name__, 2 * macs))
        except Exception: pass
    def register_hooks(self, model):
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                self.hooks.append(module.register_forward_hook(lambda m, i, o, mm=module: self._conv_flops(mm, i, o)))
    def remove_hooks(self): [h.remove() for h in self.hooks]; self.hooks = []
    def total_flops(self): return sum(f for _, f in self.layer_flops)

# --- General Utils ---
def get_model_parameters(model): return sum(p.numel() for p in model.parameters())
def get_model_size_mb(path): return os.path.getsize(path) / (1024**2) if os.path.exists(path) else 0.0

# --- Calibration ---
class RandomCalibrationDataReader(CalibrationDataReader if ORT_QUANT_AVAILABLE else object):
    def __init__(self, input_name, input_shape, num_samples=32):
        self.input_name = input_name; self.input_shape = input_shape
        self.num_samples = int(num_samples); self._idx = 0
    def get_next(self):
        if self._idx >= self.num_samples: return None
        self._idx += 1
        data = np.random.randn(*self.input_shape).astype(np.float32)
        return {self.input_name: data}

# --- ONNX Export / Quant / Benchmark ---
def export_to_onnx(model, onnx_path, input_size, input_name='input', opset=13):
    model.eval()
    dummy = torch.randn(*input_size)
    # Ensure model is not in training mode to disable aux output if logic exists
    if hasattr(model, 'training'): model.training = False 
    
    print("[*] Exporting ONNX...")
    try:
        with torch.no_grad():
            torch.onnx.export(
                model, dummy, onnx_path, opset_version=opset,
                input_names=[input_name], output_names=['output'],
                dynamic_axes={input_name: {0: 'batch'}, 'output': {0: 'batch'}},
                do_constant_folding=True, export_params=True, keep_initializers_as_inputs=False
            )
        print(f"[✓] Exported ONNX -> {onnx_path}")
    except Exception as e:
        print(f"[✗] ONNX export failed: {e}")
        # Clean up failed export file if it exists
        if os.path.exists(onnx_path): os.remove(onnx_path)

def try_onnx_quant_static(onnx_in, onnx_out, input_name, input_shape, calib_reader, per_channel=False):
    if not ORT_QUANT_AVAILABLE: raise RuntimeError("ONNX Runtime quantization not available")
    print("[*] Running ONNX Runtime static quantization...")
    qformat = QuantFormat.QDQ if hasattr(QuantFormat, 'QDQ') else None
    quantize_static(model_input=onnx_in, model_output=onnx_out, calibration_data_reader=calib_reader,
                    activation_type=QuantType.QInt8, weight_type=QuantType.QInt8, per_channel=per_channel, quant_format=qformat)

def try_onnx_quant_dynamic(onnx_in, onnx_out):
    if not ORT_QUANT_AVAILABLE: raise RuntimeError("ONNX Runtime quantization not available")
    print("[*] Running ONNX dynamic quantization...")
    quantize_dynamic(model_input=onnx_in, model_output=onnx_out, weight_type=QuantType.QInt8)
    print("[✓] Dynamic quantization produced:", onnx_out)

def run_onnxruntime(onnx_path, input_name, input_shape, warmup=10, runs=100):
    if not ORT_AVAILABLE: raise RuntimeError("onnxruntime not available")
    providers = ['CPUExecutionProvider']
    so = ort.SessionOptions(); so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    sess = ort.InferenceSession(onnx_path, sess_options=so, providers=providers)
    batch_shape = (1,) + tuple(input_shape[1:]) 
    x = np.random.randn(*batch_shape).astype(np.float32)
    feeds = {input_name: x}
    for _ in range(warmup): _ = sess.run(None, feeds)
    times = []
    for _ in range(runs):
        t0 = time.perf_counter(); _ = sess.run(None, feeds); t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)
    return {'mean_ms': np.mean(times), 'std_ms': np.std(times), 'min_ms': np.min(times), 'p50_ms': np.percentile(times, 50),
            'p95_ms': np.percentile(times, 95), 'p99_ms': np.percentile(times, 99), 'max_ms': np.max(times)}

# --- Table Printer ---
def print_analysis_table(input_size_str, flops_g, params_m, total_act_mb_fp32, peak_act_mb_fp32, lat_fp32,
                         total_act_mb_int8, peak_act_mb_int8, lat_int8, fp32_size_mb, int8_size_mb):
    header = (" Input Size | FLOPs (G) | Params (M) | Total Act FP32 (MB) | Peak Act FP32 (MB) | Latency FP32 (ms) | "
              "Total Act INT8 (MB) | Peak Act INT8 (MB) | Latency INT8 (ms) | FP32 Size (MB) | INT8 Size (MB)")
    print(header)
    print("-" * len(header))
    print(f" {input_size_str:10s} | {flops_g:9.3f} | {params_m:10.3f} | {total_act_mb_fp32:19.2f} | "
          f"{peak_act_mb_fp32:18.2f} | {lat_fp32:17.2f} | {total_act_mb_int8:19.2f} | "
          f"{peak_act_mb_int8:18.2f} | {lat_int8:17.2f} | {fp32_size_mb:14.2f} | {int8_size_mb:14.2f}")

# ===================================================================
# 4. Main Pipeline Execution
# ===================================================================

def process_resolution(resolution, num_classes, calib_size=64, per_channel=False, warmup=10, runs=100):
    c, h, w = resolution  # (C, H, W) format
    input_shape = (1, c, h, w) # PyTorch/ONNX expect (B, C, H, W)
    input_size_str = f"{w}x{h}"

    print(f"\n{'='*80}")
    print(f"Processing resolution: {input_size_str} (Input Shape: {input_shape})")
    print(f"{'='*80}")

    # Use aux=False for inference/benchmarking
    model = PSPNet(n_classes=num_classes, aux=False) 
    model.eval()

    total_params = get_model_parameters(model)
    params_m = total_params / 1e6

    # 1. FLOPs and Activation Memory Profiling (PyTorch)
    flop_prof = FlopProfiler()
    act_prof = ActivationProfiler()
    flop_prof.register_hooks(model)
    act_prof.register_hooks(model)
    with torch.no_grad(): _ = model(torch.randn(*input_shape))
    flops_total = flop_prof.total_flops(); flop_prof.remove_hooks()
    act_stats = act_prof.get_stats(); act_prof.remove_hooks()
    flops_g = flops_total / 1e9
    total_act_mb_fp32 = act_stats['total_memory_mb']
    peak_act_mb_fp32 = act_stats['peak_activation_mb']
    
    print(f"[✓] Profiling: FLOPs (Conv/Deconv): {flops_g:.3f} G | Params: {params_m:.3f} M | Peak Act (Hooked): {peak_act_mb_fp32:.2f} MB")

    # 2. ONNX Export
    onnx_path = f"pspnet_{w}x{h}.onnx"
    quant_path = f"pspnet_{w}x{h}_int8.onnx"
    export_to_onnx(model, onnx_path, input_shape) 
    fp32_size_mb = get_model_size_mb(onnx_path)

    # 3. ONNX Quantization
    did_onnx_quant = False
    int8_size_mb = 0.0
    int8_est_total_mb = total_act_mb_fp32 * (8/32) # Simple estimation
    int8_est_peak_mb = peak_act_mb_fp32 * (8/32)

    if ORT_QUANT_AVAILABLE and os.path.exists(onnx_path) and fp32_size_mb > 0:
        calib_reader = RandomCalibrationDataReader('input', input_shape, num_samples=calib_size)
        try:
            try_onnx_quant_static(onnx_path, quant_path, 'input', input_shape, calib_reader, per_channel=per_channel)
            did_onnx_quant = os.path.exists(quant_path)
        except Exception as e:
            print(f"[!] Static quant failed: {e}. Trying dynamic.")
            try:
                try_onnx_quant_dynamic(onnx_path, quant_path)
                did_onnx_quant = os.path.exists(quant_path)
            except Exception as e_dyn: print(f"[✗] Dynamic quant also failed: {e_dyn}")
        int8_size_mb = get_model_size_mb(quant_path) if did_onnx_quant else 0.0
    elif not os.path.exists(onnx_path) or fp32_size_mb == 0:
        print("[!] Skipping quantization due to failed FP32 ONNX export.")

    # 4. ONNX Runtime Benchmarking
    lat_fp32 = 0.0
    if os.path.exists(onnx_path) and fp32_size_mb > 0 and ORT_AVAILABLE:
        try: lat_fp32 = run_onnxruntime(onnx_path, 'input', input_shape, warmup=warmup, runs=runs)['mean_ms']
        except Exception as e: print(f"[!] FP32 ONNX benchmark failed: {e}")

    lat_int8 = 0.0
    if did_onnx_quant and os.path.exists(quant_path) and ORT_AVAILABLE:
        try: lat_int8 = run_onnxruntime(quant_path, 'input', input_shape, warmup=warmup, runs=runs)['mean_ms']
        except Exception as e: print(f"[!] INT8 ONNX benchmark failed: {e}")

    print(f"\nRESULTS FOR {input_size_str}")
    print_analysis_table(input_size_str, flops_g, params_m, total_act_mb_fp32, peak_act_mb_fp32, lat_fp32,
                         int8_est_total_mb, int8_est_peak_mb, lat_int8, fp32_size_mb, int8_size_mb)

    return {'resolution': input_size_str, 'flops_g': flops_g, 'params_m': params_m, 'total_act_fp32_mb': total_act_mb_fp32,
            'peak_act_fp32_mb': peak_act_mb_fp32, 'lat_fp32_ms': lat_fp32, 'total_act_int8_mb': int8_est_total_mb,
            'peak_act_int8_mb': int8_est_peak_mb, 'lat_int8_ms': lat_int8, 'fp32_size_mb': fp32_size_mb, 'int8_size_mb': int8_size_mb}

def main():
    print("\n" + "="*80)
    print("PSPNET (RESNET50 BACKBONE) INT8 QUANTIZATION & BENCHMARK PIPELINE")
    print("="*80)

    # (C, H, W) format
    resolutions = [(3, 360, 640), (3, 720, 1280), (3, 760, 1360),
                   (3, 900, 1600), (3, 1080, 1920), (3, 1152, 2048),
                   (3, 1440, 2560), (3, 2160, 3840)] 
    
    NUM_CLASSES = 21 # e.g., Pascal VOC
    calib_size = 64
    warmup = 10
    runs = 100

    print(f"Resolutions to test: {len(resolutions)}")
    print(f"Output Classes: {NUM_CLASSES}")
    print(f"Benchmark runs: {runs} (warmup: {warmup})")
    print("="*80 + "\n")

    all_results = []
    for res in resolutions:
        try:
            result = process_resolution(res, num_classes=NUM_CLASSES, calib_size=calib_size, warmup=warmup, runs=runs)
            all_results.append(result)
        except Exception as e:
            print(f"[✗] CRITICAL FAILURE processing {res} (HxW: {res[1]}x{res[2]}): {e}")
            import traceback; traceback.print_exc() 
            continue # Attempt to continue with next resolution

    print("\n" + "="*140)
    print("SUMMARY - ALL RESOLUTIONS (PSPNet)")
    print("="*140)
    header = (" Input Size | FLOPs (G) | Params (M) | Total Act FP32 (MB) | Peak Act FP32 (MB) | Latency FP32 (ms) | "
              "Total Act INT8 (MB) | Peak Act INT8 (MB) | Latency INT8 (ms) | FP32 Size (MB) | INT8 Size (MB)")
    print(header)
    print("-" * 140)
    for r in all_results:
        print(f" {r['resolution']:10s} | {r['flops_g']:9.3f} | {r['params_m']:10.3f} | "
              f"{r['total_act_fp32_mb']:19.2f} | {r['peak_act_fp32_mb']:18.2f} | {r['lat_fp32_ms']:17.2f} | "
              f"{r['total_act_int8_mb']:19.2f} | {r['peak_act_int8_mb']:18.2f} | {r['lat_int8_ms']:17.2f} | "
              f"{r['fp32_size_mb']:14.2f} | {r['int8_size_mb']:14.2f}")

    print("\n" + "="*140)
    print("BENCHMARK COMPLETE!")
    print("="*140)

    print("\nSPEEDUP AND SIZE ANALYSIS:")
    print("-" * 60)
    for r in all_results:
        speedup = r['lat_fp32_ms'] / r['lat_int8_ms'] if r['lat_int8_ms'] > 0 else 0
        compression = r['fp32_size_mb'] / r['int8_size_mb'] if r['int8_size_mb'] > 0 else 0
        print(f"{r['resolution']:10s}: Speedup: {speedup:5.2f}x | Size Compression: {compression:5.2f}x")

    print("\n" + "="*80) 
    print("All ONNX models saved in current directory:")
    print("  - pspnet_<width>x<height>.onnx (FP32)")
    print("  - pspnet_<width>x<height>_int8.onnx (INT8)")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()