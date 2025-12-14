#!/usr/bin/env python3
"""
JetNet INT8 Quantisation & Benchmark Pipeline
This script integrates the JetNet model into the robust ONNX/ORT profiling pipeline.
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
#  JetNet Model Definition
# ===================================================================

class JetBlock(nn.Module):
    """
    The custom residual block for the JetNet model.
    It includes two 3x3 convolutions with a skip connection.
    The first convolution can be dilated.
    """
    def __init__(self, in_channels, out_channels, dilation=1):
        super(JetBlock, self).__init__()
        # The first convolution uses the specified dilation
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        # The second convolution has a standard dilation of 1
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        # Element-wise addition for the residual connection
        out += residual
        out = self.relu(out)
        return out

class JetNet(nn.Module):
    """
    The main JetNet model for semantic segmentation.
    An encoder-style network with a final classifier and upsampling.
    """
    def __init__(self, num_classes=21):
        super(JetNet, self).__init__()
        # Initial downsampling layer (stride=2)
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        # Sequence of standard JetBlocks
        self.layer2 = nn.Sequential(
            JetBlock(32, 32),
            JetBlock(32, 32)
        )
        # Downsampling (stride=2) followed by dilated JetBlocks
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            JetBlock(64, 64, dilation=2),
            JetBlock(64, 64, dilation=2)
        )
        # Downsampling (stride=2) followed by more dilated JetBlocks
        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            JetBlock(128, 128, dilation=4),
            JetBlock(128, 128, dilation=4)
        )
        # Final 1x1 convolution acts as the classifier
        self.classifier = nn.Conv2d(128, num_classes, kernel_size=1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.classifier(x)
        # Upsample the output to the original input size (total stride = 2*2*2 = 8)
        x = F.interpolate(x, scale_factor=8, mode='bilinear', align_corners=False)
        return x

# ===================================================================
#  Robust Profiling & Quantization Pipeline
# ===================================================================

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
        if not self.activations: return None
        total_mb = sum(a['size_mb'] for a in self.activations)
        peak_mb = max(a['size_mb'] for a in self.activations)
        return {'total_memory_mb': total_mb, 'peak_activation_mb': peak_mb}

class FlopProfiler:
    def __init__(self): self.hooks = []; self.layer_flops = []
    def _conv_flops(self, module, inp, out):
        try:
            x = inp[0]; out_t = out; batch = x.shape[0]
            Cout, H_out, W_out = out_t.shape[1:]
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                Cin = module.in_channels
                kh, kw = module.kernel_size if isinstance(module.kernel_size, tuple) else (module.kernel_size, module.kernel_size)
                groups = module.groups if hasattr(module, 'groups') else 1
                macs = batch * Cout * H_out * W_out * (Cin // groups) * kh * kw
                self.layer_flops.append((module.__class__.__name__, 2 * macs))
        except Exception: pass
    def register_hooks(self, model):
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                self.hooks.append(module.register_forward_hook(lambda m, i, o, mm=module: self._conv_flops(mm, i, o)))
    def remove_hooks(self): [h.remove() for h in self.hooks]; self.hooks = []
    def total_flops(self): return sum(f for _, f in self.layer_flops)

class RandomCalibrationDataReader(CalibrationDataReader if ORT_QUANT_AVAILABLE else object):
    def __init__(self, input_name, input_shape, num_samples=32):
        self.input_name = input_name; self.input_shape = input_shape
        self.num_samples = int(num_samples); self._idx = 0
    def get_next(self):
        if self._idx >= self.num_samples: return None
        self._idx += 1
        return {self.input_name: np.random.randn(*self.input_shape).astype(np.float32)}

def get_model_size_mb(path): return os.path.getsize(path) / (1024**2) if os.path.exists(path) else 0.0

def analyze_onnx_model(onnx_path):
    if not ONNX_AVAILABLE or not os.path.exists(onnx_path): return None
    try:
        model = onnx.load(onnx_path)
        total_params = 0
        for init in model.graph.initializer:
            params = 1
            for dim in init.dims: params *= dim
            total_params += params
        return {'num_nodes': len(model.graph.node), 'num_params': total_params, 'file_size_mb': get_model_size_mb(onnx_path)}
    except Exception: return None

def estimate_onnx_activation_memory(onnx_path, dtype_bits=8):
    if not ONNX_AVAILABLE or not os.path.exists(onnx_path): return None
    try:
        model = onnx.load(onnx_path)
        try: model = onnx.shape_inference.infer_shapes(model)
        except Exception: pass
        tensors = []
        names_seen = set()
        def collect_vi(vi_list):
            for vi in vi_list:
                if vi.name in names_seen: continue
                names_seen.add(vi.name)
                try:
                    dims = [d.dim_value for d in vi.type.tensor_type.shape.dim]
                    if not any(d == 0 for d in dims): tensors.append((vi.name, tuple(dims)))
                except Exception: pass
        collect_vi(model.graph.value_info); collect_vi(model.graph.output); collect_vi(model.graph.input)
        bytes_per_elem = dtype_bits / 8.0
        total_mb = sum(((np.prod(shape) * bytes_per_elem) / (1024**2)) for _, shape in tensors)
        activations = [{'size_mb': (np.prod(shape) * bytes_per_elem) / (1024**2)} for _, shape in tensors]
        activations.sort(key=lambda x: x['size_mb'], reverse=True)
        return {'total_memory_mb': total_mb, 'top_activations': activations[:10]}
    except Exception: return None

def export_to_onnx(model, onnx_path, input_size):
    model.eval()
    if hasattr(model, 'training'): model.training = False
    dummy = torch.randn(*input_size)
    try:
        torch.onnx.export(model, dummy, onnx_path, opset_version=13, input_names=['input'], output_names=['output'],
                          dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}},
                          do_constant_folding=True, export_params=True)
        print(f"[✓] Exported: {onnx_path}")
    except Exception as e: print(f"[✗] Export failed: {e}")

def run_onnxruntime(onnx_path, input_shape, warmup, runs):
    if not ORT_AVAILABLE: return {'mean_ms': 0.0}
    so = ort.SessionOptions(); so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    sess = ort.InferenceSession(onnx_path, sess_options=so, providers=['CPUExecutionProvider'])
    feeds = {'input': np.random.randn(1, *input_shape[1:]).astype(np.float32)}
    for _ in range(warmup): sess.run(None, feeds)
    times = []
    for _ in range(runs):
        t0 = time.perf_counter(); sess.run(None, feeds); times.append((time.perf_counter() - t0) * 1000.0)
    return {'mean_ms': np.mean(times)}

def print_analysis_table(res, flops, params, act_fp32, peak_fp32, lat_fp32, act_int8, peak_int8, lat_int8, size_fp32, size_int8):
    header = (" Input Size | FLOPs (G) | Params (M) | Total Act FP32 (MB) | Peak Act FP32 (MB) | Latency FP32 (ms) | "
              "Total Act INT8 (MB) | Peak Act INT8 (MB) | Latency INT8 (ms) | FP32 Size (MB) | INT8 Size (MB)")
    print("-" * len(header))
    print(header)
    print("-" * len(header))
    print(f" {res:10s} | {flops:9.3f} | {params:10.3f} | {act_fp32:19.2f} | {peak_fp32:18.2f} | {lat_fp32:17.2f} | "
          f"{act_int8:19.2f} | {peak_int8:18.2f} | {lat_int8:17.2f} | {size_fp32:14.2f} | {size_int8:14.2f}")

# ===================================================================
#  Main Loop
# ===================================================================

def process_resolution(res, n_classes):
    c, w, h = res
    input_shape = (1, c, h, w)
    res_str = f"{w}x{h}"
    print(f"\nProcessing {res_str}...")

    # Initialize JetNet
    model = JetNet(num_classes=n_classes)
    model.eval()

    # Profile FLOPs/Params/Acts
    flop_prof = FlopProfiler(); act_prof = ActivationProfiler()
    flop_prof.register_hooks(model); act_prof.register_hooks(model)
    with torch.no_grad(): _ = model(torch.randn(*input_shape))
    flops_g = flop_prof.total_flops() / 1e9
    params_m = sum(p.numel() for p in model.parameters()) / 1e6
    act_stats = act_prof.get_stats()
    act_fp32 = act_stats['total_memory_mb']; peak_fp32 = act_stats['peak_activation_mb']
    flop_prof.remove_hooks(); act_prof.remove_hooks()

    # Export & Quantize
    onnx_path = f"jetnet_{w}x{h}.onnx"; quant_path = f"jetnet_{w}x{h}_int8.onnx"
    export_to_onnx(model, onnx_path, input_shape)
    
    # INT8 Quantization
    if ORT_QUANT_AVAILABLE and os.path.exists(onnx_path):
        try:
            quantize_static(onnx_path, quant_path, RandomCalibrationDataReader('input', input_shape, 64),
                           quant_format=QuantFormat.QDQ, activation_type=QuantType.QInt8, weight_type=QuantType.QInt8)
        except: 
            try: quantize_dynamic(onnx_path, quant_path, weight_type=QuantType.QInt8)
            except: pass

    # Analyze INT8 Memory
    int8_act = 0.0; int8_peak = 0.0
    if os.path.exists(onnx_path): # Estimate based on graph
        est = estimate_onnx_activation_memory(onnx_path, 8)
        if est: int8_act = est['total_memory_mb']; int8_peak = est['top_activations'][0]['size_mb']

    # Benchmark
    lat_fp32 = run_onnxruntime(onnx_path, input_shape, 5, 20)['mean_ms'] if os.path.exists(onnx_path) else 0.0
    lat_int8 = run_onnxruntime(quant_path, input_shape, 5, 20)['mean_ms'] if os.path.exists(quant_path) else 0.0
    size_fp32 = get_model_size_mb(onnx_path)
    size_int8 = get_model_size_mb(quant_path)

    print_analysis_table(res_str, flops_g, params_m, act_fp32, peak_fp32, lat_fp32, int8_act, int8_peak, lat_int8, size_fp32, size_int8)
    return {'res': res_str, 'lat_fp32': lat_fp32, 'lat_int8': lat_int8}

def main():
    print("="*100 + "\nJETNET PIPELINE\n" + "="*100)
    # Resolutions: (Channels, Width, Height)
    resolutions = [
        (3, 640, 360), (3, 1280, 720), (3, 1360, 760),
        (3, 1600, 900), (3, 1920, 1080), (3, 2048, 1152),
        (3, 2560, 1440), (3, 3840, 2160)
    ]
    
    results = []
    for res in resolutions:
        try: results.append(process_resolution(res, n_classes=21))
        except Exception as e: print(f"Failed {res}: {e}")

    print("\n" + "="*60 + "\nSPEEDUP SUMMARY\n" + "="*60)
    for r in results:
        if r['lat_int8'] > 0:
            print(f"{r['res']:10s}: {r['lat_fp32']/r['lat_int8']:.2f}x Speedup")

if __name__ == "__main__":
    main()
