#!/usr/bin/env python3
"""
ERFNet ONNX/INT8 Quantization & Benchmark Pipeline (Fixed)
- Integrates corrected ERFNet model (BN channel fix).
- Uses Whitelist Quantization to prevent CPU crashes on Transposed Convolutions.
- Supports full resolution sweep.
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
import traceback

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
#  ERFNet Model Definition (Fixed)
# ===================================================================

class ConvBNAct(nn.Sequential):
    def __init__(self, in_c, out_c, k, s=1, ph=0, pw=0, dil=1):
        super().__init__(
            nn.Conv2d(in_c, out_c, k, stride=s, padding=(ph, pw), dilation=dil, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

class DeConvBNAct(nn.Sequential):
    def __init__(self, in_c, out_c):
        super().__init__(
            nn.ConvTranspose2d(in_c, out_c, 3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

class InitialBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c - in_c, 3, stride=2, padding=1, bias=False)
        self.pool = nn.MaxPool2d(3, stride=2, padding=1)
        self.bn   = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        c = self.conv(x)
        p = self.pool(x)
        out = torch.cat([c, p], dim=1)
        return self.relu(self.bn(out))

class NonBt1DBlock(nn.Module):
    def __init__(self, ch, dilation=1):
        super().__init__()
        self.conv = nn.Sequential(
            ConvBNAct(ch, ch, (3,1), ph=1, pw=0),
            ConvBNAct(ch, ch, (1,3), ph=0, pw=1),
            ConvBNAct(ch, ch, (3,1), ph=dilation, pw=0, dil=dilation),
            ConvBNAct(ch, ch, (1,3), ph=0, pw=dilation, dil=dilation)
        )
        self.bnact = nn.Sequential(nn.BatchNorm2d(ch), nn.ReLU(True))

    def forward(self, x):
        res = x
        x = self.conv(x)
        return self.bnact(x + res)

class ERFNet(nn.Module):
    def __init__(self, num_class=19):
        super().__init__()
        self.l1  = InitialBlock(3, 16) 
        self.l2  = InitialBlock(16, 64)
        self.l3  = nn.Sequential(*[NonBt1DBlock(64) for _ in range(5)])
        self.l8  = InitialBlock(64, 128)
        self.l9  = nn.Sequential(*[NonBt1DBlock(128, d) for d in [2,4,8,16,2,4,8,16]])
        self.l17 = DeConvBNAct(128, 64)
        self.l18 = nn.Sequential(*[NonBt1DBlock(64) for _ in range(2)])
        self.l20 = DeConvBNAct(64, 16)
        self.l21 = nn.Sequential(*[NonBt1DBlock(16) for _ in range(2)])
        self.l23 = DeConvBNAct(16, num_class)

    def forward(self, x):
        x = self.l1(x); x = self.l2(x)
        x = self.l3(x); x = self.l8(x)
        x = self.l9(x); x = self.l17(x)
        x = self.l18(x); x = self.l20(x)
        x = self.l21(x); return self.l23(x)

# ===================================================================
#  Pipeline Utils
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
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                Cout, H_out, W_out = out_t.shape[1:]
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
        if activations: activations.sort(key=lambda x: x['size_mb'], reverse=True)
        return {'total_memory_mb': total_mb, 'peak_tensor_mb': activations[0]['size_mb'] if activations else 0}
    except Exception: return None

def export_to_onnx(model, onnx_path, input_size):
    model.eval()
    dummy = torch.randn(*input_size)
    try:
        # Opset 13 supports ConvTranspose (DeConv) well
        torch.onnx.export(model, dummy, onnx_path, opset_version=13, input_names=['input'], output_names=['output'],
                          dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}},
                          do_constant_folding=True, export_params=True)
        print(f"[✓] Exported: {onnx_path}")
    except Exception as e: print(f"[✗] Export failed: {e}")

def perform_quantization(onnx_fp32, onnx_int8, input_name, input_shape, calib_size=64):
    if not ORT_QUANT_AVAILABLE: return False
    print("[*] Starting Quantization...")
    calib_reader = RandomCalibrationDataReader(input_name, input_shape, num_samples=calib_size)
    
    # Whitelist: Quantize only standard Convs. 
    # Explicitly avoid quantizing ConvTranspose (DeConv) as it often causes accuracy/runtime issues on CPU.
    safe_ops = ['Conv', 'Gemm', 'MatMul', 'Add', 'Mul', 'Relu', 'Clip', 'LeakyRelu', 'Sigmoid', 'GlobalAveragePool']

    try:
        quantize_static(onnx_fp32, onnx_int8, calib_reader, 
                        quant_format=QuantFormat.QDQ, activation_type=QuantType.QInt8, weight_type=QuantType.QInt8,
                        per_channel=False, reduce_range=False, op_types_to_quantize=safe_ops)
        print("[✓] Static Quantization successful.")
        return True
    except:
        try:
            quantize_dynamic(onnx_fp32, onnx_int8, weight_type=QuantType.QInt8, op_types_to_quantize=safe_ops)
            print("[✓] Dynamic Quantization successful.")
            return True
        except: return False

def run_onnxruntime(onnx_path, input_shape, runs=20):
    if not ORT_AVAILABLE or not os.path.exists(onnx_path): return 0.0
    so = ort.SessionOptions(); so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    sess = ort.InferenceSession(onnx_path, sess_options=so, providers=['CPUExecutionProvider'])
    feeds = {'input': np.random.randn(1, *input_shape[1:]).astype(np.float32)}
    for _ in range(5): sess.run(None, feeds)
    times = []
    for _ in range(runs):
        t0 = time.perf_counter(); sess.run(None, feeds); times.append((time.perf_counter() - t0) * 1000.0)
    return float(np.mean(times))

def print_analysis_table(res, flops, params, act_fp32, peak_fp32, lat_fp32, int8_act, int8_peak, lat_int8, size_fp32, size_int8):
    header = (" Input Size | FLOPs (G) | Params (M) | Total Act FP32 (MB) | Peak Act FP32 (MB) | Latency FP32 (ms) | "
              "Total Act INT8 (MB) | Peak Act INT8 (MB) | Latency INT8 (ms) | Size FP32/INT8 (MB)")
    print("-" * 150)
    print(header)
    print("-" * 150)
    size_str = f"{size_fp32:.1f} / {size_int8:.1f}"
    print(f" {res:10s} | {flops:9.3f} | {params:10.3f} | {act_fp32:19.2f} | {peak_fp32:18.2f} | {lat_fp32:17.2f} | "
          f"{int8_act:19.2f} | {int8_peak:18.2f} | {lat_int8:17.2f} | {size_str:>18s}")

def process_resolution(res, n_classes):
    c, w, h = res
    input_shape = (1, c, h, w)
    res_str = f"{w}x{h}"
    print(f"\nProcessing {res_str}...")

    model = ERFNet(num_class=n_classes)
    model.eval()

    flop_prof = FlopProfiler(); act_prof = ActivationProfiler()
    flop_prof.register_hooks(model); act_prof.register_hooks(model)
    with torch.no_grad(): _ = model(torch.randn(*input_shape))
    flops_g = flop_prof.total_flops() / 1e9
    params_m = sum(p.numel() for p in model.parameters()) / 1e6
    act_stats = act_prof.get_stats()
    act_fp32 = act_stats['total_memory_mb']; peak_fp32 = act_stats['peak_activation_mb']
    flop_prof.remove_hooks(); act_prof.remove_hooks()

    onnx_path = f"erfnet_{w}x{h}.onnx"; quant_path = f"erfnet_{w}x{h}_int8.onnx"
    export_to_onnx(model, onnx_path, input_shape)
    
    success_quant = False
    if os.path.exists(onnx_path):
        success_quant = perform_quantization(onnx_path, quant_path, 'input', input_shape, 64)

    int8_act = 0.0; int8_peak = 0.0
    if success_quant:
        est = estimate_onnx_activation_memory(quant_path, 8)
        if est: int8_act = est['total_memory_mb']; int8_peak = est['peak_tensor_mb']
        else: int8_act = act_fp32/4.0; int8_peak = peak_fp32/4.0

    lat_fp32 = run_onnxruntime(onnx_path, input_shape, 10)
    lat_int8 = run_onnxruntime(quant_path, input_shape, 10) if success_quant else 0.0
    size_fp32 = get_model_size_mb(onnx_path)
    size_int8 = get_model_size_mb(quant_path)

    print_analysis_table(res_str, flops_g, params_m, act_fp32, peak_fp32, lat_fp32, int8_act, int8_peak, lat_int8, size_fp32, size_int8)
    return {'res': res_str, 'lat_fp32': lat_fp32, 'lat_int8': lat_int8}

def main():
    print("="*100 + "\nERFNET FIXED INT8 PIPELINE\n" + "="*100)
    resolutions = [
        (3, 640, 360), (3, 1280, 720), (3, 1360, 760),
        (3, 1600, 900), (3, 1920, 1080), (3, 2048, 1152),
        (3, 2560, 1440), (3, 3840, 2160)
    ]
    results = []
    for res in resolutions:
        try: results.append(process_resolution(res, n_classes=19))
        except Exception as e: 
            print(f"Failed {res}: {e}")
            traceback.print_exc()

    print("\n" + "="*60 + "\nSPEEDUP SUMMARY\n" + "="*60)
    for r in results:
        if r['lat_int8'] > 0:
            print(f"{r['res']:10s}: {r['lat_fp32']/r['lat_int8']:.2f}x Speedup")

if __name__ == "__main__":
    main()
