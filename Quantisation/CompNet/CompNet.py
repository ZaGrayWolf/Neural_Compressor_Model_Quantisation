#!/usr/bin/env python3
"""
COmpNet (Autoencoder) FIXED PIPELINE - Strict Whitelist Mode
Fixes: Uses an op-type whitelist to force PixelShuffle (DepthToSpace) to remain in FP32,
preventing 'Failed to find kernel' crashes on CPU.
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
#  COmpNet Model Definition
# ===================================================================

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
        return self.encoder(x)

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
        return self.decoder(x)

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
        x = torch.clamp(x, 0, 1) 
        x = self.decoder(x)
        x = torch.clamp(x, 0, 1) 
        return x

# ===================================================================
#  Benchmarking Pipeline
# ===================================================================

class ActivationProfiler:
    def __init__(self):
        self.activations = []
        self.hooks = []

    def register_hooks(self, model):
        def hook_fn(module, input, output):
            if isinstance(output, torch.Tensor):
                size_bytes = output.element_size() * output.nelement()
                self.activations.append({
                    'module': module.__class__.__name__,
                    'shape': tuple(output.shape),
                    'size_mb': size_bytes / (1024**2),
                    'dtype': str(output.dtype)
                })
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:
                hook = module.register_forward_hook(hook_fn)
                self.hooks.append(hook)

    def clear(self):
        self.activations = []

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def get_stats(self):
        if not self.activations:
            return {'total_memory_mb': 0.0, 'peak_activation_mb': 0.0}
        total_mb = sum(a['size_mb'] for a in self.activations)
        peak_mb = max(a['size_mb'] for a in self.activations)
        return {'total_memory_mb': total_mb, 'peak_activation_mb': peak_mb}

class FlopProfiler:
    def __init__(self):
        self.hooks = []
        self.layer_flops = []

    def _conv_flops(self, module, inp, out):
        try:
            x = inp[0]; out_t = out; batch = x.shape[0]
            Cout = out_t.shape[1]; H_out = out_t.shape[2]; W_out = out_t.shape[3]
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                Cin = module.in_channels
                kh, kw = module.kernel_size if isinstance(module.kernel_size, tuple) else (module.kernel_size, module.kernel_size)
                groups = module.groups if hasattr(module, 'groups') else 1
                macs = batch * Cout * H_out * W_out * (Cin // groups) * kh * kw
                flops = 2 * macs
                self.layer_flops.append((module.__class__.__name__, flops))
        except Exception: pass

    def register_hooks(self, model):
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                h = module.register_forward_hook(lambda m, i, o, mm=module: self._conv_flops(mm, i, o))
                self.hooks.append(h)

    def remove_hooks(self):
        for h in self.hooks: h.remove()
        self.hooks = []

    def total_flops(self): return sum(f for _, f in self.layer_flops)

class RandomCalibrationDataReader(CalibrationDataReader if ORT_QUANT_AVAILABLE else object):
    def __init__(self, input_name, input_shape, num_samples=32):
        self.input_name = input_name
        self.input_shape = input_shape
        self.num_samples = int(num_samples)
        self._idx = 0
    def get_next(self):
        if self._idx >= self.num_samples: return None
        self._idx += 1
        return {self.input_name: np.random.randn(*self.input_shape).astype(np.float32)}

def get_model_size_mb(path):
    return os.path.getsize(path) / (1024**2) if os.path.exists(path) else 0.0

def estimate_onnx_activation_memory(onnx_path, dtype_bits=8):
    if not ONNX_AVAILABLE or not os.path.exists(onnx_path): return None
    try:
        model = onnx.load(onnx_path)
        try: model = onnx.shape_inference.infer_shapes(model)
        except: pass
        tensors = []
        names_seen = set()
        def collect_vi(vi_list):
            for vi in vi_list:
                name = vi.name
                if name in names_seen: continue
                names_seen.add(name)
                try:
                    dims = [d.dim_value for d in vi.type.tensor_type.shape.dim]
                    if any(d == 0 for d in dims): continue
                    tensors.append((name, tuple(dims)))
                except: pass
        collect_vi(model.graph.value_info); collect_vi(model.graph.output); collect_vi(model.graph.input)
        bytes_per_elem = dtype_bits / 8.0
        total_mb = sum(((np.prod(shape) * bytes_per_elem) / (1024**2)) for _, shape in tensors)
        activations = [{'size_mb': (np.prod(shape) * bytes_per_elem) / (1024**2)} for _, shape in tensors]
        if activations: activations.sort(key=lambda x: x['size_mb'], reverse=True)
        return {'total_memory_mb': total_mb, 'peak_tensor_mb': activations[0]['size_mb'] if activations else 0}
    except Exception: return None

def export_to_onnx(model, onnx_path, input_size=(1, 3, 640, 360), input_name='input'):
    model.eval()
    dummy = torch.randn(*input_size)
    with torch.no_grad():
        torch.onnx.export(
            model, dummy, onnx_path, opset_version=13,
            input_names=[input_name], output_names=['output'],
            dynamic_axes={input_name: {0: 'batch'}, 'output': {0: 'batch'}},
            do_constant_folding=True, export_params=True
        )
    print(f"[✓] Exported ONNX -> {onnx_path}")

def perform_quantization(onnx_fp32, onnx_int8, input_name, input_shape, calib_size=64):
    if not ORT_QUANT_AVAILABLE:
        print("[!] ORT Quantization not installed.")
        return False
        
    print("[*] Starting Quantization...")
    calib_reader = RandomCalibrationDataReader(input_name, input_shape, num_samples=calib_size)
    
    # *** CRITICAL FIX: WHITELIST ONLY SAFE OPS ***
    # By strictly defining this list, we force 'DepthToSpace' (PixelShuffle)
    # and 'Resize' (Upsample) to be ignored by the quantizer. 
    # They will be wrapped in Dequantize->Op->Quantize blocks automatically.
    safe_ops_to_quantize = ['Conv', 'Gemm', 'MatMul', 'Add', 'Mul', 'Relu', 'Clip', 'LeakyRelu', 'Sigmoid', 'GlobalAveragePool', 'MaxPool']

    try:
        quantize_static(
            model_input=onnx_fp32,
            model_output=onnx_int8,
            calibration_data_reader=calib_reader,
            quant_format=QuantFormat.QDQ,
            activation_type=QuantType.QInt8,
            weight_type=QuantType.QInt8,
            per_channel=False,
            reduce_range=False,
            op_types_to_quantize=safe_ops_to_quantize # <--- The solution
        )
        print("[✓] Static Quantization successful (Whitelist Mode).")
        return True
    except Exception as e:
        print(f"[!] Static quantization failed: {e}")
        
    try:
        print("[*] Falling back to Dynamic Quantization...")
        quantize_dynamic(
            model_input=onnx_fp32,
            model_output=onnx_int8,
            weight_type=QuantType.QInt8,
            op_types_to_quantize=safe_ops_to_quantize
        )
        print("[✓] Dynamic Quantization successful.")
        return True
    except Exception as e:
        print(f"[✗] Dynamic quantization failed: {e}")
        return False

def run_onnxruntime(onnx_path, input_name, input_shape, warmup=5, runs=20):
    if not ORT_AVAILABLE or not os.path.exists(onnx_path): return 0.0
    try:
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        sess = ort.InferenceSession(onnx_path, sess_options=so, providers=['CPUExecutionProvider'])
        
        batch_shape = (1,) + tuple(input_shape[1:])
        x = np.random.randn(*batch_shape).astype(np.float32)
        feeds = {input_name: x}

        for _ in range(warmup): sess.run(None, feeds)
        times = []
        for _ in range(runs):
            t0 = time.perf_counter()
            sess.run(None, feeds)
            times.append((time.perf_counter() - t0) * 1000.0)
        return float(np.mean(times))
    except Exception as e:
        print(f"[!] Runtime failed for {onnx_path}: {e}")
        return 0.0

def print_analysis_table(input_size_str, flops_g, params_m, total_act_mb_fp32, peak_act_mb_fp32, lat_fp32,
                          total_act_mb_int8, peak_act_mb_int8, lat_int8, model_fp32_size_mb, model_int8_size_mb):
    header = (" Input Size | FLOPs (G) | Params (M) | Total Act FP32 (MB) | Peak Act FP32 (MB) | "
              "Latency FP32 (ms) | Total Act INT8 (MB) | Peak Act INT8 (MB) | Latency INT8 (ms) | Size FP32/INT8 (MB)")
    print(header)
    print("-" * 150)
    size_str = f"{model_fp32_size_mb:.1f} / {model_int8_size_mb:.1f}"
    print(f" {input_size_str:10s} | {flops_g:9.3f} | {params_m:10.3f} | {total_act_mb_fp32:19.2f} | "
          f"{peak_act_mb_fp32:18.2f} | {lat_fp32:17.2f} | {total_act_mb_int8:19.2f} | "
          f"{peak_act_mb_int8:18.2f} | {lat_int8:16.2f} | {size_str:>18s}")

def process_resolution(resolution, in_channels, out_channels, n_updownconv=3, calib_size=64):
    c, w, h = resolution
    input_shape = (1, c, h, w)
    input_size_str = f"{w}x{h}"

    print(f"\n{'='*80}")
    print(f"Processing resolution: {input_size_str}")
    print(f"{'='*80}")

    # 1. PyTorch
    model = autoencoder(n_updownconv=n_updownconv, in_chn=in_channels, out_chn=out_channels)
    model.eval()
    flop_prof = FlopProfiler(); act_prof = ActivationProfiler()
    flop_prof.register_hooks(model); act_prof.register_hooks(model)
    with torch.no_grad(): _ = model(torch.randn(*input_shape))
    flops_g = flop_prof.total_flops() / 1e9
    act_stats = act_prof.get_stats()
    total_act_mb_fp32 = act_stats['total_memory_mb']
    peak_act_mb_fp32 = act_stats['peak_activation_mb']
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    flop_prof.remove_hooks(); act_prof.remove_hooks()

    # 2. ONNX Export
    onnx_path = f"compnet_{w}x{h}.onnx"
    quant_path = f"compnet_{w}x{h}_int8.onnx"
    export_to_onnx(model, onnx_path, input_shape)
    
    # 3. Quantization
    success_quant = perform_quantization(onnx_path, quant_path, 'input', input_shape, calib_size)
    
    # 4. Benchmark
    lat_fp32 = run_onnxruntime(onnx_path, 'input', input_shape)
    lat_int8 = 0.0; int8_act_total = 0.0; int8_act_peak = 0.0
    
    if success_quant:
        lat_int8 = run_onnxruntime(quant_path, 'input', input_shape)
        mem_est = estimate_onnx_activation_memory(quant_path, 8)
        if mem_est:
            int8_act_total = mem_est['total_memory_mb']
            int8_act_peak = mem_est['peak_tensor_mb']
        else:
            int8_act_total = total_act_mb_fp32 / 4.0
            int8_act_peak = peak_act_mb_fp32 / 4.0

    fp32_size = get_model_size_mb(onnx_path)
    int8_size = get_model_size_mb(quant_path)

    print_analysis_table(input_size_str, flops_g, total_params, total_act_mb_fp32, peak_act_mb_fp32, lat_fp32,
                         int8_act_total, int8_act_peak, lat_int8, fp32_size, int8_size)

    return {'resolution': input_size_str, 'lat_fp32': lat_fp32, 'lat_int8': lat_int8, 'size_fp32': fp32_size, 'size_int8': int8_size}

def main():
    print("\n" + "="*80)
    print("COMPNET (AUTOENCODER) FIXED INT8 PIPELINE - WHITELIST MODE")
    print("="*80)
    resolutions_hw = [(360, 640), (720, 1280), (760, 1360), (900, 1600), (1080, 1920), (1152, 2048), (1440, 2560), (3840, 2160)]
    resolutions = [(3, w, h) for h, w in resolutions_hw] 
    in_channels = 3; out_channels = 3; n_updownconv = 3

    all_results = []
    for res in resolutions:
        try:
            r = process_resolution(res, in_channels, out_channels, n_updownconv)
            all_results.append(r)
        except Exception as e:
            print(f"[✗] Failed {res}: {e}")
            traceback.print_exc()

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    for r in all_results:
        speedup = r['lat_fp32'] / r['lat_int8'] if r['lat_int8'] > 0 else 0
        comp = r['size_fp32'] / r['size_int8'] if r['size_int8'] > 0 else 0
        print(f"{r['resolution']:10s} | Speedup: {speedup:5.2f}x | Compression: {comp:5.2f}x | FP32: {r['lat_fp32']:.1f}ms -> INT8: {r['lat_int8']:.1f}ms")

if __name__ == "__main__":
    main()
