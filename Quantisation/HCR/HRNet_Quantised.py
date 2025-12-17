#!/usr/bin/env python3
"""
HRNet-W48 + OCR Full Benchmarking Pipeline
Features:
1. Dynamic Import of your local 'hrnet_w48_ocr_full.py' file.
2. Robust INT8 Quantization (Whitelist strategy to protect Upsampling layers).
3. Full resolution sweep (360p -> 4K).
"""

import os
import sys
import time
import gc
import numpy as np
import torch
import torch.nn as nn
import warnings
import importlib.util
from pathlib import Path
from types import SimpleNamespace

# ---- PATCH numpy for legacy code ----
if not hasattr(np, 'int'):
    np.int = int

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
#  Model Loading Logic
# ===================================================================

MODULE_FILE = "hrnet_w48_ocr_full.py"

# HRNet Config Dictionary
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

def load_hrnet_model():
    if not Path(MODULE_FILE).is_file():
        print(f"‼️  CRITICAL: '{MODULE_FILE}' not found in current directory.")
        sys.exit(1)

    try:
        spec = importlib.util.spec_from_file_location("hrnet_w48_ocr", MODULE_FILE)
        hrnet_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(hrnet_module)
        
        config_obj = Config()
        model = hrnet_module.HighResolutionNet(config_obj, num_classes=Config.DATASET.NUM_CLASSES)
        return model
    except Exception as e:
        print(f"‼️  Error loading HRNet module: {e}")
        sys.exit(1)

# ===================================================================
#  Benchmarking Pipeline Utils
# ===================================================================

class ActivationProfiler:
    def __init__(self):
        self.activations = []
        self.hooks = []

    def register_hooks(self, model):
        def hook_fn(module, input, output):
            # Handle list/tuple outputs (common in HRNet/OCR)
            outputs = output if isinstance(output, (tuple, list)) else [output]
            for out in outputs:
                if isinstance(out, torch.Tensor):
                    size_bytes = out.element_size() * out.nelement()
                    self.activations.append({
                        'module': module.__class__.__name__,
                        'size_mb': size_bytes / (1024**2)
                    })

        for name, module in model.named_modules():
            # Only leaf modules
            if len(list(module.children())) == 0:
                hook = module.register_forward_hook(hook_fn)
                self.hooks.append(hook)

    def remove_hooks(self):
        for h in self.hooks: h.remove()
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
            x = inp[0]; out_t = out
            if isinstance(out_t, (list, tuple)): out_t = out_t[0]
            
            batch = x.shape[0]
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
    def __init__(self, input_name, input_shape, num_samples=8):
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

def export_to_onnx(model, onnx_path, input_size, input_name='input'):
    model.eval()
    dummy = torch.randn(*input_size)
    try:
        # Opset 12/13 is ideal for HRNet's Resize operations
        torch.onnx.export(
            model, dummy, onnx_path, 
            opset_version=13, 
            input_names=[input_name], output_names=['output'],
            dynamic_axes={input_name: {0: 'batch'}, 'output': {0: 'batch'}},
            do_constant_folding=True, 
            export_params=True
        )
        print(f"[✓] Exported ONNX -> {onnx_path}")
    except Exception as e:
        print(f"[✗] ONNX Export failed: {e}")

def perform_quantization(onnx_fp32, onnx_int8, input_name, input_shape, calib_size=8):
    if not ORT_QUANT_AVAILABLE: return False
    
    print("[*] Starting Quantization...")
    calib_reader = RandomCalibrationDataReader(input_name, input_shape, num_samples=calib_size)
    
    # --- WHITELIST STRATEGY ---
    # HRNet uses extensive bilinear Upsampling. If we quantize these nodes,
    # ONNX Runtime CPU execution provider often crashes or produces garbage.
    # We strictly quantize ONLY: Conv, MatMul, Gemm, Add, Relu.
    safe_ops = ['Conv', 'Gemm', 'MatMul', 'Add', 'Mul', 'Relu', 'Clip', 'LeakyRelu', 'Sigmoid', 'GlobalAveragePool']

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
            op_types_to_quantize=safe_ops 
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
                op_types_to_quantize=safe_ops
            )
            print("[✓] Dynamic Quantization successful.")
            return True
        except Exception as e:
            print(f"[✗] Dynamic quantization failed: {e}")
            return False

def run_onnxruntime(onnx_path, input_name, input_shape, warmup=3, runs=10):
    if not ORT_AVAILABLE or not os.path.exists(onnx_path): return 0.0
    try:
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        # CPU Provider
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

def print_analysis_table(input_size_str, flops_g, params_m, total_act_mb_fp32, peak_act_mb_fp32, lat_fp32, lat_int8, model_fp32_size_mb, model_int8_size_mb):
    header = (" Input Size | FLOPs (G) | Params (M) | Total Act FP32 (MB) | Peak Act FP32 (MB) | "
              "Latency FP32 (ms) | Latency INT8 (ms) | Size FP32/INT8 (MB)")
    print(header)
    print("-" * 130)
    size_str = f"{model_fp32_size_mb:.1f} / {model_int8_size_mb:.1f}"
    print(f" {input_size_str:10s} | {flops_g:9.3f} | {params_m:10.3f} | {total_act_mb_fp32:19.2f} | "
          f"{peak_act_mb_fp32:18.2f} | {lat_fp32:17.2f} | {lat_int8:17.2f} | {size_str:>18s}")

def process_resolution(resolution):
    c, w, h = resolution
    input_shape = (1, c, h, w)
    input_size_str = f"{w}x{h}"

    print(f"\n{'='*80}")
    print(f"Processing resolution: {input_size_str}")
    print(f"{'='*80}")

    # Check for potential OOM on standard machines (HRNet W48 is huge)
    if h * w > 2560 * 1440:
        print("[!] Warning: High resolution detected. This may cause OOM on standard RAM.")

    # 1. Load fresh model
    try:
        model = load_hrnet_model()
        model.eval()
    except Exception as e:
        print(f"[✗] Failed to load model: {e}")
        return {}

    # 2. PyTorch Profiling
    flop_prof = FlopProfiler(); act_prof = ActivationProfiler()
    flop_prof.register_hooks(model); act_prof.register_hooks(model)
    try:
        with torch.no_grad(): _ = model(torch.randn(*input_shape))
        flops_g = flop_prof.total_flops() / 1e9
        act_stats = act_prof.get_stats()
        total_act_mb_fp32 = act_stats['total_memory_mb']
        peak_act_mb_fp32 = act_stats['peak_activation_mb']
    except Exception as e:
        print(f"[!] PyTorch profiling failed (likely OOM): {e}")
        flops_g = 0; total_act_mb_fp32 = 0; peak_act_mb_fp32 = 0
    
    flop_prof.remove_hooks(); act_prof.remove_hooks()
    total_params = sum(p.numel() for p in model.parameters()) / 1e6

    # 3. ONNX Export
    onnx_path = f"hrnet_{w}x{h}.onnx"
    quant_path = f"hrnet_{w}x{h}_int8.onnx"
    export_to_onnx(model, onnx_path, input_shape)
    
    # Free up torch memory before quantization
    del model
    gc.collect()
    
    # 4. Quantization
    success_quant = False
    if os.path.exists(onnx_path):
        success_quant = perform_quantization(onnx_path, quant_path, 'input', input_shape)

    # 5. Benchmark (Reduced runs for high-res to save time)
    lat_fp32 = run_onnxruntime(onnx_path, 'input', input_shape, runs=5) 
    lat_int8 = run_onnxruntime(quant_path, 'input', input_shape, runs=5) if success_quant else 0.0

    fp32_size = get_model_size_mb(onnx_path)
    int8_size = get_model_size_mb(quant_path)

    print_analysis_table(input_size_str, flops_g, total_params, total_act_mb_fp32, peak_act_mb_fp32, lat_fp32, lat_int8, fp32_size, int8_size)
    
    return {'resolution': input_size_str, 'lat_fp32': lat_fp32, 'lat_int8': lat_int8, 'size_fp32': fp32_size, 'size_int8': int8_size}

def main():
    print("\n" + "="*80)
    print("HRNET-W48 + OCR | ROBUST PIPELINE")
    print("="*80)
    
    resolutions_hw = [
        (360, 640), (720, 1280), (760, 1360), (900, 1600), 
        (1080, 1920), (1152, 2048), (1440, 2560), (2160, 3840)
    ]
    resolutions = [(3, w, h) for h, w in resolutions_hw] 

    all_results = []
    for res in resolutions:
        try:
            r = process_resolution(res)
            if r: all_results.append(r)
        except Exception as e:
            print(f"[✗] Failed {res}: {e}")
            traceback.print_exc()
        # Aggressive cleanup between resolutions
        gc.collect()

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    for r in all_results:
        speedup = r['lat_fp32'] / r['lat_int8'] if r['lat_int8'] > 0 else 0
        comp = r['size_fp32'] / r['size_int8'] if r['size_int8'] > 0 else 0
        print(f"{r['resolution']:10s} | Speedup: {speedup:5.2f}x | Compression: {comp:5.2f}x | FP32: {r['lat_fp32']:.1f}ms -> INT8: {r['lat_int8']:.1f}ms")

if __name__ == "__main__":
    main()
