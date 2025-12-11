#!/usr/bin/env python3
"""
UNet Encoder (No Skip) benchmarking with robust ONNX + ONNX Runtime INT8 fallback.

Flow:
1) Measure PyTorch CPU latency
2) Export ONNX FP32, run ORT FP32
3) Try dynamic quantization (weights-only). If ORT can't run it (ConvInteger NOT_IMPLEMENTED),
   fall back to static QDQ quantization using a tiny synthetic calibration set and try ORT again.
4) Save JSON results.

Notes:
- For reliable INT8 runtime on CPU, use an onnxruntime build that supports the quant kernels,
  or prefer QDQ quantized models which are often more portable across ORT builds.
"""

import os
import time
import json
import gc
import warnings
from typing import Tuple, List, Dict

import numpy as np
import torch
import torch.nn as nn

warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

# Check optional libs
try:
    import onnx
    ONNX_AVAILABLE = True
except Exception:
    ONNX_AVAILABLE = False

try:
    import onnxruntime as ort
    ORT_AVAILABLE = True
except Exception:
    ORT_AVAILABLE = False

# quantization utilities (optional)
try:
    from onnxruntime.quantization import quantize_dynamic, QuantType, quantize_static, CalibrationDataReader, QuantFormat
    ORT_QUANT_AVAILABLE = True
except Exception:
    ORT_QUANT_AVAILABLE = False

# -------------------------
# Model (UNetEncoderNoSkip)
# -------------------------
class UNetEncoderNoSkip(nn.Module):
    def __init__(self):
        super(UNetEncoderNoSkip, self).__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.pool1(x1)
        x3 = self.enc2(x2)
        x4 = self.pool2(x3)
        x5 = self.enc3(x4)
        x6 = self.pool3(x5)
        x7 = self.enc4(x6)
        return x7

# -------------------------
# Utility functions
# -------------------------
def calculate_flops_and_params(input_shape: Tuple[int, int, int], output_channels: int,
                               kernel_size: Tuple[int,int]=(3,3), stride: Tuple[int,int]=(1,1)):
    in_h, in_w, in_c = input_shape
    kh, kw = kernel_size
    sh, sw = stride
    pad_h = pad_w = 1
    out_h = (in_h - kh + 2*pad_h) // sh + 1
    out_w = (in_w - kw + 2*pad_w) // sw + 1
    if out_h <= 0 or out_w <= 0:
        out_h = max(0, out_h); out_w = max(0, out_w)
    params = (kh * kw * in_c + 1) * output_channels
    mults = (kh * kw * in_c) * out_h * out_w * output_channels
    divs = out_h * out_w * output_channels
    add_subs = (kh * kw * in_c - 1) * out_h * out_w * output_channels
    flops = mults + divs + add_subs
    return params, flops, (out_h, out_w, output_channels)

def measure_latency_pytorch(model: nn.Module, input_tensor: torch.Tensor,
                            warmup: int = 10, runs: int = 50):
    device = torch.device("cpu")
    model = model.to(device).eval()
    x = input_tensor.to(device)
    with torch.no_grad():
        for _ in range(max(1,warmup)): _ = model(x)
    times = []
    with torch.no_grad():
        for _ in range(max(1,runs)):
            t0 = time.perf_counter()
            _ = model(x)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000.0)
    arr = np.array(times)
    return {'mean_ms': float(arr.mean()), 'p95_ms': float(np.percentile(arr,95)), 'p99_ms': float(np.percentile(arr,99)), 'runs': len(arr)}

def export_to_onnx(model: nn.Module, onnx_path: str, input_size: Tuple[int,int,int], input_name: str='input', opset: int = 13):
    model_cpu = model.cpu().eval()
    C,H,W = input_size
    dummy = torch.randn(1, C, H, W, dtype=torch.float32)
    torch.onnx.export(model_cpu, dummy, onnx_path,
                      input_names=[input_name], output_names=['output'],
                      opset_version=opset, do_constant_folding=True,
                      dynamic_axes={input_name: {0: 'batch'}, 'output': {0: 'batch'}})
    return onnx_path

def onnx_file_size_mb(path: str) -> float:
    try:
        return os.path.getsize(path) / (1024.0**2)
    except Exception:
        return 0.0

def run_onnxruntime_inference(onnx_path: str, input_size: Tuple[int,int,int],
                              warmup: int = 5, runs: int = 30, use_cuda: bool = False):
    if not ORT_AVAILABLE:
        raise RuntimeError("onnxruntime not installed")
    providers = ort.get_available_providers()
    if use_cuda and 'CUDAExecutionProvider' in providers:
        providers_to_use = ['CUDAExecutionProvider']
    else:
        providers_to_use = ['CPUExecutionProvider']

    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    sess = ort.InferenceSession(onnx_path, sess_options=so, providers=providers_to_use)
    input_name = sess.get_inputs()[0].name
    C,H,W = input_size
    x = np.random.randn(1, C, H, W).astype(np.float32)
    feed = {input_name: x}
    for _ in range(max(1,warmup)): sess.run(None, feed)
    times = []
    for _ in range(max(1,runs)):
        t0 = time.perf_counter()
        sess.run(None, feed)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)
    arr = np.array(times)
    return {'mean_ms': float(arr.mean()), 'p95_ms': float(np.percentile(arr,95)), 'p99_ms': float(np.percentile(arr,99)), 'runs': len(arr)}

def quantize_dynamic_onnx(fp32_path: str, int8_out: str):
    if not ORT_QUANT_AVAILABLE:
        raise RuntimeError("onnxruntime.quantization not available")
    quantize_dynamic(model_input=fp32_path, model_output=int8_out, weight_type=QuantType.QInt8)
    return int8_out

# --- Fallback: Simple CalibrationDataReader for quantize_static (QDQ) ---
if ORT_QUANT_AVAILABLE:
    class RandomCalibrationDataReader(CalibrationDataReader):
        def __init__(self, input_name: str, input_shape: Tuple[int,int,int], num_samples: int = 10):
            # input_shape is (C, H, W) for each sample
            self.data = []
            C,H,W = input_shape
            for _ in range(num_samples):
                arr = np.random.randn(1, C, H, W).astype(np.float32)
                self.data.append({input_name: arr})
            self._iter = iter(self.data)

        def get_next(self):
            try:
                return next(self._iter)
            except StopIteration:
                return None

def quantize_static_qdq(fp32_path: str, int8_out: str, input_size: Tuple[int,int,int], num_calib: int = 10):
    """Use quantize_static with QDQ format (more portable in ORT)."""
    if not ORT_QUANT_AVAILABLE:
        raise RuntimeError("onnxruntime.quantization not available")
    # create a session to get the input name
    so = ort.SessionOptions()
    sess = ort.InferenceSession(fp32_path, sess_options=so, providers=['CPUExecutionProvider'])
    input_name = sess.get_inputs()[0].name
    # create calibration reader with random data (user can replace with real samples)
    dr = RandomCalibrationDataReader(input_name, input_size, num_samples=num_calib)
    # quantize_static parameters: use QDQ format so ORT uses Q/DQ nodes
    quantize_static(model_input=fp32_path, model_output=int8_out, calibration_data_reader=dr,
                    quant_format=QuantFormat.QDQ, per_channel=False,
                    activation_type=QuantType.QInt8, weight_type=QuantType.QInt8)
    return int8_out

# --------------------------
# Benchmark harness
# --------------------------
def run_all_benchmarks(save_json: str = "unet_encoder_benchmark_results.json",
                       resolutions: List[Tuple[int,int]] = None,
                       do_onnx: bool = True,
                       do_quant: bool = True,
                       ort_use_cuda: bool = False):
    if resolutions is None:
        resolutions = [
            (360, 640),
            (720, 1280),
            (760, 1360),
            (900, 1600),
            (1080, 1920),
            (1152, 2048),
            (1440, 2560),
        ]

    in_ch = 3
    results = []

    print("Environment:")
    print(f"  ONNX available: {ONNX_AVAILABLE}")
    print(f"  ONNXRuntime available: {ORT_AVAILABLE}")
    print(f"  ONNXRuntime.quantization available: {ORT_QUANT_AVAILABLE}")
    if ORT_AVAILABLE:
        print("  ORT providers:", ort.get_available_providers())
    print()

    for (h, w) in resolutions:
        print("="*100)
        print(f"Benchmarking input: 3 x {h} x {w}")
        model = UNetEncoderNoSkip().eval()

        # Theoretical params/flops (layer-by-layer)
        curr_shape = (h, w, 3)
        encoder_channels = [64, 64, 128, 128, 256, 256, 512, 512]
        total_params = 0
        total_flops = 0
        for i, out_ch in enumerate(encoder_channels):
            p, f, out_shape = calculate_flops_and_params(curr_shape, out_ch, (3,3), (1,1))
            total_params += p
            total_flops += f
            curr_shape = out_shape
            if i in [1, 3, 5]:
                curr_shape = (curr_shape[0] // 2, curr_shape[1] // 2, curr_shape[2])

        input_tensor = torch.randn(1, 3, h, w)

        # PyTorch latency
        pt_latency = None
        try:
            pt_latency = measure_latency_pytorch(model, input_tensor, warmup=10, runs=50)
            print(f"PyTorch latency mean {pt_latency['mean_ms']:.2f} ms (p95 {pt_latency['p95_ms']:.2f}, p99 {pt_latency['p99_ms']:.2f})")
        except Exception as e:
            print(f"[WARN] PyTorch latency failed: {e}")

        onnx_fp32_path = f"unet_encoder_3x{h}x{w}_fp32.onnx"
        onnx_int8_path_dyn = f"unet_encoder_3x{h}x{w}_int8_dynamic.onnx"
        onnx_int8_path_qdq = f"unet_encoder_3x{h}x{w}_int8_qdq.onnx"
        onnx_result = {'fp32': None, 'int8_dynamic': None, 'int8_qdq': None}

        if do_onnx and ONNX_AVAILABLE:
            try:
                export_to_onnx(model, onnx_fp32_path, (3,h,w), input_name='input', opset=13)
                fp32_size = onnx_file_size_mb(onnx_fp32_path)
                print(f"Exported ONNX FP32 -> {onnx_fp32_path} ({fp32_size:.2f} MB)")
                onnx_result['fp32'] = {'path': os.path.abspath(onnx_fp32_path), 'size_mb': fp32_size, 'ort': None}
                if ORT_AVAILABLE:
                    try:
                        ort_stats = run_onnxruntime_inference(onnx_fp32_path, (3,h,w), warmup=5, runs=30, use_cuda=ort_use_cuda)
                        onnx_result['fp32']['ort'] = ort_stats
                        print(f"ORT FP32 mean {ort_stats['mean_ms']:.2f} ms (p95 {ort_stats['p95_ms']:.2f})")
                    except Exception as e:
                        print(f"[WARN] ORT FP32 inference failed: {e}")
            except Exception as e:
                print(f"[WARN] ONNX export failed: {e}")

            # Try dynamic quantization first
            if do_quant and ORT_QUANT_AVAILABLE and onnx_result.get('fp32') and onnx_result['fp32'].get('path'):
                # 1) dynamic quant
                try:
                    quantize_dynamic_onnx(onnx_result['fp32']['path'], onnx_int8_path_dyn)
                    int8_size = onnx_file_size_mb(onnx_int8_path_dyn)
                    onnx_result['int8_dynamic'] = {'path': os.path.abspath(onnx_int8_path_dyn), 'size_mb': int8_size, 'ort': None}
                    print(f"Saved quantized ONNX (dynamic) -> {onnx_int8_path_dyn} ({int8_size:.2f} MB)")

                    if ORT_AVAILABLE:
                        try:
                            ortq_stats = run_onnxruntime_inference(onnx_int8_path_dyn, (3,h,w), warmup=5, runs=30, use_cuda=False)
                            onnx_result['int8_dynamic']['ort'] = ortq_stats
                            print(f"ORT INT8(dynamic) mean {ortq_stats['mean_ms']:.2f} ms (p95 {ortq_stats['p95_ms']:.2f})")
                        except Exception as e:
                            # Common error: NOT_IMPLEMENTED ConvInteger
                            print(f"[WARN] ORT INT8(dynamic) inference failed (ConvInteger/impl issue): {e}")
                            onnx_result['int8_dynamic']['ort_error'] = str(e)
                            # FALLBACK: try static QDQ quantization
                            try:
                                print("Attempting static QDQ quantization (fallback)...")
                                quantize_static_qdq(onnx_result['fp32']['path'], onnx_int8_path_qdq, (3,h,w), num_calib=8)
                                int8_qdq_size = onnx_file_size_mb(onnx_int8_path_qdq)
                                onnx_result['int8_qdq'] = {'path': os.path.abspath(onnx_int8_path_qdq), 'size_mb': int8_qdq_size, 'ort': None}
                                print(f"Saved quantized ONNX (static QDQ) -> {onnx_int8_path_qdq} ({int8_qdq_size:.2f} MB)")
                                # try ORT on QDQ model
                                try:
                                    ortq_stats2 = run_onnxruntime_inference(onnx_int8_path_qdq, (3,h,w), warmup=5, runs=30, use_cuda=False)
                                    onnx_result['int8_qdq']['ort'] = ortq_stats2
                                    print(f"ORT INT8(QDQ) mean {ortq_stats2['mean_ms']:.2f} ms (p95 {ortq_stats2['p95_ms']:.2f})")
                                except Exception as e2:
                                    print(f"[WARN] ORT INT8(QDQ) inference failed: {e2}")
                                    onnx_result['int8_qdq']['ort_error'] = str(e2)
                            except Exception as e_qdq:
                                print(f"[WARN] Static QDQ quantization failed: {e_qdq}")
                                onnx_result['int8_qdq'] = {'error': str(e_qdq)}
                except Exception as e:
                    print(f"[WARN] ONNX dynamic quantization failed: {e}")
                    # Try QDQ fallback directly if dynamic quantization failed
                    try:
                        print("Attempting static QDQ quantization (direct fallback)...")
                        quantize_static_qdq(onnx_result['fp32']['path'], onnx_int8_path_qdq, (3,h,w), num_calib=8)
                        int8_qdq_size = onnx_file_size_mb(onnx_int8_path_qdq)
                        onnx_result['int8_qdq'] = {'path': os.path.abspath(onnx_int8_path_qdq), 'size_mb': int8_qdq_size, 'ort': None}
                        print(f"Saved quantized ONNX (static QDQ) -> {onnx_int8_path_qdq} ({int8_qdq_size:.2f} MB)")
                        if ORT_AVAILABLE:
                            try:
                                ortq_stats2 = run_onnxruntime_inference(onnx_int8_path_qdq, (3,h,w), warmup=5, runs=30, use_cuda=False)
                                onnx_result['int8_qdq']['ort'] = ortq_stats2
                                print(f"ORT INT8(QDQ) mean {ortq_stats2['mean_ms']:.2f} ms (p95 {ortq_stats2['p95_ms']:.2f})")
                            except Exception as e2:
                                print(f"[WARN] ORT INT8(QDQ) inference failed: {e2}")
                                onnx_result['int8_qdq']['ort_error'] = str(e2)
                    except Exception as e_qdq:
                        print(f"[WARN] Static QDQ quantization failed: {e_qdq}")
                        onnx_result['int8_qdq'] = {'error': str(e_qdq)}
        else:
            if do_onnx and not ONNX_AVAILABLE:
                print("ONNX export skipped (onnx package not installed).")

        # store results
        res = {
            'input_shape': f"3 x {h} x {w}",
            'height': h,
            'width': w,
            'theoretical_params': int(total_params),
            'theoretical_params_m': total_params / 1e6,
            'theoretical_flops': int(total_flops),
            'theoretical_flops_g': total_flops / 1e9,
            'pytorch_latency': pt_latency,
            'onnx': onnx_result
        }
        results.append(res)
        gc.collect()

    with open(save_json, 'w') as f:
        json.dump(results, f, indent=2)

    # Print summary
    print("\nSummary:")
    header = ("{:<14} {:>14} {:>14} {:>12} {:>12} {:>14} {:>12}"
              .format("resolution", "theo_params(M)", "theo_FLOPs(G)",
                      "pt_mean_ms", "onnx_fp32_ms", "onnx_fp32_MB", "onnx_int8_ms"))
    print(header)
    for r in results:
        inp = r['input_shape']
        theo_p = r['theoretical_params_m']
        theo_f = r['theoretical_flops_g']
        pt_mean = r['pytorch_latency']['mean_ms'] if r.get('pytorch_latency') else None
        onnx_fp32_ms = None
        onnx_fp32_mb = None
        onnx_int8_ms = None
        onnx_int8_mb = None
        if r.get('onnx') and r['onnx'].get('fp32') and r['onnx']['fp32'].get('ort'):
            onnx_fp32_ms = r['onnx']['fp32']['ort']['mean_ms']
            onnx_fp32_mb = r['onnx']['fp32']['size_mb']
        # prefer successful dynamic OR QDQ if present
        if r.get('onnx') and r['onnx'].get('int8_dynamic'):
            if r['onnx']['int8_dynamic'].get('ort'):
                onnx_int8_ms = r['onnx']['int8_dynamic']['ort']['mean_ms']
                onnx_int8_mb = r['onnx']['int8_dynamic']['size_mb']
            elif r['onnx']['int8_dynamic'].get('ort_error'):
                onnx_int8_ms = f"ERR"
                onnx_int8_mb = r['onnx']['int8_dynamic']['size_mb']
        if r.get('onnx') and r['onnx'].get('int8_qdq'):
            if r['onnx']['int8_qdq'].get('ort'):
                onnx_int8_ms = r['onnx']['int8_qdq']['ort']['mean_ms']
                onnx_int8_mb = r['onnx']['int8_qdq']['size_mb']
            elif r['onnx']['int8_qdq'].get('ort_error'):
                onnx_int8_ms = f"ERR"
                onnx_int8_mb = r['onnx']['int8_qdq'].get('size_mb', 'N/A')

        print("{:<14} {:14.2f} {:14.3f} {:12} {:12} {:14} {:12}".format(
            inp,
            theo_p if theo_p is not None else 0.0,
            theo_f if theo_f is not None else 0.0,
            f"{pt_mean:.2f}" if pt_mean is not None else "N/A",
            f"{onnx_fp32_ms:.2f}" if onnx_fp32_ms is not None else "N/A",
            f"{onnx_fp32_mb:.2f}MB" if onnx_fp32_mb is not None else "N/A",
            f"{onnx_int8_ms}" if onnx_int8_ms is not None else "N/A"
        ))

    print(f"\nResults saved to {save_json}")
    return results

if __name__ == "__main__":
    run_all_benchmarks()
