#!/usr/bin/env python3
"""
SN_fixed_v3.py

SegNet INT8 Quantisation & Benchmark Pipeline (CPU-optimised)

- Uses ONNX-friendly SegNet (interpolate instead of max_unpool2d)
- Exports to ONNX if onnx is available
- Uses ONNX Runtime for benchmarking & quantization if available
- Falls back to PyTorch CPU timings when ONNX Runtime is absent
- Activation / FLOP profiling via forward hooks (approximate)
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
    QuantFormat = None
    CalibrationDataReader = None
    ORT_QUANT_AVAILABLE = False

# ===================================================================
#  ONNX-friendly SegNet (uses interpolate instead of MaxUnpool2d)
# ===================================================================
class SegNet(nn.Module):
    def __init__(self, input_channels=3, n_labels=21, kernel_size=3):
        super(SegNet, self).__init__()
        # Encoder
        self.conv1_1 = nn.Conv2d(input_channels, 64, kernel_size, padding=1)
        self.bn1_1 = nn.BatchNorm2d(64)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size, padding=1)
        self.bn1_2 = nn.BatchNorm2d(64)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size, padding=1)
        self.bn2_1 = nn.BatchNorm2d(128)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size, padding=1)
        self.bn2_2 = nn.BatchNorm2d(128)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size, padding=1)
        self.bn3_1 = nn.BatchNorm2d(256)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size, padding=1)
        self.bn3_2 = nn.BatchNorm2d(256)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size, padding=1)
        self.bn3_3 = nn.BatchNorm2d(256)

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size, padding=1)
        self.bn4_1 = nn.BatchNorm2d(512)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size, padding=1)
        self.bn4_2 = nn.BatchNorm2d(512)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size, padding=1)
        self.bn4_3 = nn.BatchNorm2d(512)

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size, padding=1)
        self.bn5_1 = nn.BatchNorm2d(512)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size, padding=1)
        self.bn5_2 = nn.BatchNorm2d(512)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size, padding=1)
        self.bn5_3 = nn.BatchNorm2d(512)

        # Decoder
        self.conv5_3_D = nn.Conv2d(512, 512, kernel_size, padding=1)
        self.bn5_3_D = nn.BatchNorm2d(512)
        self.conv5_2_D = nn.Conv2d(512, 512, kernel_size, padding=1)
        self.bn5_2_D = nn.BatchNorm2d(512)
        self.conv5_1_D = nn.Conv2d(512, 512, kernel_size, padding=1)
        self.bn5_1_D = nn.BatchNorm2d(512)

        self.conv4_3_D = nn.Conv2d(512, 512, kernel_size, padding=1)
        self.bn4_3_D = nn.BatchNorm2d(512)
        self.conv4_2_D = nn.Conv2d(512, 512, kernel_size, padding=1)
        self.bn4_2_D = nn.BatchNorm2d(512)
        self.conv4_1_D = nn.Conv2d(512, 256, kernel_size, padding=1)
        self.bn4_1_D = nn.BatchNorm2d(256)

        self.conv3_3_D = nn.Conv2d(256, 256, kernel_size, padding=1)
        self.bn3_3_D = nn.BatchNorm2d(256)
        self.conv3_2_D = nn.Conv2d(256, 256, kernel_size, padding=1)
        self.bn3_2_D = nn.BatchNorm2d(256)
        self.conv3_1_D = nn.Conv2d(256, 128, kernel_size, padding=1)
        self.bn3_1_D = nn.BatchNorm2d(128)

        self.conv2_2_D = nn.Conv2d(128, 128, kernel_size, padding=1)
        self.bn2_2_D = nn.BatchNorm2d(128)
        self.conv2_1_D = nn.Conv2d(128, 64, kernel_size, padding=1)
        self.bn2_1_D = nn.BatchNorm2d(64)

        self.conv1_2_D = nn.Conv2d(64, 64, kernel_size, padding=1)
        self.bn1_2_D = nn.BatchNorm2d(64)

        self.classifier = nn.Conv2d(64, n_labels, 1)

        # Use MaxPool without indices (ONNX-friendly)
        self.pool = nn.MaxPool2d(2, 2, return_indices=False)

    def forward(self, x):
        # Encoder
        x = F.relu(self.bn1_1(self.conv1_1(x)))
        x = F.relu(self.bn1_2(self.conv1_2(x)))
        x = self.pool(x)

        x = F.relu(self.bn2_1(self.conv2_1(x)))
        x = F.relu(self.bn2_2(self.conv2_2(x)))
        x = self.pool(x)

        x = F.relu(self.bn3_1(self.conv3_1(x)))
        x = F.relu(self.bn3_2(self.conv3_2(x)))
        x = F.relu(self.bn3_3(self.conv3_3(x)))
        x = self.pool(x)

        x = F.relu(self.bn4_1(self.conv4_1(x)))
        x = F.relu(self.bn4_2(self.conv4_2(x)))
        x = F.relu(self.bn4_3(self.conv4_3(x)))
        x = self.pool(x)

        x = F.relu(self.bn5_1(self.conv5_1(x)))
        x = F.relu(self.bn5_2(self.conv5_2(x)))
        x = F.relu(self.bn5_3(self.conv5_3(x)))
        x = self.pool(x)

        # Decoder using nearest neighbor upsampling (ONNX-friendly)
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = F.relu(self.bn5_3_D(self.conv5_3_D(x)))
        x = F.relu(self.bn5_2_D(self.conv5_2_D(x)))
        x = F.relu(self.bn5_1_D(self.conv5_1_D(x)))

        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = F.relu(self.bn4_3_D(self.conv4_3_D(x)))
        x = F.relu(self.bn4_2_D(self.conv4_2_D(x)))
        x = F.relu(self.bn4_1_D(self.conv4_1_D(x)))

        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = F.relu(self.bn3_3_D(self.conv3_3_D(x)))
        x = F.relu(self.bn3_2_D(self.conv3_2_D(x)))
        x = F.relu(self.bn3_1_D(self.conv3_1_D(x)))

        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = F.relu(self.bn2_2_D(self.conv2_2_D(x)))
        x = F.relu(self.bn2_1_D(self.conv2_1_D(x)))

        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = F.relu(self.bn1_2_D(self.conv1_2_D(x)))

        return self.classifier(x)

# ===================================================================
#  Activation Profiler
# ===================================================================
class ActivationProfiler:
    def __init__(self):
        self.activations = []
        self.hooks = []

    def register_hooks(self, model):
        def hook_fn(module, input, output):
            try:
                if isinstance(output, torch.Tensor):
                    size_bytes = output.element_size() * output.nelement()
                    self.activations.append({
                        'module': module.__class__.__name__,
                        'shape': tuple(output.shape),
                        'size_mb': size_bytes / (1024**2),
                        'dtype': str(output.dtype)
                    })
            except Exception:
                pass

        for name, module in model.named_modules():
            if len(list(module.children())) == 0:
                try:
                    h = module.register_forward_hook(hook_fn)
                    self.hooks.append(h)
                except Exception:
                    pass

    def clear(self):
        self.activations = []

    def remove_hooks(self):
        for h in self.hooks:
            try:
                h.remove()
            except Exception:
                pass
        self.hooks = []

    def get_stats(self):
        if not self.activations:
            return None
        total_mb = sum(a['size_mb'] for a in self.activations)
        peak_mb = max(a['size_mb'] for a in self.activations)
        return {
            'total_activations': len(self.activations),
            'total_memory_mb': total_mb,
            'peak_activation_mb': peak_mb,
            'average_activation_mb': total_mb / len(self.activations)
        }

# ===================================================================
#  FLOPs Profiler (approximate)
# ===================================================================
class FlopProfiler:
    def __init__(self):
        self.hooks = []
        self.layer_flops = []

    def _conv_flops(self, module, inp, out):
        try:
            x = inp[0]
            out_t = out
            batch = x.shape[0]
            Cout = out_t.shape[1]
            H_out = out_t.shape[2]
            W_out = out_t.shape[3]
            if isinstance(module, nn.Conv2d):
                Cin = module.in_channels
                kh, kw = module.kernel_size if isinstance(module.kernel_size, tuple) else (module.kernel_size, module.kernel_size)
                groups = module.groups if hasattr(module, 'groups') else 1
                macs = batch * Cout * H_out * W_out * (Cin // groups) * kh * kw
                flops = 2 * macs
                self.layer_flops.append(('Conv2d', flops))
            elif isinstance(module, nn.ConvTranspose2d):
                Cin = module.in_channels
                kh, kw = module.kernel_size if isinstance(module.kernel_size, tuple) else (module.kernel_size, module.kernel_size)
                groups = module.groups if hasattr(module, 'groups') else 1
                macs = batch * out_t.shape[1] * H_out * W_out * (Cin // groups) * kh * kw
                flops = 2 * macs
                self.layer_flops.append(('ConvTranspose2d', flops))
        except Exception:
            pass

    def _linear_flops(self, module, inp, out):
        try:
            x = inp[0]
            batch = x.shape[0] if x.dim() > 1 else 1
            in_features = module.in_features
            out_features = module.out_features
            macs = batch * in_features * out_features
            flops = 2 * macs
            self.layer_flops.append(('Linear', flops))
        except Exception:
            pass

    def register_hooks(self, model):
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                try:
                    h = module.register_forward_hook(lambda m, i, o, mm=module: self._conv_flops(mm, i, o))
                    self.hooks.append(h)
                except Exception:
                    pass
            elif isinstance(module, nn.Linear):
                try:
                    h = module.register_forward_hook(lambda m, i, o, mm=module: self._linear_flops(mm, i, o))
                    self.hooks.append(h)
                except Exception:
                    pass

    def remove_hooks(self):
        for h in self.hooks:
            try:
                h.remove()
            except Exception:
                pass
        self.hooks = []

    def total_flops(self):
        return sum(f for _, f in self.layer_flops)

# -------------------------  Calibration Data Reader  -------------------------
if ORT_QUANT_AVAILABLE and CalibrationDataReader is not None:
    class RandomCalibrationDataReader(CalibrationDataReader):
        def __init__(self, input_name, input_shape, num_samples=32):
            self.input_name = input_name
            self.input_shape = input_shape
            self.num_samples = int(num_samples)
            self._idx = 0

        def get_next(self):
            if self._idx >= self.num_samples:
                return None
            self._idx += 1
            data = np.random.randn(*self.input_shape).astype(np.float32)
            return {self.input_name: data}
else:
    class RandomCalibrationDataReader(object):
        def __init__(self, input_name, input_shape, num_samples=32):
            self.input_name = input_name
            self.input_shape = input_shape
            self.num_samples = int(num_samples)
            self._idx = 0

        def get_next(self):
            if self._idx >= self.num_samples:
                return None
            self._idx += 1
            data = np.random.randn(*self.input_shape).astype(np.float32)
            return {self.input_name: data}

# -------------------------  ONNX Utilities  -------------------------
def get_model_size_mb(path):
    if os.path.exists(path):
        return os.path.getsize(path) / (1024**2)
    return 0.0

def analyze_onnx_model(onnx_path):
    if not ONNX_AVAILABLE or not os.path.exists(onnx_path):
        return None
    try:
        model = onnx.load(onnx_path)
        total_params = 0
        for init in model.graph.initializer:
            params = 1
            for dim in init.dims:
                params *= dim
            total_params += params
        return {
            'num_nodes': len(model.graph.node),
            'num_params': total_params,
            'file_size_mb': get_model_size_mb(onnx_path)
        }
    except Exception as e:
        print("[!] analyze_onnx_model failed:", e)
        return None

def estimate_onnx_activation_memory(onnx_path, dtype_bits=8):
    if not ONNX_AVAILABLE or not os.path.exists(onnx_path):
        return None
    try:
        model = onnx.load(onnx_path)
        try:
            model = onnx.shape_inference.infer_shapes(model)
        except Exception:
            pass
        tensors = []
        names_seen = set()

        def collect_vi(vi_list):
            for vi in vi_list:
                name = vi.name
                if name in names_seen:
                    continue
                names_seen.add(name)
                shape = None
                try:
                    tp = vi.type.tensor_type
                    dims = [d.dim_value for d in tp.shape.dim]
                    if any(d == 0 for d in dims):
                        continue
                    shape = tuple(dims)
                except Exception:
                    shape = None
                if shape:
                    tensors.append((name, shape))

        collect_vi(model.graph.value_info)
        collect_vi(model.graph.output)
        collect_vi(model.graph.input)

        bytes_per_elem = dtype_bits / 8.0
        activations = []
        total_mb = 0.0
        for name, shape in tensors:
            nelems = 1
            for d in shape:
                nelems *= d
            size_mb = (nelems * bytes_per_elem) / (1024**2)
            activations.append({'name': name, 'shape': shape, 'size_mb': size_mb})
            total_mb += size_mb

        if not activations:
            return None
        activations = sorted(activations, key=lambda x: x['size_mb'], reverse=True)
        return {
            'dtype_bits': dtype_bits,
            'num_tensors': len(activations),
            'total_memory_mb': total_mb,
            'top_activations': activations[:10]
        }
    except Exception as e:
        print("[!] estimate_onnx_activation_memory failed:", e)
        return None

# -------------------------  ONNX Export / Quant  -------------------------
def export_to_onnx(model, onnx_path, input_size=(1,3,640,360), input_name='input', opset=13):
    if not ONNX_AVAILABLE:
        raise RuntimeError("onnx python package not available")
    model.eval()
    dummy = torch.randn(*input_size)
    with torch.no_grad():
        torch.onnx.export(
            model, dummy, onnx_path, opset_version=opset,
            input_names=[input_name], output_names=['output'],
            dynamic_axes={input_name: {0: 'batch'}, 'output': {0: 'batch'}},
            do_constant_folding=True,
            export_params=True,
            keep_initializers_as_inputs=False
        )
    print(f"[✓] Exported ONNX -> {onnx_path}")

def try_onnx_quant_static(onnx_in, onnx_out, input_name, input_shape, calib_reader, per_channel=False):
    if not ORT_QUANT_AVAILABLE:
        raise RuntimeError("ONNX Runtime quantization not available")
    print("[*] Running ONNX Runtime static quantization...")
    qformat = None
    try:
        qformat = QuantFormat.QDQ if hasattr(QuantFormat, 'QDQ') else None
    except Exception:
        qformat = None
    quantize_static(
        model_input=onnx_in,
        model_output=onnx_out,
        calibration_data_reader=calib_reader,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        per_channel=per_channel,
        quant_format=qformat
    )
    print("[✓] Static quantization attempted.")

def try_onnx_quant_dynamic(onnx_in, onnx_out):
    if not ORT_QUANT_AVAILABLE:
        raise RuntimeError("ONNX Runtime quantization not available")
    print("[*] Running ONNX dynamic quantization...")
    quantize_dynamic(model_input=onnx_in, model_output=onnx_out, weight_type=QuantType.QInt8)
    print("[✓] quantize_dynamic produced:", onnx_out)

def run_onnxruntime(onnx_path, input_name, input_shape, warmup=10, runs=100):
    if not ORT_AVAILABLE:
        raise RuntimeError("onnxruntime not available")
    providers = ['CPUExecutionProvider']
    print(f"[*] ORT providers: {providers}")
    so = ort.SessionOptions()
    try:
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    except Exception:
        pass
    sess = ort.InferenceSession(onnx_path, sess_options=so, providers=providers)

    batch_shape = (1,) + tuple(input_shape[1:])
    x = np.random.randn(*batch_shape).astype(np.float32)
    feeds = {input_name: x}

    for _ in range(warmup):
        _ = sess.run(None, feeds)

    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        _ = sess.run(None, feeds)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)

    return {
        'mean_ms': float(np.mean(times)),
        'std_ms': float(np.std(times)),
        'min_ms': float(np.min(times)),
        'p50_ms': float(np.percentile(times, 50)),
        'p95_ms': float(np.percentile(times, 95)),
        'p99_ms': float(np.percentile(times, 99)),
        'max_ms': float(np.max(times)),
        'providers': providers
    }

# -------------------------  PyTorch CPU benchmark fallback  -------------------------
def run_torch_cpu(model, input_shape, warmup=10, runs=50, num_threads=None):
    if num_threads is not None:
        torch.set_num_threads(num_threads)
    model_cpu = model.cpu()
    model_cpu.eval()
    x = torch.randn(*input_shape, dtype=torch.float32)
    with torch.no_grad():
        for _ in range(warmup):
            _ = model_cpu(x)
    times = []
    with torch.no_grad():
        for _ in range(runs):
            t0 = time.perf_counter()
            _ = model_cpu(x)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000.0)
    return {
        'mean_ms': float(np.mean(times)),
        'std_ms': float(np.std(times)),
        'min_ms': float(np.min(times)),
        'p50_ms': float(np.percentile(times, 50)),
        'p95_ms': float(np.percentile(times, 95)),
        'p99_ms': float(np.percentile(times, 99)),
        'max_ms': float(np.max(times))
    }

# -------------------------  Table Printer  -------------------------
def print_analysis_table(input_size_str, flops_g, params_m, total_act_mb_fp32, peak_act_mb_fp32, lat_fp32,
                         total_act_mb_int8, peak_act_mb_int8, lat_int8, model_fp32_size_mb=None, model_int8_size_mb=None):
    header = (" Input Size | FLOPs (G) | Params (M) | Total Act FP32 (MB) | Peak Act FP32 (MB) | "
              "Latency FP32 (ms) | Total Act INT8 (MB) | Peak Act INT8 (MB) | Latency INT8 (ms)")
    print(header)
    print("-" * len(header))
    print(f" {input_size_str:10s} | {flops_g:9.3f} | {params_m:10.3f} | {total_act_mb_fp32:19.2f} | "
          f"{peak_act_mb_fp32:18.2f} | {lat_fp32:17.2f} | {total_act_mb_int8:19.2f} | "
          f"{peak_act_mb_int8:18.2f} | {lat_int8:16.2f}")

# -------------------------  Process Single Resolution  -------------------------
def process_resolution(resolution, num_classes=21, calib_size=16, per_channel=False, warmup=5, runs=25, torch_threads=1):
    c, h, w = resolution
    input_shape = (1, c, h, w)
    input_size_str = f"{w}x{h}"
    print(f"\n{'='*80}")
    print(f"Processing resolution: {input_size_str}")
    print(f"{'='*80}")

    model = SegNet(input_channels=c, n_labels=num_classes)
    model.eval()

    # quick forward to ensure model works at this resolution
    with torch.no_grad():
        try:
            _ = model(torch.randn(*input_shape))
        except Exception as e:
            print(f"[!] Model forward failed at this resolution: {e}")
            return None

    total_params = sum(p.numel() for p in model.parameters())
    params_m = total_params / 1e6

    # Profilers
    flop_prof = FlopProfiler()
    act_prof = ActivationProfiler()
    flop_prof.register_hooks(model)
    act_prof.register_hooks(model)

    with torch.no_grad():
        _ = model(torch.randn(*input_shape))

    flops_total = flop_prof.total_flops()
    flop_prof.remove_hooks()
    act_stats = act_prof.get_stats() or {'total_memory_mb': 0.0, 'peak_activation_mb': 0.0}
    act_prof.remove_hooks()
    flops_g = flops_total / 1e9

    total_act_mb_fp32 = act_stats['total_memory_mb']
    peak_act_mb_fp32 = act_stats['peak_activation_mb']

    onnx_path = f"segnet_{w}x{h}.onnx"
    quant_path = f"segnet_{w}x{h}_int8.onnx"
    fp32_info = None

    # Try ONNX export if onnx lib is available
    if ONNX_AVAILABLE:
        try:
            export_to_onnx(model, onnx_path, input_size=input_shape, input_name='input', opset=13)
            fp32_info = analyze_onnx_model(onnx_path)
            if fp32_info:
                print(f"    FP32 ONNX: {fp32_info['num_nodes']} nodes, {fp32_info['num_params']:,} params, {fp32_info['file_size_mb']:.2f} MB")
        except Exception as e:
            print("[!] ONNX export failed:", e)
    else:
        print("[i] onnx python package not available — skipping ONNX export.")

    int8_est_total_mb = 0.0
    int8_est_peak_mb = 0.0
    if os.path.exists(onnx_path) and ONNX_AVAILABLE:
        static_est = estimate_onnx_activation_memory(onnx_path, dtype_bits=8)
        if static_est:
            int8_est_total_mb = static_est['total_memory_mb']
            if static_est.get('top_activations'):
                int8_est_peak_mb = static_est['top_activations'][0]['size_mb']

    # Calibration dataset for static quant (synthetic)
    calib_reader = RandomCalibrationDataReader('input', input_shape, num_samples=calib_size)

    did_onnx_quant = False
    if ORT_QUANT_AVAILABLE and os.path.exists(onnx_path):
        try:
            try_onnx_quant_static(onnx_path, quant_path, 'input', input_shape, calib_reader, per_channel=per_channel)
            did_onnx_quant = os.path.exists(quant_path)
        except Exception as e_static:
            print("[!] static quant failed, trying dynamic:", e_static)
            try:
                try_onnx_quant_dynamic(onnx_path, quant_path)
                did_onnx_quant = os.path.exists(quant_path)
            except Exception as e_dyn:
                print("[✗] dynamic quant also failed:", e_dyn)
                did_onnx_quant = False

        if did_onnx_quant:
            print(f"[✓] ONNX quantization produced: {quant_path}")
            try:
                int8_info = analyze_onnx_model(quant_path)
                if int8_info:
                    print(f"    INT8 ONNX: {int8_info['num_nodes']} nodes, {int8_info['file_size_mb']:.2f} MB")
            except Exception:
                pass
    else:
        if not ORT_QUANT_AVAILABLE:
            print("[i] ONNX Runtime quantization package not available — skipping quantization.")
        elif not os.path.exists(onnx_path):
            print("[i] FP32 ONNX model not present; cannot quantize.")

    # Benchmarking: prefer ORT if available else PyTorch fallback
    lat_fp32 = 0.0
    lat_int8 = 0.0
    fp32_model_size = get_model_size_mb(onnx_path) if os.path.exists(onnx_path) else None
    int8_model_size = get_model_size_mb(quant_path) if os.path.exists(quant_path) else None

    if ONNX_AVAILABLE and os.path.exists(onnx_path) and ORT_AVAILABLE:
        try:
            fp32_results = run_onnxruntime(onnx_path, 'input', input_shape, warmup=warmup, runs=runs)
            lat_fp32 = fp32_results['mean_ms']
        except Exception as e:
            print("[!] FP32 ONNX benchmark failed:", e)

    # PyTorch fallback
    if lat_fp32 == 0.0:
        try:
            print("[*] Running PyTorch CPU FP32 benchmark (fallback)...")
            t = run_torch_cpu(model, input_shape, warmup=warmup, runs=runs, num_threads=torch_threads)
            lat_fp32 = t['mean_ms']
        except Exception as e:
            print("[!] PyTorch CPU benchmark failed:", e)
            lat_fp32 = 0.0

    if did_onnx_quant and os.path.exists(quant_path) and ORT_AVAILABLE:
        try:
            int8_results = run_onnxruntime(quant_path, 'input', input_shape, warmup=warmup, runs=runs)
            lat_int8 = int8_results['mean_ms']
        except Exception as e:
            print("[!] INT8 ONNX benchmark failed:", e)

    # We cannot reliably simulate INT8 PyTorch runtime here; leave lat_int8 as 0.0 if unavailable.

    print(f"\nRESULTS FOR {input_size_str}")
    print_analysis_table(
        input_size_str,
        flops_g,
        params_m,
        total_act_mb_fp32,
        peak_act_mb_fp32,
        lat_fp32,
        int8_est_total_mb,
        int8_est_peak_mb,
        lat_int8,
        model_fp32_size_mb=fp32_model_size,
        model_int8_size_mb=int8_model_size
    )

    return {
        'resolution': input_size_str,
        'flops_g': flops_g,
        'params_m': params_m,
        'total_act_fp32_mb': total_act_mb_fp32,
        'peak_act_fp32_mb': peak_act_mb_fp32,
        'lat_fp32_ms': lat_fp32,
        'total_act_int8_mb': int8_est_total_mb,
        'peak_act_int8_mb': int8_est_peak_mb,
        'lat_int8_ms': lat_int8,
        'fp32_size_mb': fp32_model_size,
        'int8_size_mb': int8_model_size
    }

# -------------------------  Main  -------------------------
def main():
    print("\n" + "="*80)
    print("SEGNET INT8 QUANTIZATION & BENCHMARK PIPELINE (CPU-OPTIMIZED) - v3")
    print("="*80)

    # Default resolutions (keep heavy ones commented for faster runs)
    resolutions = [
        (3, 640, 360),
        (3, 1280, 720),
        (3, 1920, 1080),
        (3, 1360, 760),
        (3, 1600, 900),
        (3, 2048, 1152),
        (3, 2560, 1440),
        (3, 3840, 2160)
    ]

    num_classes = 21
    calib_size = 8
    per_channel = False
    warmup = 3
    runs = 20
    torch_threads = 1

    print(f"Resolutions to test: {len(resolutions)}")
    print(f"Number of classes: {num_classes}")
    print(f"Calibration samples: {calib_size}")
    print(f"Benchmark runs: {runs} (warmup: {warmup})")
    print(f"ONNX available: {ONNX_AVAILABLE} | ORT available: {ORT_AVAILABLE} | ORT quant available: {ORT_QUANT_AVAILABLE}")
    print("="*80 + "\n")

    all_results = []
    for res in resolutions:
        try:
            r = process_resolution(res, num_classes, calib_size, per_channel, warmup, runs, torch_threads=torch_threads)
            if r:
                all_results.append(r)
        except Exception as e:
            print(f"[✗] Failed to process {res}: {e}")
            continue

    # Summary
    print("\n" + "="*80)
    print("SUMMARY - ALL RESOLUTIONS")
    print("="*80)
    header = (" Input Size | FLOPs (G) | Params (M) | Total Act FP32 (MB) | Peak Act FP32 (MB) | "
              "Latency FP32 (ms) | Total Act INT8 (MB) | Peak Act INT8 (MB) | Latency INT8 (ms)")
    print(header)
    print("-" * len(header))
    for r in all_results:
        print(f" {r['resolution']:10s} | {r['flops_g']:9.3f} | {r['params_m']:10.3f} | "
              f"{r['total_act_fp32_mb']:19.2f} | {r['peak_act_fp32_mb']:18.2f} | {r['lat_fp32_ms']:17.2f} | "
              f"{r['total_act_int8_mb']:19.2f} | {r['peak_act_int8_mb']:18.2f} | {r['lat_int8_ms']:16.2f}")

    print("\n" + "="*80)
    print("BENCHMARK COMPLETE!")
    print("="*80)

    print("\nSPEEDUP ANALYSIS (FP32 vs INT8):")
    print("-" * 60)
    for r in all_results:
        if r['lat_fp32_ms'] > 0 and r['lat_int8_ms'] > 0:
            speedup = r['lat_fp32_ms'] / r['lat_int8_ms']
            print(f"{r['resolution']:10s}: {speedup:5.2f}x speedup (FP32: {r['lat_fp32_ms']:6.2f}ms -> INT8: {r['lat_int8_ms']:6.2f}ms)")
        else:
            print(f"{r['resolution']:10s}: N/A")

    print("\nMODEL SIZE COMPARISON:")
    print("-" * 60)
    for r in all_results:
        if r['fp32_size_mb'] and r['int8_size_mb']:
            try:
                compression = r['fp32_size_mb'] / r['int8_size_mb']
                print(f"{r['resolution']:10s}: FP32: {r['fp32_size_mb']:6.2f} MB -> INT8: {r['int8_size_mb']:6.2f} MB ({compression:.2f}x compression)")
            except Exception:
                pass

    print("\n" + "="*80)
    print("Note: INT8 runtime timing requires ONNXRuntime (ORT) with a CPU provider or a purpose-built INT8 runtime.")
    print("If ORT is not installed, the script falls back to PyTorch CPU FP32 timings (useful but not representative of INT8).")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
