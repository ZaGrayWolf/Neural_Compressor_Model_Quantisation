#!/usr/bin/env python3
"""
LinkNet INT8 Quantisation & Benchmark Pipeline (CPU-optimised)
This script uses the LinkNet model definition but applies the exact same
profiling, ONNX export, quantization, and benchmarking pipeline as
the RTFormer-Slim script.
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights
import warnings
warnings.filterwarnings("ignore")

# ----------  ONNX / ORT imports (same as RTFormer script)  ----------
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
#  LinkNet model definition
# (Replaces the RTFormer-Slim model definition)
# ===================================================================

class DecoderBlock(nn.Module):
    """
    Decoder block for LinkNet with adaptive upsampling.
    This block takes in features from the previous decoder block and upsamples them
    to match the spatial dimensions of the corresponding encoder skip connection.
    """
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        
        # 1x1 convolution to reduce channel dimensions
        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nn.ReLU(inplace=True)
        
        # 3x3 transposed convolution for upsampling
        self.deconv = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 
                                         kernel_size=3, stride=2, padding=1, 
                                         output_padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nn.ReLU(inplace=True)
        
        # 1x1 convolution to expand channels to match encoder output
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
        
        # If target_size is provided, use interpolation to match exact dimensions
        # This is critical for handling arbitrary input sizes
        if target_size is not None and (x.shape[2:] != target_size):
            x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        return x

class LinkNet(nn.Module):
    """
    The LinkNet architecture for semantic segmentation, built upon a ResNet-18 encoder.
    Fixed to handle arbitrary input dimensions by using adaptive upsampling.
    """
    def __init__(self, num_classes=19, pretrained=True):
        super(LinkNet, self).__init__()
        
        # Load a ResNet-18 model
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        resnet = resnet18(weights=weights)
        
        # ----------------- Encoder -----------------
        self.encoder_conv1 = resnet.conv1
        self.encoder_bn1 = resnet.bn1
        self.encoder_relu = resnet.relu
        self.encoder_maxpool = resnet.maxpool
        
        self.encoder_layer1 = resnet.layer1  # Output channels: 64
        self.encoder_layer2 = resnet.layer2  # Output channels: 128
        self.encoder_layer3 = resnet.layer3  # Output channels: 256
        self.encoder_layer4 = resnet.layer4  # Output channels: 512
        
        # ----------------- Decoder -----------------
        self.decoder_layer4 = DecoderBlock(512, 256)
        self.decoder_layer3 = DecoderBlock(256, 128)
        self.decoder_layer2 = DecoderBlock(128, 64)
        self.decoder_layer1 = DecoderBlock(64, 64)
        
        # ----------------- Final Upsampling Layers -----------------
        self.final_deconv1 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        self.final_bn1 = nn.BatchNorm2d(32)
        self.final_relu1 = nn.ReLU(inplace=True)
        
        self.final_conv = nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False)
        self.final_bn2 = nn.BatchNorm2d(32)
        self.final_relu2 = nn.ReLU(inplace=True)
        
        self.final_deconv2 = nn.ConvTranspose2d(32, num_classes, kernel_size=2, stride=2, bias=False)

    def forward(self, x):
        # Store original input size for final upsampling
        input_size = x.shape[2:]
        
        # ----------------- Encoder Path -----------------
        e0 = self.encoder_conv1(x)
        e0 = self.encoder_bn1(e0)
        e0 = self.encoder_relu(e0)
        
        e0_pool = self.encoder_maxpool(e0)
        
        e1 = self.encoder_layer1(e0_pool)
        e2 = self.encoder_layer2(e1)
        e3 = self.encoder_layer3(e2)
        e4 = self.encoder_layer4(e3) # Bottleneck
        
        # ----------------- Decoder Path with Adaptive Skip Connections -----------------
        # Pass target sizes to ensure exact dimension matching
        d4 = self.decoder_layer4(e4, target_size=e3.shape[2:]) + e3
        d3 = self.decoder_layer3(d4, target_size=e2.shape[2:]) + e2
        d2 = self.decoder_layer2(d3, target_size=e1.shape[2:]) + e1
        d1 = self.decoder_layer1(d2, target_size=e0.shape[2:]) + e0
        
        # ----------------- Final Upsampling -----------------
        f1 = self.final_deconv1(d1)
        f1 = self.final_bn1(f1)
        f1 = self.final_relu1(f1)
        
        f2 = self.final_conv(f1)
        f2 = self.final_bn2(f2)
        f2 = self.final_relu2(f2)
        
        # Final output - use interpolation to match exact input size
        out = self.final_deconv2(f2)
        if out.shape[2:] != input_size:
            out = F.interpolate(out, size=input_size, mode='bilinear', align_corners=False)
        
        return out

# ===================================================================
#  Everything below is the **identical** pipeline from the RTFormer script
# ===================================================================

# -------------------------  Activation Memory Profiler  -------------------------
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
            return None
        total_mb = sum(a['size_mb'] for a in self.activations)
        peak_mb = max(a['size_mb'] for a in self.activations)
        return {
            'total_activations': len(self.activations),
            'total_memory_mb': total_mb,
            'peak_activation_mb': peak_mb,
            'average_activation_mb': total_mb / len(self.activations)
        }

# -------------------------  FLOPs Profiler  -------------------------
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
                h = module.register_forward_hook(lambda m, i, o, mm=module: self._conv_flops(mm, i, o))
                self.hooks.append(h)
            elif isinstance(module, nn.Linear):
                h = module.register_forward_hook(lambda m, i, o, mm=module: self._linear_flops(mm, i, o))
                self.hooks.append(h)

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def total_flops(self):
        return sum(f for _, f in self.layer_flops)

# -------------------------  Calibration Data Reader  -------------------------
class RandomCalibrationDataReader(CalibrationDataReader if ORT_QUANT_AVAILABLE else object):
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
def export_to_onnx(model, onnx_path, input_size=(1, 3, 640, 360), input_name='input', opset=13):
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
    qformat = QuantFormat.QDQ if hasattr(QuantFormat, 'QDQ') else None
    quantize_static(
        model_input=onnx_in,
        model_output=onnx_out,
        calibration_data_reader=calib_reader,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        per_channel=per_channel,
        quant_format=qformat
    )

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
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
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
def process_resolution(resolution, num_classes=19, calib_size=64, per_channel=False, warmup=10, runs=100):
    c, h, w = resolution
    input_shape = (1, c, h, w)
    input_size_str = f"{w}x{h}"

    print(f"\n{'='*80}")
    print(f"Processing resolution: {input_size_str}")
    print(f"{'='*80}")

    # *** CHANGED MODEL ***
    model = LinkNet(num_classes=num_classes, pretrained=False)
    model.eval()

    with torch.no_grad():
        out = model(torch.randn(*input_shape))
    print(f"[✓] Model created, output shape: {tuple(out.shape)}")

    total_params = sum(p.numel() for p in model.parameters())
    params_m = total_params / 1e6

    flop_prof = FlopProfiler()
    act_prof = ActivationProfiler()
    flop_prof.register_hooks(model)
    act_prof.register_hooks(model)

    with torch.no_grad():
        _ = model(torch.randn(*input_shape))

    flops_total = flop_prof.total_flops()
    flop_prof.remove_hooks()
    act_stats = act_prof.get_stats() or {'total_memory_mb': 0.0, 'peak_activation_mb': 0.0}
    flops_g = flops_total / 1e9

    total_act_mb_fp32 = act_stats['total_memory_mb']
    peak_act_mb_fp32 = act_stats['peak_activation_mb']

    # *** CHANGED PATHS ***
    onnx_path = f"linknet_{w}x{h}.onnx"
    quant_path = f"linknet_{w}x{h}_int8.onnx"

    try:
        export_to_onnx(model, onnx_path, input_shape)
        fp32_info = analyze_onnx_model(onnx_path)
        if fp32_info:
            print(f"    FP32 ONNX: {fp32_info['num_nodes']} nodes, {fp32_info['num_params']:,} params, {fp32_info['file_size_mb']:.2f} MB")
    except Exception as e:
        print(f"[✗] ONNX export failed: {e}")
        fp32_info = None

    int8_est_total_mb = 0.0
    int8_est_peak_mb = 0.0
    if os.path.exists(onnx_path):
        static_est = estimate_onnx_activation_memory(onnx_path, dtype_bits=8)
        if static_est:
            int8_est_total_mb = static_est['total_memory_mb']
            if static_est['top_activations']:
                int8_est_peak_mb = static_est['top_activations'][0]['size_mb']

    print("[*] Using synthetic random calibration")
    calib_reader = RandomCalibrationDataReader('input', input_shape, num_samples=calib_size)

    did_onnx_quant = False
    if ORT_QUANT_AVAILABLE:
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
            int8_info = analyze_onnx_model(quant_path)
            if int8_info:
                print(f"    INT8 ONNX: {int8_info['num_nodes']} nodes, {int8_info['file_size_mb']:.2f} MB")
                if fp32_info:
                    compression = fp32_info['file_size_mb'] / int8_info['file_size_mb']
                    print(f"    Compression ratio: {compression:.2f}x")

    lat_fp32 = 0.0
    fp32_model_size = get_model_size_mb(onnx_path) if os.path.exists(onnx_path) else None
    if os.path.exists(onnx_path) and ORT_AVAILABLE:
        try:
            fp32_results = run_onnxruntime(onnx_path, 'input', input_shape, warmup=warmup, runs=runs)
            lat_fp32 = fp32_results['mean_ms']
        except Exception as e:
            print("[!] FP32 ONNX benchmark failed:", e)

    lat_int8 = 0.0
    int8_model_size = get_model_size_mb(quant_path) if os.path.exists(quant_path) else None
    int8_runtime_total_mb = int8_est_total_mb
    int8_runtime_peak_mb = int8_est_peak_mb

    if did_onnx_quant and os.path.exists(quant_path) and ORT_AVAILABLE:
        try:
            int8_results = run_onnxruntime(quant_path, 'input', input_shape, warmup=warmup, runs=runs)
            lat_int8 = int8_results['mean_ms']
        except Exception as e:
            print("[!] INT8 ONNX benchmark failed:", e)

    print(f"\nRESULTS FOR {input_size_str}")
    print_analysis_table(
        input_size_str,
        flops_g,
        params_m,
        total_act_mb_fp32,
        peak_act_mb_fp32,
        lat_fp32,
        int8_runtime_total_mb,
        int8_runtime_peak_mb,
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
        'total_act_int8_mb': int8_runtime_total_mb,
        'peak_act_int8_mb': int8_runtime_peak_mb,
        'lat_int8_ms': lat_int8,
        'fp32_size_mb': fp32_model_size,
        'int8_size_mb': int8_model_size
    }

# -------------------------  Main  -------------------------
def main():
    print("\n" + "="*80)
    # *** CHANGED TITLE ***
    print("LINKNET INT8 QUANTIZATION & BENCHMARK PIPELINE (CPU-OPTIMIZED)")
    print("="*80)

    resolutions = [
        (3, 640, 360),
        (3, 1280, 720),
        (3, 1360, 760),
        (3, 1600, 900),
        (3, 1920, 1080),
        (3, 2048, 1152),
        (3, 2560, 1440),
        (3, 3840, 2160)
    ]

    num_classes = 19  # Default for Cityscapes, matching LinkNet script
    calib_size = 64
    per_channel = False
    warmup = 10
    runs = 100

    print(f"Resolutions to test: {len(resolutions)}")
    print(f"Number of classes: {num_classes}")
    print(f"Calibration samples: {calib_size}")
    print(f"Benchmark runs: {runs} (warmup: {warmup})")
    print("="*80 + "\n")

    all_results = []
    for res in resolutions:
        try:
            result = process_resolution(res, num_classes, calib_size, per_channel, warmup, runs)
            all_results.append(result)
        except Exception as e:
            print(f"[✗] Failed to process {res}: {e}")
            continue

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
            compression = r['fp32_size_mb'] / r['int8_size_mb']
            print(f"{r['resolution']:10s}: FP32: {r['fp32_size_mb']:6.2f} MB -> INT8: {r['int8_size_mb']:6.2f} MB ({compression:.2f}x compression)")

    print("\n" + "="*80)
    print("All ONNX models saved in current directory:")
    # *** CHANGED PATHS ***
    print("  - linknet_<width>x<height>.onnx (FP32)")
    print("  - linknet_<width>x<height>_int8.onnx (INT8)")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()