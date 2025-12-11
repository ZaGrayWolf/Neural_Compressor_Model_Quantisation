#!/usr/bin/env python3
"""
ENet (Efficient Neural Network) ONNX/INT8 Quantisation & Benchmark Pipeline
This script integrates the ENet model with the comprehensive profiling and 
quantization pipeline, including a fix for the NameError in activation memory calculation.
"""
#NEW VERSION

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
#  ENet model definition
# ===================================================================


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
                 return_indices=False, dropout_prob=0, bias=False, relu=True):
        super().__init__()
        self.return_indices = return_indices
        internal_channels = in_channels // internal_ratio
        activation = nn.ReLU if relu else nn.PReLU
        self.main_max1 = nn.MaxPool2d(2, 2, return_indices=return_indices)
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
        if self.return_indices: 
            main, idx = self.main_max1(x)
        else: 
            main, idx = self.main_max1(x), None
        ext = self.ext_conv1(x)
        ext = self.ext_conv2(ext)
        ext = self.ext_conv3(ext)
        ext = self.ext_regul(ext)
        n, ch_ext, h, w = ext.size()
        ch_main = main.size()[1]
        padding = torch.zeros(n, ch_ext - ch_main, h, w, device=ext.device)
        main = torch.cat((main, padding), 1)
        out = main + ext
        return self.out_activation(out), idx


class UpsamplingBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, internal_ratio=4,
                 dropout_prob=0, bias=False, relu=True):
        super().__init__()
        internal_channels = in_channels // internal_ratio
        activation = nn.ReLU if relu else nn.PReLU
        self.main_conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=bias),
            nn.BatchNorm2d(out_channels))
        self.main_unpool1 = nn.MaxUnpool2d(2)
        self.ext_conv1 = nn.Sequential(
            nn.Conv2d(in_channels, internal_channels, 1, bias=bias),
            nn.BatchNorm2d(internal_channels), activation())
        self.ext_tconv1 = nn.ConvTranspose2d(internal_channels, internal_channels, 2, 2, bias=bias)
        self.ext_tconv1_bnorm = nn.BatchNorm2d(internal_channels)
        self.ext_tconv1_activation = activation()
        self.ext_conv2 = nn.Sequential(
            nn.Conv2d(internal_channels, out_channels, 1, bias=bias),
            nn.BatchNorm2d(out_channels))
        self.ext_regul = nn.Dropout2d(p=dropout_prob)
        self.out_activation = activation()
    def forward(self, x, max_indices, output_size):
        main = self.main_conv1(x)
        main = self.main_unpool1(main, max_indices, output_size=output_size)
        ext = self.ext_conv1(x)
        ext = self.ext_tconv1(ext, output_size=output_size)
        ext = self.ext_tconv1_bnorm(ext)
        ext = self.ext_tconv1_activation(ext)
        ext = self.ext_conv2(ext)
        ext = self.ext_regul(ext)
        out = main + ext
        return self.out_activation(out)


class ENet(nn.Module):
    def __init__(self, num_classes, encoder_relu=False, decoder_relu=True):
        super().__init__()
        self.num_classes = num_classes 
        self.initial_block = InitialBlock(3, 16, relu=encoder_relu)
        self.downsample1_0 = DownsamplingBottleneck(16, 64, return_indices=True, dropout_prob=0.01, relu=encoder_relu)
        self.regular1_1 = RegularBottleneck(64, padding=1, dropout_prob=0.01, relu=encoder_relu)
        self.regular1_2 = RegularBottleneck(64, padding=1, dropout_prob=0.01, relu=encoder_relu)
        self.regular1_3 = RegularBottleneck(64, padding=1, dropout_prob=0.01, relu=encoder_relu)
        self.regular1_4 = RegularBottleneck(64, padding=1, dropout_prob=0.01, relu=encoder_relu)
        self.downsample2_0 = DownsamplingBottleneck(64, 128, return_indices=True, dropout_prob=0.1, relu=encoder_relu)
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
        self.transposed_conv = nn.ConvTranspose2d(16, num_classes, 3, 2, 1, 1, bias=False)
        
    def forward(self, x):
        s0_size = x.size()
        x = self.initial_block(x)
        s1_size = x.size()
        x, idx1 = self.downsample1_0(x)
        x = self.regular1_1(x); x = self.regular1_2(x); x = self.regular1_3(x); x = self.regular1_4(x)
        s2_size = x.size()
        x, idx2 = self.downsample2_0(x)
        x = self.regular2_1(x); x = self.dilated2_2(x); x = self.asymmetric2_3(x); x = self.dilated2_4(x)
        x = self.regular2_5(x); x = self.dilated2_6(x); x = self.asymmetric2_7(x); x = self.dilated2_8(x)
        x = self.regular3_0(x); x = self.dilated3_1(x); x = self.asymmetric3_2(x); x = self.dilated3_3(x)
        x = self.regular3_4(x); x = self.dilated3_5(x); x = self.asymmetric3_6(x); x = self.dilated3_7(x)
        x = self.upsample4_0(x, idx2, output_size=s2_size)
        x = self.regular4_1(x); x = self.regular4_2(x)
        x = self.upsample5_0(x, idx1, output_size=s1_size)
        x = self.regular5_1(x)
        x = self.transposed_conv(x, output_size=s0_size)
        return x


# ===================================================================
# 2. Profiling Utilities
# ===================================================================


class ActivationProfiler:
    def __init__(self):
        self.activations = []
        self.hooks = []

    def register_hooks(self, model):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                output = output[0]
            if isinstance(output, torch.Tensor):
                size_bytes = output.element_size() * output.nelement()
                self.activations.append({
                    'module': module.__class__.__name__,
                    'shape': tuple(output.shape),
                    'size_mb': size_bytes / (1024**2),
                    'dtype': str(output.dtype)
                })
        for name, module in model.named_modules():
            if len(list(module.children())) == 0 and not isinstance(module, ENet):
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
            'total_memory_mb': total_mb,
            'peak_activation_mb': peak_mb,
        }

class FlopProfiler:
    def __init__(self):
        self.hooks = []
        self.layer_flops = []

    def _conv_flops(self, module, inp, out):
        try:
            x = inp[0]
            out_t = out
            if isinstance(out, tuple): out_t = out[0]

            batch = x.shape[0]
            Cout = out_t.shape[1]
            H_out = out_t.shape[2]
            W_out = out_t.shape[3]

            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                Cin = module.in_channels
                kh, kw = module.kernel_size if isinstance(module.kernel_size, tuple) else (module.kernel_size, module.kernel_size)
                groups = module.groups if hasattr(module, 'groups') else 1

                macs = batch * Cout * H_out * W_out * (Cin // groups) * kh * kw
                flops = 2 * macs
                self.layer_flops.append((module.__class__.__name__, flops))
        except Exception:
            pass

    def register_hooks(self, model):
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                h = module.register_forward_hook(lambda m, i, o, mm=module: self._conv_flops(mm, i, o))
                self.hooks.append(h)

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def total_flops(self):
        return sum(f for _, f in self.layer_flops)

def calculate_enet_activation_memory_fixed(input_shape, num_classes):
    total_mb, max_mb, shape = 0.0, 0.0, input_shape
    H, W = shape[1], shape[2]

    shape = (16, H // 2, W // 2)
    mem = (shape[0] * shape[1] * shape[2] * 4) / (1024**2)
    total_mb += mem; max_mb = max(max_mb, mem)

    shape = (64, H // 2, W // 2)
    mem = (shape[0] * shape[1] * shape[2] * 4) / (1024**2)
    total_mb += mem * 5
    max_mb = max(max_mb, mem)

    shape = (128, H // 4, W // 4)
    mem = (shape[0] * shape[1] * shape[2] * 4) / (1024**2)
    total_mb += mem * (1 + 8 + 8)
    max_mb = max(max_mb, mem)

    shape = (64, H // 2, W // 2)
    mem = (shape[0] * shape[1] * shape[2] * 4) / (1024**2)
    total_mb += mem * 3
    max_mb = max(max_mb, mem)

    shape = (16, H, W)
    mem = (shape[0] * shape[1] * shape[2] * 4) / (1024**2)
    total_mb += mem * 2
    max_mb = max(max_mb, mem)

    final_mem = (num_classes * H * W * 4) / (1024**2)
    total_mb += final_mem; max_mb = max(max_mb, final_mem)

    return total_mb, max_mb

def get_model_parameters(model):
    return sum(p.numel() for p in model.parameters())

def get_model_size_mb(path):
    if os.path.exists(path):
        return os.path.getsize(path) / (1024**2)
    return 0.0

class RandomCalibrationDataReader(onnxruntime.quantization.CalibrationDataReader if ORT_QUANT_AVAILABLE else object):
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

def export_to_onnx(model, onnx_path, input_size, input_name='input', opset=13):
    model.eval()
    dummy = torch.randn(*input_size)
    original_forward = ENet.forward

    def onnx_forward(self, x):
        s0_size = x.size()
        x = self.initial_block(x)
        s1_size = x.size()
        x, idx1 = self.downsample1_0(x)
        x = self.regular1_1(x); x = self.regular1_2(x); x = self.regular1_3(x); x = self.regular1_4(x)
        s2_size = x.size()
        x, idx2 = self.downsample2_0(x)
        x = self.regular2_1(x); x = self.dilated2_2(x); x = self.asymmetric2_3(x); x = self.dilated2_4(x)
        x = self.regular2_5(x); x = self.dilated2_6(x); x = self.asymmetric2_7(x); x = self.dilated2_8(x)
        x = self.regular3_0(x); x = self.dilated3_1(x); x = self.asymmetric3_2(x); x = self.dilated3_3(x)
        x = self.regular3_4(x); x = self.dilated3_5(x); x = self.asymmetric3_6(x); x = self.dilated3_7(x)
        warnings.warn("Using nearest neighbor upsampling for ONNX export due to MaxUnpool limitations.")
        target_h_s4 = s2_size[2]; target_w_s4 = s2_size[3]
        x = F.interpolate(x, size=(target_h_s4, target_w_s4), mode='nearest')
        x = self.regular4_1(x); x = self.regular4_2(x)
        target_h_s5 = s1_size[2]; target_w_s5 = s1_size[3]
        x = F.interpolate(x, size=(target_h_s5, target_w_s5), mode='nearest')
        x = self.regular5_1(x)
        x = self.transposed_conv(x, output_size=s0_size)
        return x

    ENet.forward = onnx_forward
    with torch.no_grad():
        torch.onnx.export(
            model, dummy, onnx_path, opset_version=opset,
            input_names=[input_name], output_names=['output'],
            dynamic_axes={input_name: {0: 'batch'}, 'output': {0: 'batch'}},
            do_constant_folding=True,
            export_params=True,
            keep_initializers_as_inputs=False
        )
    ENet.forward = original_forward
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


def print_analysis_table(input_size_str, flops_g, params_m, total_act_mb_fp32, peak_act_mb_fp32, lat_fp32,
                         total_act_mb_int8, peak_act_mb_int8, lat_int8, fp32_size_mb, int8_size_mb):
    header = (" Input Size | FLOPs (G) | Params (M) | Total Act FP32 (MB) | Peak Act FP32 (MB) | Latency FP32 (ms) | "
              "Total Act INT8 (MB) | Peak Act INT8 (MB) | Latency INT8 (ms) | FP32 Size (MB) | INT8 Size (MB)")
    print(header)
    print("-" * len(header))
    print(f" {input_size_str:10s} | {flops_g:9.3f} | {params_m:10.3f} | {total_act_mb_fp32:19.2f} | "
          f"{peak_act_mb_fp32:18.2f} | {lat_fp32:17.2f} | {total_act_mb_int8:19.2f} | "
          f"{peak_act_mb_int8:18.2f} | {lat_int8:17.2f} | {fp32_size_mb:14.2f} | {int8_size_mb:14.2f}")


def process_resolution(resolution, num_classes, calib_size=64, per_channel=False, warmup=10, runs=100):
    c, h, w = resolution  # (C, H, W) format
    input_shape = (1, c, h, w) # PyTorch/ONNX expect (B, C, H, W)
    input_size_str = f"{w}x{h}"

    print(f"\n{'='*80}")
    print(f"Processing resolution: {input_size_str} (Input Shape: {input_shape})")
    print(f"{'='*80}")

    model = ENet(num_classes=num_classes)
    model.eval()

    total_params = get_model_parameters(model)
    params_m = total_params / 1e6

    flop_prof = FlopProfiler()
    act_prof = ActivationProfiler()
    flop_prof.register_hooks(model)
    act_prof.register_hooks(model)

    with torch.no_grad():
        original_forward = ENet.forward
        ENet.forward = lambda self, x: original_forward(self, x)[0] if isinstance(original_forward(self, x), tuple) else original_forward(self, x)
        _ = model(torch.randn(*input_shape))
        ENet.forward = original_forward
    
    flops_total = flop_prof.total_flops()
    flop_prof.remove_hooks()
    act_stats = act_prof.get_stats() or {'total_memory_mb': 0.0, 'peak_activation_mb': 0.0}
    flops_g = flops_total / 1e9

    total_act_mb_fp32, peak_act_mb_fp32 = calculate_enet_activation_memory_fixed((c, h, w), num_classes)

    print(f"[✓] FLOPs (Conv/Deconv only): {flops_g:.3f} G | Params: {params_m:.3f} M")

    onnx_path = f"enet_{w}x{h}.onnx"
    quant_path = f"enet_{w}x{h}_int8.onnx"
    
    try:
        export_to_onnx(model, onnx_path, input_shape) 
    except Exception as e:
        print(f"[✗] ONNX export failed: {e}")

    fp32_size_mb = get_model_size_mb(onnx_path)

    did_onnx_quant = False
    int8_size_mb = 0.0
    int8_est_total_mb = total_act_mb_fp32 / 4  # divide by 4 for INT8 estimate
    int8_est_peak_mb = peak_act_mb_fp32 / 4

    if ORT_QUANT_AVAILABLE and os.path.exists(onnx_path):
        calib_reader = RandomCalibrationDataReader('input', input_shape, num_samples=calib_size)
        try:
            try_onnx_quant_static(onnx_path, quant_path, 'input', input_shape, calib_reader, per_channel=per_channel)
            did_onnx_quant = os.path.exists(quant_path)
        except Exception as e:
            print(f"[!] Static quant failed, trying dynamic: {e}")
            try:
                try_onnx_quant_dynamic(onnx_path, quant_path)
                did_onnx_quant = os.path.exists(quant_path)
            except Exception as e_dyn:
                print(f"[✗] Dynamic quant also failed: {e_dyn}")
        
        int8_size_mb = get_model_size_mb(quant_path) if did_onnx_quant else 0.0

    lat_fp32 = 0.0
    if os.path.exists(onnx_path) and ORT_AVAILABLE:
        try:
            fp32_results = run_onnxruntime(onnx_path, 'input', input_shape, warmup=warmup, runs=runs)
            lat_fp32 = fp32_results['mean_ms']
        except Exception as e:
            print(f"[!] FP32 ONNX benchmark failed: {e}")

    lat_int8 = 0.0
    if did_onnx_quant and os.path.exists(quant_path) and ORT_AVAILABLE:
        try:
            int8_results = run_onnxruntime(quant_path, 'input', input_shape, warmup=warmup, runs=runs)
            lat_int8 = int8_results['mean_ms']
        except Exception as e:
            print(f"[!] INT8 ONNX benchmark failed: {e}")

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
        fp32_size_mb,
        int8_size_mb
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
        'fp32_size_mb': fp32_size_mb,
        'int8_size_mb': int8_size_mb
    }


def main():
    print("\n" + "="*80)
    print("ENET (EFFICIENT NEURAL NETWORK) INT8 QUANTIZATION & BENCHMARK PIPELINE")
    print("="*80)

    resolutions = [(3, 360, 640), (3, 720, 1280), (3, 760, 1360),
                   (3, 900, 1600), (3, 1080, 1920), (3, 1152, 2048),
                   (3, 1440, 2560), (3, 2160, 3840)] 

    NUM_CLASSES = 21 
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
            result = process_resolution(
                res, 
                num_classes=NUM_CLASSES, 
                calib_size=calib_size, 
                warmup=warmup, 
                runs=runs
            )
            all_results.append(result)
        except Exception as e:
            print(f"[✗] Failed to process {res} (HxW: {res[1]}x{res[2]}): {e}")
            continue

    print("\n" + "="*80)
    print("SUMMARY - ALL RESOLUTIONS (ENet)")
    print("="*80)
    header = (" Input Size | FLOPs (G) | Params (M) | Total Act FP32 (MB) | Peak Act FP32 (MB) | Latency FP32 (ms) | "
              "Total Act INT8 (MB) | Peak Act INT8 (MB) | Latency INT8 (ms) | FP32 Size (MB) | INT8 Size (MB)")
    print(header)
    print("-" * len(header))
    for r in all_results:
        print(f" {r['resolution']:10s} | {r['flops_g']:9.3f} | {r['params_m']:10.3f} | "
              f"{r['total_act_fp32_mb']:19.2f} | {r['peak_act_fp32_mb']:18.2f} | {r['lat_fp32_ms']:17.2f} | "
              f"{r['total_act_int8_mb']:19.2f} | {r['peak_act_int8_mb']:18.2f} | {r['lat_int8_ms']:17.2f} | "
              f"{r['fp32_size_mb']:14.2f} | {r['int8_size_mb']:14.2f}")

    print("\n" + "="*80)
    print("BENCHMARK COMPLETE!")
    print("="*80)

    print("\nSPEEDUP AND SIZE ANALYSIS:")
    print("-" * 60)
    for r in all_results:
        speedup = r['lat_fp32_ms'] / r['lat_int8_ms'] if r['lat_int8_ms'] > 0 else 0
        compression = r['fp32_size_mb'] / r['int8_size_mb'] if r['int8_size_mb'] > 0 else 0
        print(f"{r['resolution']:10s}: Speedup: {speedup:5.2f}x | Size Compression: {compression:5.2f}x")


if __name__ == "__main__":
    main()
