#!/usr/bin/env python3
"""
ENet (Efficient Neural Network) FIXED INT8 PIPELINE
Fixes: 
1. Solves Channel Mismatch by fixing UpsamplingBottleneck logic instead of skipping it.
2. Supports ONNX export by swapping MaxUnpool2d for Interpolate dynamically.
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
#  ENet Model Definition (Fixed for ONNX)
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
        
        # Main branch
        self.main_conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=bias),
            nn.BatchNorm2d(out_channels))
        self.main_unpool1 = nn.MaxUnpool2d(2)
        
        # Extension branch
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

    def forward(self, x, max_indices=None, output_size=None):
        # --- Main Branch ---
        main = self.main_conv1(x)
        
        # CRITICAL FIX: If no indices provided (ONNX export mode), use Interpolate instead of MaxUnpool
        if max_indices is not None:
            main = self.main_unpool1(main, max_indices, output_size=output_size)
        else:
            # Upsample using Nearest Neighbor to match MaxUnpool spatial logic
            main = F.interpolate(main, size=output_size, mode='nearest')

        # --- Extension Branch ---
        # ConvTranspose2d handles upsampling naturally, so no changes needed here
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
        self.export_mode = False # Flag for ONNX export logic

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
        s0_size = x.size()[2:] # Only spatial dims H,W
        x = self.initial_block(x)
        
        s1_size = x.size()[2:]
        x, idx1 = self.downsample1_0(x)
        x = self.regular1_1(x); x = self.regular1_2(x); x = self.regular1_3(x); x = self.regular1_4(x)
        
        s2_size = x.size()[2:]
        x, idx2 = self.downsample2_0(x)
        x = self.regular2_1(x); x = self.dilated2_2(x); x = self.asymmetric2_3(x); x = self.dilated2_4(x)
        x = self.regular2_5(x); x = self.dilated2_6(x); x = self.asymmetric2_7(x); x = self.dilated2_8(x)
        x = self.regular3_0(x); x = self.dilated3_1(x); x = self.asymmetric3_2(x); x = self.dilated3_3(x)
        x = self.regular3_4(x); x = self.dilated3_5(x); x = self.asymmetric3_6(x); x = self.dilated3_7(x)
        
        # Logic to skip indices if in export mode
        use_idx2 = None if self.export_mode else idx2
        use_idx1 = None if self.export_mode else idx1
        
        x = self.upsample4_0(x, use_idx2, output_size=s2_size)
        x = self.regular4_1(x); x = self.regular4_2(x)
        x = self.upsample5_0(x, use_idx1, output_size=s1_size)
        x = self.regular5_1(x)
        x = self.transposed_conv(x, output_size=s0_size)
        return x

# ===================================================================
#  Pipeline Utils
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
                self.activations.append({'module': module.__class__.__name__, 'size_mb': size_bytes / (1024**2)})
        for name, module in model.named_modules():
            if len(list(module.children())) == 0 and not isinstance(module, ENet):
                hook = module.register_forward_hook(hook_fn)
                self.hooks.append(hook)

    def remove_hooks(self):
        for h in self.hooks: h.remove()
        self.hooks = []

    def get_stats(self):
        if not self.activations: return None
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
            if isinstance(out, tuple): out_t = out[0]
            batch, Cout, H_out, W_out = out_t.shape
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

    def remove_hooks(self):
        for h in self.hooks: h.remove()
        self.hooks = []

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

def export_to_onnx(model, onnx_path, input_size):
    model.eval()
    model.export_mode = True # ENABLE ONNX MODE (drops MaxUnpool indices)
    dummy = torch.randn(*input_size)
    try:
        torch.onnx.export(model, dummy, onnx_path, opset_version=13, input_names=['input'], output_names=['output'],
                          dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}},
                          do_constant_folding=True, export_params=True)
        print(f"[✓] Exported: {onnx_path}")
    except Exception as e: print(f"[✗] Export failed: {e}")
    finally: model.export_mode = False # Reset

def perform_quantization(onnx_fp32, onnx_int8, input_name, input_shape, calib_size=64):
    if not ORT_QUANT_AVAILABLE: return False
    print("[*] Starting Quantization...")
    calib_reader = RandomCalibrationDataReader(input_name, input_shape, num_samples=calib_size)
    
    # Whitelist to prevent crashes on complex Upsampling/MaxUnpool replacements
    safe_ops = ['Conv', 'Gemm', 'MatMul', 'Add', 'Mul', 'Relu', 'LeakyRelu', 'GlobalAveragePool']

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

def print_analysis_table(res, flops, params, act_fp32, peak_fp32, lat_fp32, lat_int8, size_fp32, size_int8):
    header = (" Input Size | FLOPs (G) | Params (M) | Total Act FP32 (MB) | Peak Act FP32 (MB) | Latency FP32 (ms) | "
              "Latency INT8 (ms) | Size FP32/INT8 (MB)")
    print("-" * 130)
    print(header)
    print("-" * 130)
    size_str = f"{size_fp32:.1f} / {size_int8:.1f}"
    print(f" {res:10s} | {flops:9.3f} | {params:10.3f} | {act_fp32:19.2f} | {peak_fp32:18.2f} | {lat_fp32:17.2f} | "
          f"{lat_int8:17.2f} | {size_str:>18s}")

def process_resolution(res, n_classes):
    c, w, h = res
    input_shape = (1, c, h, w)
    res_str = f"{w}x{h}"
    print(f"\nProcessing {res_str}...")

    model = ENet(num_classes=n_classes)
    model.eval()

    flop_prof = FlopProfiler(); act_prof = ActivationProfiler()
    flop_prof.register_hooks(model); act_prof.register_hooks(model)
    with torch.no_grad(): _ = model(torch.randn(*input_shape))
    flops_g = flop_prof.total_flops() / 1e9
    params_m = sum(p.numel() for p in model.parameters()) / 1e6
    act_stats = act_prof.get_stats()
    act_fp32 = act_stats['total_memory_mb']; peak_fp32 = act_stats['peak_activation_mb']
    flop_prof.remove_hooks(); act_prof.remove_hooks()

    onnx_path = f"enet_{w}x{h}.onnx"; quant_path = f"enet_{w}x{h}_int8.onnx"
    export_to_onnx(model, onnx_path, input_shape)
    
    success_quant = False
    if os.path.exists(onnx_path):
        success_quant = perform_quantization(onnx_path, quant_path, 'input', input_shape)

    lat_fp32 = run_onnxruntime(onnx_path, input_shape, runs=10)
    lat_int8 = run_onnxruntime(quant_path, input_shape, runs=10) if success_quant else 0.0
    size_fp32 = get_model_size_mb(onnx_path)
    size_int8 = get_model_size_mb(quant_path)

    print_analysis_table(res_str, flops_g, params_m, act_fp32, peak_fp32, lat_fp32, lat_int8, size_fp32, size_int8)
    return {'res': res_str, 'lat_fp32': lat_fp32, 'lat_int8': lat_int8}

def main():
    print("="*100 + "\nENET FIXED INT8 PIPELINE\n" + "="*100)
    resolutions = [
        (3, 640, 360), (3, 1280, 720), (3, 1360, 760),
        (3, 1600, 900), (3, 1920, 1080), (3, 2048, 1152),
        (3, 2560, 1440), (3, 3840, 2160)
    ]
    results = []
    for res in resolutions:
        try: results.append(process_resolution(res, n_classes=21))
        except Exception as e: 
            print(f"Failed {res}: {e}")
            traceback.print_exc()

    print("\n" + "="*60 + "\nSPEEDUP SUMMARY\n" + "="*60)
    for r in results:
        if r['lat_int8'] > 0:
            print(f"{r['res']:10s}: {r['lat_fp32']/r['lat_int8']:.2f}x Speedup")

if __name__ == "__main__":
    main()
