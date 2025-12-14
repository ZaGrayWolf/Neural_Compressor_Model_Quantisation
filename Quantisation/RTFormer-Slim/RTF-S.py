#!/usr/bin/env python3
"""
RTFormer-Slim INT8 Quantisation & Benchmark Pipeline (Fixed)
Fixes:
1. Uses RobustAdaptiveAvgPool2d to fix ONNX export errors at 720p/odd resolutions.
2. Uses Whitelist Quantization to prevent CPU crashes on Upsample/LayerNorm.
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
#  Robust Utils (Fixing Export Issues)
# ===================================================================

class RobustAdaptiveAvgPool2d(nn.Module):
    """
    Drop-in replacement for nn.AdaptiveAvgPool2d.
    Calculates dynamic kernel/stride to ensure ONNX export works 
    even when input_size % output_size != 0.
    """
    def __init__(self, output_size):
        super().__init__()
        self.output_size = output_size

    def forward(self, x):
        in_h, in_w = x.shape[-2], x.shape[-1]
        out_h = out_w = self.output_size if isinstance(self.output_size, int) else self.output_size
        
        if in_h == out_h and in_w == out_w: return x
        if out_h == 1 and out_w == 1: return x.mean(dim=(-2, -1), keepdim=True)

        stride_h = in_h // out_h
        stride_w = in_w // out_w
        kernel_h = in_h - (out_h - 1) * stride_h
        kernel_w = in_w - (out_w - 1) * stride_w
        
        return F.avg_pool2d(x, kernel_size=(kernel_h, kernel_w), stride=(stride_h, stride_w))

# ===================================================================
#  RTFormer-Slim Model Definition
# ===================================================================

class GFA(nn.Module):
    def __init__(self, dim, reduction=1):
        super().__init__()
        self.reduction = reduction
        if reduction > 1:
            self.sr = nn.Conv2d(dim, dim, reduction, reduction)
            self.norm = nn.LayerNorm(dim)
        else:
            self.sr = None
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)
        self.scale = (dim) ** -0.5

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, C).unsqueeze(1)
        if self.sr:
            x2 = x.transpose(1, 2).view(B, C, H, W)
            x2 = self.sr(x2).view(B, C, -1).transpose(1, 2)
            x2 = self.norm(x2)
        else:
            x2 = x
        kv = self.kv(x2).reshape(B, -1, 2, C).permute(2, 0, 1, 3)
        k, v = kv[0], kv[1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).squeeze(1)
        return self.proj(out)

class RTFormerBlock(nn.Module):
    def __init__(self, dim, reduction):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = GFA(dim, reduction)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
    def forward(self, x, H, W):
        x = x + self.attn(self.norm1(x), H, W)
        x = x + self.ffn(self.norm2(x))
        return x

class OverlapPatchMerging(nn.Module):
    def __init__(self, in_ch, out_ch, patch_size, stride):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, patch_size, stride, patch_size // 2, bias=False)
        self.norm = nn.LayerNorm(out_ch)
    def forward(self, x):
        x = self.conv(x)
        B, C2, H2, W2 = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H2, W2

class RTFormerBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        widths = [32, 64, 128, 256]
        depths = [2, 2, 2, 2]
        reductions = [8, 4, 2, 1]
        patch_sz = [3, 3, 3, 3]
        strides = [1, 2, 2, 2]
        self.stages = nn.ModuleList()
        self.norms = nn.ModuleList()
        in_ch = 3
        for w, d, r, ps, st in zip(widths, depths, reductions, patch_sz, strides):
            layers = [OverlapPatchMerging(in_ch, w, ps, st)]
            for _ in range(d):
                layers.append(RTFormerBlock(w, r))
            self.stages.append(nn.Sequential(*layers))
            self.norms.append(nn.LayerNorm(w))
            in_ch = w
    def forward(self, x):
        feats = []
        for stage, norm in zip(self.stages, self.norms):
            x, H, W = stage[0](x)
            for blk in stage[1:]:
                x = blk(x, H, W)
            x = norm(x)
            B, N, C = x.shape
            x_sp = x.transpose(1, 2).view(B, C, H, W)
            feats.append(x_sp)
            x = x_sp
        return feats

class DAPPM(nn.Module):
    def __init__(self, in_ch, mid_ch):
        super().__init__()
        # *** FIX: Use RobustAdaptiveAvgPool2d ***
        self.scale1 = nn.Sequential(RobustAdaptiveAvgPool2d(1), nn.Conv2d(in_ch, mid_ch, 1, bias=False), nn.BatchNorm2d(mid_ch), nn.ReLU())
        self.scale2 = nn.Sequential(RobustAdaptiveAvgPool2d(2), nn.Conv2d(in_ch, mid_ch, 1, bias=False), nn.BatchNorm2d(mid_ch), nn.ReLU())
        self.scale3 = nn.Sequential(RobustAdaptiveAvgPool2d(4), nn.Conv2d(in_ch, mid_ch, 1, bias=False), nn.BatchNorm2d(mid_ch), nn.ReLU())
        self.scale4 = nn.Sequential(RobustAdaptiveAvgPool2d(8), nn.Conv2d(in_ch, mid_ch, 1, bias=False), nn.BatchNorm2d(mid_ch), nn.ReLU())
        self.process = nn.Sequential(nn.Conv2d(in_ch, mid_ch, 1, bias=False), nn.BatchNorm2d(mid_ch), nn.ReLU())
        self.compress = nn.Sequential(nn.Conv2d(mid_ch * 5, mid_ch, 1, bias=False), nn.BatchNorm2d(mid_ch), nn.ReLU())
    def forward(self, x):
        size = x.shape[2:]
        x0 = self.process(x)
        x1 = F.interpolate(self.scale1(x), size, mode='bilinear', align_corners=False)
        x2 = F.interpolate(self.scale2(x), size, mode='bilinear', align_corners=False)
        x3 = F.interpolate(self.scale3(x), size, mode='bilinear', align_corners=False)
        x4 = F.interpolate(self.scale4(x), size, mode='bilinear', align_corners=False)
        return self.compress(torch.cat([x0, x1, x2, x3, x4], 1))

class RTFormerHead(nn.Module):
    def __init__(self, in_chs=[32, 64, 128, 256], mid_ch=128, num_classes=19):
        super().__init__()
        self.mlps = nn.ModuleList([
            nn.Sequential(nn.Conv2d(ic, mid_ch, 1, bias=False), nn.BatchNorm2d(mid_ch), nn.ReLU())
            for ic in in_chs
        ])
        self.dappm = DAPPM(mid_ch * len(in_chs), mid_ch)
        self.final = nn.Sequential(
            nn.Conv2d(mid_ch, mid_ch, 3, 1, 1, bias=False),
            nn.BatchNorm2d(mid_ch), nn.ReLU(),
            nn.Dropout2d(0.1),
            nn.Conv2d(mid_ch, num_classes, 1)
        )
    def forward(self, feats):
        size = feats[0].shape[2:]
        outs = []
        for mlp, f in zip(self.mlps, feats):
            y = mlp(f)
            if y.shape[2:] != size:
                y = F.interpolate(y, size, mode='bilinear', align_corners=False)
            outs.append(y)
        x = torch.cat(outs, 1)
        x = self.dappm(x)
        return self.final(x)

class RTFormerSlim(nn.Module):
    def __init__(self, num_classes=19):
        super().__init__()
        self.backbone = RTFormerBackbone()
        self.head = RTFormerHead(num_classes=num_classes)
    def forward(self, x):
        feats = self.backbone(x)
        return self.head(feats)

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
            Cout, H_out, W_out = out_t.shape[1:]
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                Cin = module.in_channels
                kh, kw = module.kernel_size if isinstance(module.kernel_size, tuple) else (module.kernel_size, module.kernel_size)
                groups = module.groups if hasattr(module, 'groups') else 1
                macs = batch * Cout * H_out * W_out * (Cin // groups) * kh * kw
                self.layer_flops.append((module.__class__.__name__, 2 * macs))
            elif isinstance(module, nn.Linear):
                # Approximation for linear layers in transformer
                in_f = module.in_features; out_f = module.out_features
                flops = 2 * batch * x.shape[1] * in_f * out_f 
                self.layer_flops.append(('Linear', flops))
        except Exception: pass
    def register_hooks(self, model):
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                self.hooks.append(module.register_forward_hook(lambda m, i, o, mm=module: self._conv_flops(mm, i, o)))
            elif isinstance(module, nn.Linear):
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

def export_to_onnx(model, onnx_path, input_size):
    model.eval()
    dummy = torch.randn(*input_size)
    try:
        torch.onnx.export(model, dummy, onnx_path, opset_version=13, input_names=['input'], output_names=['output'],
                          dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}},
                          do_constant_folding=True, export_params=True)
        print(f"[✓] Exported: {onnx_path}")
    except Exception as e: print(f"[✗] Export failed: {e}")

def perform_quantization(onnx_fp32, onnx_int8, input_name, input_shape, calib_size=64):
    if not ORT_QUANT_AVAILABLE: return False
    print("[*] Starting Quantization...")
    calib_reader = RandomCalibrationDataReader(input_name, input_shape, num_samples=calib_size)
    
    # Whitelist Strategy: Explicitly excludes LayerNorm (often tricky) and Resize
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
    # Warmup
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

def process_resolution(res):
    c, w, h = res
    input_shape = (1, c, h, w)
    res_str = f"{w}x{h}"
    print(f"\nProcessing {res_str}...")

    model = RTFormerSlim(num_classes=19)
    model.eval()

    flop_prof = FlopProfiler(); act_prof = ActivationProfiler()
    flop_prof.register_hooks(model); act_prof.register_hooks(model)
    with torch.no_grad(): _ = model(torch.randn(*input_shape))
    flops_g = flop_prof.total_flops() / 1e9
    act_stats = act_prof.get_stats()
    act_fp32 = act_stats['total_memory_mb']; peak_fp32 = act_stats['peak_activation_mb']
    params_m = sum(p.numel() for p in model.parameters()) / 1e6
    flop_prof.remove_hooks(); act_prof.remove_hooks()

    onnx_path = f"rtformer_{w}x{h}.onnx"; quant_path = f"rtformer_{w}x{h}_int8.onnx"
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
    print("="*100 + "\nRTFORMER-SLIM FIXED INT8 PIPELINE\n" + "="*100)
    resolutions_hw = [
        (360, 640), (720, 1280), (760, 1360), (900, 1600), 
        (1080, 1920), (1152, 2048), (1440, 2560), (2160, 3840)
    ]
    resolutions = [(3, w, h) for h, w in resolutions_hw] 

    results = []
    for res in resolutions:
        try: results.append(process_resolution(res))
        except Exception as e: 
            print(f"Failed {res}: {e}")
            traceback.print_exc()

    print("\n" + "="*60 + "\nSPEEDUP SUMMARY\n" + "="*60)
    for r in results:
        if r['lat_int8'] > 0:
            print(f"{r['res']:10s}: {r['lat_fp32']/r['lat_int8']:.2f}x Speedup")

if __name__ == "__main__":
    main()
