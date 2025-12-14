#!/usr/bin/env python3
"""
SegFormer-B5 FIXED INT8 PIPELINE
Fixes:
1. Uses Whitelist Quantization to keep LayerNorm/GELU/Resize in FP32 for stability.
2. Integrates standard profiling and ONNX export.
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
#  SegFormer-B5 Model Definition
# ===================================================================

class OverlapPatchMerging(nn.Module):
    def __init__(self, in_ch, out_ch, patch_size, stride):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, patch_size, stride, patch_size//2, bias=False)
        self.norm = nn.LayerNorm(out_ch)
    def forward(self, x):
        B,C,H,W = x.shape
        x = self.conv(x)
        _,C2,H2,W2 = x.shape
        x = x.flatten(2).transpose(1,2)
        x = self.norm(x)
        return x, H2, W2

class EfficientSelfAttention(nn.Module):
    def __init__(self, C, heads, reduction):
        super().__init__()
        self.heads = heads
        self.scale = (C//heads)**-0.5
        self.q = nn.Linear(C, C)
        self.kv = nn.Linear(C, 2*C)
        self.proj = nn.Linear(C, C)
        if reduction > 1:
            self.sr = nn.Conv2d(C, C, reduction, reduction)
            self.norm = nn.LayerNorm(C)
        else:
            self.sr = None
    def forward(self, x, H, W):
        B,N,C = x.shape
        q = self.q(x).view(B,N,self.heads,C//self.heads).permute(0,2,1,3)
        if self.sr:
            x2 = x.permute(0,2,1).view(B,C,H,W)
            x2 = self.sr(x2).view(B,C,-1).permute(0,2,1)
            x2 = self.norm(x2)
        else:
            x2 = x
        kv = self.kv(x2).view(B,-1,2,self.heads,C//self.heads).permute(2,0,3,1,4)
        k,v = kv[0], kv[1]
        attn = (q @ k.transpose(-2,-1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1,2).reshape(B,N,C)
        return self.proj(x)

class MixFFN(nn.Module):
    def __init__(self, C, hidden):
        super().__init__()
        self.fc1 = nn.Linear(C, hidden)
        self.dw  = nn.Conv2d(hidden, hidden, 3, 1, 1, groups=hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, C)
    def forward(self, x, H, W):
        B,N,C = x.shape
        x = self.fc1(x)
        x = x.transpose(1,2).view(B,-1,H,W)
        x = self.dw(x)
        x = x.flatten(2).transpose(1,2)
        x = self.act(x)
        return self.fc2(x)

class SegFormerBlock(nn.Module):
    def __init__(self, C, heads, mlp_ratio, reduction):
        super().__init__()
        self.norm1 = nn.LayerNorm(C)
        self.attn  = EfficientSelfAttention(C, heads, reduction)
        self.norm2 = nn.LayerNorm(C)
        self.ffn   = MixFFN(C, int(C*mlp_ratio))
    def forward(self, x, H, W):
        x = x + self.attn(self.norm1(x), H, W)
        x = x + self.ffn(self.norm2(x), H, W)
        return x

class SegFormerB5(nn.Module):
    def __init__(self, num_classes=19):
        super().__init__()
        ws = [64, 128, 320, 512]
        ds = [3, 6, 40, 3]
        heads = [1, 2, 5, 8]
        rr = [8, 4, 2, 1]
        ps = [7, 3, 3, 3]
        strides = [4, 2, 2, 2]
        mlp_ratio = 4
        dec_ch = 768

        self.stages = nn.ModuleList()
        for i in range(4):
            layers = []
            layers.append(OverlapPatchMerging(
                in_ch = 3 if i==0 else ws[i-1],
                out_ch= ws[i],
                patch_size= ps[i],
                stride= strides[i]
            ))
            for _ in range(ds[i]):
                layers.append(SegFormerBlock(ws[i], heads[i], mlp_ratio, rr[i]))
            self.stages.append(nn.Sequential(*layers))

        self.norms = nn.ModuleList([nn.LayerNorm(w) for w in ws])
        self.decoder_mlps = nn.ModuleList([nn.Conv2d(w, dec_ch, 1, bias=False) for w in ws])
        self.head = nn.Conv2d(dec_ch*4, num_classes, 1)

    def forward(self, x):
        B,C,H,W = x.shape
        feats = []
        for i, stage in enumerate(self.stages):
            x, H2, W2 = stage[0](x)
            for blk in stage[1:]:
                x = blk(x, H2, W2)
            x = self.norms[i](x)
            x_sp = x.transpose(1,2).reshape(B, H2, W2, -1).permute(0,3,1,2)
            feats.append(x_sp)
            x = x_sp

        outs = []
        tgt = feats[0].shape[2:]
        for mlp, feat in zip(self.decoder_mlps, feats):
            y = mlp(feat)
            if y.shape[2:] != tgt:
                y = F.interpolate(y, size=tgt, mode='bilinear', align_corners=False)
            outs.append(y)

        x = torch.cat(outs, 1)
        x = self.head(x)
        if x.shape[2:] != (H, W):
            x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False)
        return x

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
            elif isinstance(module, nn.Linear):
                # SegFormer uses Linear layers for QKV projections
                in_f = module.in_features; out_f = module.out_features
                # Input to linear in transformer is often (B, N, C)
                # MACs = B * N * Cin * Cout
                num_elements_input = x.numel()
                macs = num_elements_input * out_f
                self.layer_flops.append(('Linear', 2 * macs))
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
    if hasattr(model, 'training'): model.training = False
    dummy = torch.randn(*input_size)
    try:
        # Opset 13 is generally robust for Transformers
        torch.onnx.export(model, dummy, onnx_path, opset_version=13, input_names=['input'], output_names=['output'],
                          dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}},
                          do_constant_folding=True, export_params=True)
        print(f"[✓] Exported: {onnx_path}")
    except Exception as e: print(f"[✗] Export failed: {e}")

def perform_quantization(onnx_fp32, onnx_int8, input_name, input_shape, calib_size=64):
    if not ORT_QUANT_AVAILABLE: return False
    print("[*] Starting Quantization...")
    calib_reader = RandomCalibrationDataReader(input_name, input_shape, num_samples=calib_size)
    
    # Whitelist Strategy: Explicitly allow only robust ops.
    # Excludes LayerNorm, GELU, Softmax, Resize -> Keeps them in FP32 to prevent crashes/accuracy loss
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

def run_onnxruntime(onnx_path, input_shape, warmup, runs):
    if not ORT_AVAILABLE or not os.path.exists(onnx_path): return 0.0
    so = ort.SessionOptions(); so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    sess = ort.InferenceSession(onnx_path, sess_options=so, providers=['CPUExecutionProvider'])
    feeds = {'input': np.random.randn(1, *input_shape[1:]).astype(np.float32)}
    for _ in range(warmup): sess.run(None, feeds)
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

    model = SegFormerB5(num_classes=n_classes)
    model.eval()

    flop_prof = FlopProfiler(); act_prof = ActivationProfiler()
    flop_prof.register_hooks(model); act_prof.register_hooks(model)
    with torch.no_grad(): _ = model(torch.randn(*input_shape))
    flops_g = flop_prof.total_flops() / 1e9
    params_m = sum(p.numel() for p in model.parameters()) / 1e6
    act_stats = act_prof.get_stats()
    act_fp32 = act_stats['total_memory_mb']; peak_fp32 = act_stats['peak_activation_mb']
    flop_prof.remove_hooks(); act_prof.remove_hooks()

    onnx_path = f"segformer_{w}x{h}.onnx"; quant_path = f"segformer_{w}x{h}_int8.onnx"
    export_to_onnx(model, onnx_path, input_shape)
    
    success_quant = False
    if os.path.exists(onnx_path):
        success_quant = perform_quantization(onnx_path, quant_path, 'input', input_shape, 64)

    int8_act = 0.0; int8_peak = 0.0
    if success_quant:
        est = estimate_onnx_activation_memory(quant_path, 8)
        if est: int8_act = est['total_memory_mb']; int8_peak = est['peak_tensor_mb']
        else: int8_act = act_fp32/4.0; int8_peak = peak_fp32/4.0

    lat_fp32 = run_onnxruntime(onnx_path, input_shape, 2, 5)
    lat_int8 = run_onnxruntime(quant_path, input_shape, 2, 5) if success_quant else 0.0
    size_fp32 = get_model_size_mb(onnx_path)
    size_int8 = get_model_size_mb(quant_path)

    print_analysis_table(res_str, flops_g, params_m, act_fp32, peak_fp32, lat_fp32, int8_act, int8_peak, lat_int8, size_fp32, size_int8)
    return {'res': res_str, 'lat_fp32': lat_fp32, 'lat_int8': lat_int8}

def main():
    print("="*100 + "\nSEGFORMER-B5 FIXED INT8 PIPELINE\n" + "="*100)
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
