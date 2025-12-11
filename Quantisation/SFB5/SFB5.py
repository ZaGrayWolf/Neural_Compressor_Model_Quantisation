#!/usr/bin/env python3
"""
segformer_b5_profile.py

SegFormer-B5 Profiling & (optional) ONNX export / ORT benchmarking script.

What it does:
 - Builds SegFormer-B5 model (from the snippet you provided).
 - Computes a theoretical FLOPs estimate (your analytic routine).
 - Optionally computes PTFLOPs via ptflops (if installed).
 - Counts parameters.
 - Measures CPU latency (PyTorch) and activation memory (MB) via forward hooks.
 - Optionally exports to ONNX and runs ONNXRuntime CPU benchmark and quantization
   (requires onnx + onnxruntime + onnxruntime.quantization).
 - Robust to missing optional packages and large resolutions (keeps defaults conservative).

Save as `segformer_b5_profile.py` and run:
    python3 segformer_b5_profile.py
"""

import os
import time
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore")

# Optional: ptflops for measured FLOPs
try:
    from ptflops import get_model_complexity_info
    PTFLOPS_AVAILABLE = True
except Exception:
    PTFLOPS_AVAILABLE = False

# Optional: ONNX / ORT imports for export, quantization, and runtime benchmarking
try:
    import onnx
    ONNX_AVAILABLE = True
except Exception:
    onnx = None
    ONNX_AVAILABLE = False

try:
    import onnxruntime as ort
    ORT_AVAILABLE = True
except Exception:
    ort = None
    ORT_AVAILABLE = False

try:
    from onnxruntime.quantization import quantize_static, quantize_dynamic, CalibrationDataReader, QuantType, QuantFormat
    ORT_QUANT_AVAILABLE = True
except Exception:
    quantize_static = None
    quantize_dynamic = None
    CalibrationDataReader = None
    QuantType = None
    QuantFormat = None
    ORT_QUANT_AVAILABLE = False

# ------------------------------------------------------------------------------
# Model Definition (SegFormer-B5)
# ------------------------------------------------------------------------------

class OverlapPatchMerging(nn.Module):
    def __init__(self, in_ch, out_ch, patch_size, stride):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, patch_size, stride, patch_size//2, bias=False)
        # LayerNorm with normalized shape = out_ch
        self.norm = nn.LayerNorm(out_ch)

    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        x = self.conv(x)                       # (B, out_ch, H2, W2)
        B2, C2, H2, W2 = x.shape
        x = x.flatten(2).transpose(1, 2)       # (B, N, C)
        x = self.norm(x)                       # LayerNorm normalizes last dim (C)
        return x, H2, W2

class EfficientSelfAttention(nn.Module):
    def __init__(self, C, heads, reduction):
        super().__init__()
        self.heads = heads
        self.scale = (C // heads) ** -0.5
        self.q = nn.Linear(C, C)
        self.kv = nn.Linear(C, 2 * C)
        self.proj = nn.Linear(C, C)
        if reduction > 1:
            self.sr = nn.Conv2d(C, C, reduction, reduction)
            self.norm = nn.LayerNorm(C)
        else:
            self.sr = None
            self.norm = None

    def forward(self, x, H, W):
        # x: (B, N, C)
        B, N, C = x.shape
        q = self.q(x).view(B, N, self.heads, C // self.heads).permute(0, 2, 1, 3)  # (B, heads, N, C_h)
        if self.sr is not None:
            x2 = x.permute(0, 2, 1).view(B, C, H, W)            # (B, C, H, W)
            x2 = self.sr(x2).flatten(2).permute(0, 2, 1)        # (B, N2, C)
            x2 = self.norm(x2)
        else:
            x2 = x
        kv = self.kv(x2).view(B, -1, 2, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]                                    # each (B, heads, N_k, C_h)
        attn = (q @ k.transpose(-2, -1)) * self.scale          # (B, heads, N_q, N_k)
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)

class MixFFN(nn.Module):
    def __init__(self, C, hidden):
        super().__init__()
        self.fc1 = nn.Linear(C, hidden)
        self.dw = nn.Conv2d(hidden, hidden, 3, 1, 1, groups=hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, C)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = self.fc1(x)                       # (B, N, hidden)
        x = x.transpose(1, 2).view(B, -1, H, W)  # (B, hidden, H, W)
        x = self.dw(x)
        x = x.flatten(2).transpose(1, 2)      # (B, N, hidden)
        x = self.act(x)
        return self.fc2(x)

class SegFormerBlock(nn.Module):
    def __init__(self, C, heads, mlp_ratio, reduction):
        super().__init__()
        self.norm1 = nn.LayerNorm(C)
        self.attn = EfficientSelfAttention(C, heads, reduction)
        self.norm2 = nn.LayerNorm(C)
        self.ffn = MixFFN(C, int(C * mlp_ratio))

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
                in_ch = 3 if i == 0 else ws[i-1],
                out_ch = ws[i],
                patch_size = ps[i],
                stride = strides[i]
            ))
            for _ in range(ds[i]):
                layers.append(SegFormerBlock(ws[i], heads[i], mlp_ratio, rr[i]))
            self.stages.append(nn.Sequential(*layers))

        self.norms = nn.ModuleList([nn.LayerNorm(w) for w in ws])
        self.decoder_mlps = nn.ModuleList([nn.Conv2d(w, dec_ch, 1, bias=False) for w in ws])
        self.head = nn.Conv2d(dec_ch * 4, num_classes, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        feats = []
        for i, stage in enumerate(self.stages):
            x, H2, W2 = stage[0](x)        # OverlapPatchMerging
            for blk in stage[1:]:
                x = blk(x, H2, W2)
            x = self.norms[i](x)          # (B, N, Cw)
            # reshape -> (B, Cw, H2, W2)
            x_sp = x.transpose(1, 2).reshape(B, -1, H2, W2).permute(0, 1, 2, 3)
            # shape is (B, Cw, H2, W2) already after transpose/reshape; permute is harmless
            x_sp = x_sp.permute(0, 1, 2, 3)  # ensure contiguous ordering
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

# ------------------------------------------------------------------------------
# Profiling Utilities
# ------------------------------------------------------------------------------

def _conv_flops(cin, cout, h, w, k, stride=1, pad=0):
    hout = (h + 2 * pad - k) // stride + 1
    wout = (w + 2 * pad - k) // stride + 1
    # 2 * MACs
    return 2 * cin * cout * k * k * hout * wout, hout, wout

def calculate_theoretical_flops(inp, num_classes=19):
    c, h, w = inp
    ws=[64,128,320,512]
    ds=[3,6,40,3]
    heads=[1,2,5,8]
    rr=[8,4,2,1]
    ps=[7,3,3,3]
    strides=[4,2,2,2]
    mlp_ratio=4
    dec_ch=768

    fl = 0
    H, W = h, w
    for i in range(4):
        inch = c if i == 0 else ws[i-1]
        outch = ws[i]
        p, H, W = _conv_flops(inch, outch, H, W, ps[i], strides[i], ps[i]//2); fl += p
        for _ in range(ds[i]):
            seq = H * W
            # attention qkv & proj
            fl += 2 * seq * outch * outch * 2
            # attn matmul (q @ k^T): 2*heads*seq*seq*(outch//heads)
            fl += 2 * heads[i] * seq * seq * (outch // heads[i])
            hid = outch * mlp_ratio
            fl += 2 * seq * outch * hid
            fl += hid * 9 * H * W
            fl += 2 * seq * hid * outch

    fh, fw = h // 4, w // 4
    for w_i in ws:
        fl += 2 * w_i * dec_ch * fh * fw
        fl += 7 * dec_ch * fh * fw
    fl += 2 * (dec_ch * 4) * num_classes * fh * fw
    fl += 7 * num_classes * h * w
    return fl

def count_parameters(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)

def measure_latency(m, inp, runs=3, rep=5):
    dev = torch.device('cpu')
    m = m.to(dev).eval()
    x = torch.randn(1, *inp).to(dev)
    with torch.no_grad():
        for _ in range(2):
            _ = m(x)
    ts = []
    with torch.no_grad():
        for _ in range(runs):
            s = time.perf_counter()
            for _ in range(rep):
                _ = m(x)
            e = time.perf_counter()
            ts.append((e - s) * 1000.0 / rep)
    return float(sum(ts) / len(ts))

# Activation tracker uses hooks on leaf modules to accumulate activation bytes
class ActivationTracker:
    def __init__(self, model):
        self.model = model
        self.act_bytes = 0

    def _hook_add(self, module, inp, out):
        # only count tensor outputs (avoid lists/tuples)
        if isinstance(out, torch.Tensor):
            self.act_bytes += out.numel() * 4
        elif isinstance(out, (list, tuple)):
            for o in out:
                if isinstance(o, torch.Tensor):
                    self.act_bytes += o.numel() * 4

    def compute(self, x):
        # register forward hooks on leaf modules (no children)
        hooks = []
        for m in self.model.modules():
            if len(list(m.children())) == 0:
                try:
                    hooks.append(m.register_forward_hook(self._hook_add))
                except Exception:
                    pass
        # run forward
        self.act_bytes = 0
        with torch.no_grad():
            _ = self.model(x)
        # remove hooks
        for h in hooks:
            try:
                h.remove()
            except Exception:
                pass
        return self.act_bytes

# -------------------------
# ONNX / ORT helpers
# -------------------------
def export_to_onnx(model, onnx_path, input_size=(1,3,640,360), input_name='input', opset=18):
    if not ONNX_AVAILABLE:
        raise RuntimeError("onnx package not installed")
    model = model.eval().cpu()
    dummy = torch.randn(*input_size, dtype=torch.float32)
    with torch.no_grad():
        torch.onnx.export(
            model, dummy, onnx_path, opset_version=opset,
            input_names=[input_name], output_names=['output'],
            dynamic_axes={input_name: {0: 'batch'}, 'output': {0: 'batch'}},
            do_constant_folding=True,
            export_params=True
        )
    print(f"[✓] Exported ONNX -> {onnx_path}")

def run_onnxruntime(onnx_path, input_name, input_shape, warmup=5, runs=20):
    if not ORT_AVAILABLE:
        raise RuntimeError("onnxruntime not installed")
    providers = ['CPUExecutionProvider']
    so = ort.SessionOptions()
    try:
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    except Exception:
        pass
    sess = ort.InferenceSession(onnx_path, sess_options=so, providers=providers)
    x = np.random.randn(1, *input_shape).astype(np.float32)
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

# Calibration reader for static quant (synthetic data)
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
            return {self.input_name: np.random.randn(*self.input_shape).astype(np.float32)}
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
            return {self.input_name: np.random.randn(*self.input_shape).astype(np.float32)}

# -------------------------
# Main runner
# -------------------------
def main():
    # resolutions (kept conservative by default)
    inputs = [
        (3, 640, 360),
        (3, 1280, 720),
        (3, 1360, 760),
        (3, 1600, 900),
        (3, 1920, 1080),
        (3, 2048, 1152),
        (3, 2560, 1440),
        (3, 3840, 2160)
    ]

    model = SegFormerB5(num_classes=19)
    params_m = count_parameters(model) / 1e6
    tracker = ActivationTracker(model)

    # Optional: show PTFLOPs if available
    print("\nSegFormer-B5 profiling (CPU). Optional packages: ptflops={}, onnx={}, onnxruntime={}, ort-quant={}".format(
        PTFLOPS_AVAILABLE, ONNX_AVAILABLE, ORT_AVAILABLE, ORT_QUANT_AVAILABLE))
    print(f"{'Input':>16} | {'TheoFLOPs(G)':>12} | {'PTFLOPs(G)':>12} | {'Params(M)':>10} | {'Latency(ms)':>12} | {'Act(MB)':>8}")
    print("-" * 84)

    for inp in inputs:
        theo = calculate_theoretical_flops(inp, num_classes=19) / 1e9
        if PTFLOPS_AVAILABLE:
            try:
                # ptflops expects (C,H,W)
                clean = SegFormerB5(num_classes=19)
                flops, params = get_model_complexity_info(clean, (inp[0], inp[1], inp[2]),
                                                         as_strings=False,
                                                         print_per_layer_stat=False,
                                                         verbose=False)
                pt = flops / 1e9
            except Exception:
                pt = "err"
        else:
            pt = "N/A"

        # latency (PyTorch CPU)
        lat = measure_latency(model, inp, runs=3, rep=3)

        # activation bytes
        act_bytes = tracker.compute(torch.randn(1, *inp))
        act_mb = act_bytes / (1024**2)

        # try ONNX export + ORT benchmark (small inputs only by default)
        onnx_fp32_size = None
        onnx_int8_size = None
        ort_fp32_lat = None

        # export ONNX for this input shape with opset 18 (if available)
        onnx_path = f"segformer_b5_{inp[1]}x{inp[2]}.onnx"
        quant_path = f"segformer_b5_{inp[1]}x{inp[2]}_int8.onnx"

        if ONNX_AVAILABLE:
            try:
                export_to_onnx(model, onnx_path, input_size=(1, inp[0], inp[1], inp[2]), input_name='input', opset=18)
                onnx_fp32_size = os.path.getsize(onnx_path) / (1024**2)
                # try ORT FP32 benchmark
                if ORT_AVAILABLE:
                    try:
                        ort_res = run_onnxruntime(onnx_path, 'input', (inp[0], inp[1], inp[2]), warmup=3, runs=10)
                        ort_fp32_lat = ort_res['mean_ms']
                    except Exception as e:
                        ort_fp32_lat = None
                # try static quant (synthetic calib) if available
                if ORT_QUANT_AVAILABLE:
                    try:
                        calib = RandomCalibrationDataReader('input', (1, inp[0], inp[1], inp[2]), num_samples=8)
                        try:
                            # attempt static
                            quantize_static(model_input=onnx_path, model_output=quant_path,
                                            calibration_data_reader=calib,
                                            activation_type=QuantType.QInt8,
                                            weight_type=QuantType.QInt8)
                        except Exception:
                            # fallback to dynamic
                            quantize_dynamic(model_input=onnx_path, model_output=quant_path, weight_type=QuantType.QInt8)
                        if os.path.exists(quant_path):
                            onnx_int8_size = os.path.getsize(quant_path) / (1024**2)
                    except Exception:
                        onnx_int8_size = None
            except Exception as e:
                # ONNX export sometimes fails for esoteric ops; continue gracefully
                onnx_fp32_size = None

        # choose printed PTFLOPs value
        pt_str = f"{pt:12.2f}" if isinstance(pt, float) else f"{pt:>12}"

        ort_lat_str = f"{ort_fp32_lat:.2f}" if ort_fp32_lat is not None else "   N/A   "

        print(f"{str(inp):>16} | {theo:12.2f} | {pt_str} | {params_m:10.2f} | {lat:12.2f} | {act_mb:8.2f}   ORT_FP32(ms): {ort_lat_str}")

    print("\nDone.")

if __name__ == "__main__":
    main()
