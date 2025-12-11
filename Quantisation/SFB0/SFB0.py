#!/usr/bin/env python3
"""
segformer_b0_profile_with_onnx.py

SegFormer-B0 Profiling Script (cleaned & corrected) + ONNX / ONNX Runtime benchmarking & optional quantization.

What this script computes for each resolution:
 - Analytical/theoretical FLOPs (simplified formula)
 - Optional ptflops measured FLOPs (if ptflops installed)
 - Parameter counts
 - CPU latency (PyTorch)
 - Activation size (MB) measured during a forward pass
 - Export FP32 ONNX model (per-resolution)
 - Run ONNXRuntime FP32 inference benchmark (CPUExecutionProvider)
 - (Optional) Produce INT8 ONNX via ONNX Runtime quantization and benchmark

Notes:
 - Runs on CPU by default (PyTorch and ORT CPU).
 - ONNX / onnxruntime / ptflops are optional; script will skip features gracefully if missing.
 - Large resolutions may consume a lot of RAM. Use fewer resolutions for low-memory machines.
"""

import os
import sys
import time
import math
import gc
import warnings
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

# Optional: ptflops for measured FLOPs
try:
    from ptflops import get_model_complexity_info
    PTFLOPS_AVAILABLE = True
except Exception:
    PTFLOPS_AVAILABLE = False

# Optional ONNX / ONNX Runtime
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

# Optional ORT quant
try:
    from onnxruntime.quantization import (
        quantize_dynamic, quantize_static, CalibrationDataReader, QuantType, QuantFormat
    )
    ORT_QUANT_AVAILABLE = True
except Exception:
    ORT_QUANT_AVAILABLE = False

# ---------------------------
# Model implementation (SegFormer-B0)
# (copied/cleaned from your provided implementation)
# ---------------------------

class OverlapPatchMerging(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, patch_size: int, stride: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=patch_size, stride=stride,
                              padding=patch_size // 2, bias=False)
        self.norm = nn.LayerNorm(out_channels)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        B, C, H, W = x.shape
        y = self.conv(x)
        _, out_ch, H2, W2 = y.shape
        y = y.flatten(2).transpose(1, 2)
        y = self.norm(y)
        return y, H2, W2


class EfficientSelfAttention(nn.Module):
    def __init__(self, channels: int, num_heads: int, reduction_ratio: int = 1):
        super().__init__()
        assert channels % num_heads == 0, "channels must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = (self.head_dim) ** -0.5
        self.reduction_ratio = reduction_ratio

        self.q = nn.Linear(channels, channels, bias=True)
        self.kv = nn.Linear(channels, channels * 2, bias=True)
        self.proj = nn.Linear(channels, channels)

        if reduction_ratio > 1:
            self.sr = nn.Conv2d(channels, channels, kernel_size=reduction_ratio, stride=reduction_ratio)
            self.norm = nn.LayerNorm(channels)
        else:
            self.sr = None
            self.norm = None

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, N, C = x.shape
        q = self.q(x).view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        if self.sr is not None:
            x_sp = x.transpose(1, 2).reshape(B, C, H, W)
            x_reduced = self.sr(x_sp)
            _, _, H_r, W_r = x_reduced.shape
            x_reduced = x_reduced.flatten(2).transpose(1, 2)
            x_reduced = self.norm(x_reduced)
            kv_input = x_reduced
        else:
            kv_input = x

        kv = self.kv(kv_input).view(B, -1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        return out


class MixFFN(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int):
        super().__init__()
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.dwconv = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, stride=1, padding=1, groups=hidden_channels, bias=True)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_channels, in_channels)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, N, C = x.shape
        x = self.fc1(x)
        hidden = x.shape[-1]
        x = x.transpose(1, 2).reshape(B, hidden, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.act(x)
        x = self.fc2(x)
        return x


class SegFormerEncoderBlock(nn.Module):
    def __init__(self, channels: int, num_heads: int, mlp_ratio: int = 4, reduction_ratio: int = 1):
        super().__init__()
        self.norm1 = nn.LayerNorm(channels)
        self.attn = EfficientSelfAttention(channels, num_heads, reduction_ratio)
        self.norm2 = nn.LayerNorm(channels)
        self.ffn = MixFFN(channels, channels * mlp_ratio)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), H, W)
        x = x + self.ffn(self.norm2(x), H, W)
        return x


class MiTEncoder(nn.Module):
    def __init__(self,
                 in_channels: int,
                 widths: List[int],
                 depths: List[int],
                 num_heads: List[int],
                 reduction_ratios: List[int],
                 patch_sizes: List[int],
                 strides: List[int],
                 mlp_ratio: int):
        super().__init__()
        assert len(widths) == len(depths) == len(num_heads) == len(reduction_ratios) == len(patch_sizes) == len(strides)
        self.num_stages = len(widths)
        self.patch_merging_layers = nn.ModuleList()
        self.transformer_layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(self.num_stages):
            in_ch = in_channels if i == 0 else widths[i - 1]
            pm = OverlapPatchMerging(in_ch, widths[i], patch_size=patch_sizes[i], stride=strides[i])
            self.patch_merging_layers.append(pm)

            blocks = nn.ModuleList([SegFormerEncoderBlock(widths[i], num_heads[i], mlp_ratio, reduction_ratios[i])
                                    for _ in range(depths[i])])
            self.transformer_layers.append(blocks)
            self.norms.append(nn.LayerNorm(widths[i]))

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        features: List[torch.Tensor] = []
        current = x
        for i in range(self.num_stages):
            current, H, W = self.patch_merging_layers[i](current)
            for blk in self.transformer_layers[i]:
                current = blk(current, H, W)
            current = self.norms[i](current)
            B, N, C = current.shape
            spatial = current.transpose(1, 2).reshape(B, C, H, W)
            features.append(spatial)
            current = spatial
        return features


class SegFormerDecoder(nn.Module):
    def __init__(self, in_channels_list: List[int], out_channels: int):
        super().__init__()
        self.mlps = nn.ModuleList([nn.Conv2d(in_ch, out_channels, kernel_size=1, bias=False)
                                   for in_ch in in_channels_list])

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        target = features[0].shape[2:]
        outs = []
        for feat, mlp in zip(features, self.mlps):
            y = mlp(feat)
            if y.shape[2:] != target:
                y = F.interpolate(y, size=target, mode="bilinear", align_corners=False)
            outs.append(y)
        return torch.cat(outs, dim=1)


class SegFormer(nn.Module):
    def __init__(self,
                 in_channels: int,
                 widths: List[int],
                 depths: List[int],
                 num_heads: List[int],
                 reduction_ratios: List[int],
                 patch_sizes: List[int],
                 strides: List[int],
                 mlp_ratio: int,
                 decoder_channels: int,
                 num_classes: int):
        super().__init__()
        self.encoder = MiTEncoder(in_channels, widths, depths, num_heads, reduction_ratios, patch_sizes, strides, mlp_ratio)
        self.decoder = SegFormerDecoder(widths, decoder_channels)
        self.head = nn.Conv2d(decoder_channels * len(widths), num_classes, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_spatial = x.shape[-2:]
        feats = self.encoder(x)
        x = self.decoder(feats)
        x = self.head(x)
        if x.shape[2:] != input_spatial:
            x = F.interpolate(x, size=input_spatial, mode="bilinear", align_corners=False)
        return x


def segformer_b0(num_classes: int = 19) -> SegFormer:
    return SegFormer(
        in_channels=3,
        widths=[32, 64, 160, 256],
        depths=[2, 2, 2, 2],
        num_heads=[1, 2, 5, 8],
        reduction_ratios=[8, 4, 2, 1],
        patch_sizes=[7, 3, 3, 3],
        strides=[4, 2, 2, 2],
        mlp_ratio=4,
        decoder_channels=256,
        num_classes=num_classes
    )


# Activation-tracking variant
class ActivationTrackerMixin:
    def reset_activation_bytes(self):
        self.activation_bytes = 0

    def add_activation(self, tensor: torch.Tensor):
        if tensor is None:
            return
        if isinstance(tensor, torch.Tensor):
            self.activation_bytes += tensor.numel() * 4
        elif isinstance(tensor, (list, tuple)):
            for t in tensor:
                if isinstance(t, torch.Tensor):
                    self.activation_bytes += t.numel() * 4


class SegFormerWithActivation(ActivationTrackerMixin, SegFormer):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.reset_activation_bytes()
        self.add_activation(x)

        input_spatial = x.shape[-2:]
        current = x
        features = []
        encoder: MiTEncoder = self.encoder
        for i in range(encoder.num_stages):
            current, H, W = encoder.patch_merging_layers[i](current)
            self.add_activation(current)
            for blk in encoder.transformer_layers[i]:
                current = blk(current, H, W)
                self.add_activation(current)
            current = encoder.norms[i](current)
            self.add_activation(current)
            B, N, C = current.shape
            spatial = current.transpose(1, 2).reshape(B, C, H, W)
            self.add_activation(spatial)
            features.append(spatial)
            current = spatial

        dec_out = self.decoder(features)
        self.add_activation(dec_out)
        head_out = self.head(dec_out)
        self.add_activation(head_out)
        if head_out.shape[2:] != input_spatial:
            head_out = F.interpolate(head_out, size=input_spatial, mode="bilinear", align_corners=False)
            self.add_activation(head_out)
        return head_out


def segformer_b0_with_activation(num_classes: int = 19) -> SegFormerWithActivation:
    return SegFormerWithActivation(
        in_channels=3,
        widths=[32, 64, 160, 256],
        depths=[2, 2, 2, 2],
        num_heads=[1, 2, 5, 8],
        reduction_ratios=[8, 4, 2, 1],
        patch_sizes=[7, 3, 3, 3],
        strides=[4, 2, 2, 2],
        mlp_ratio=4,
        decoder_channels=256,
        num_classes=num_classes
    )


# ---------------------------
# Utilities: cleanup, theoretical flops, parameter count, latency
# ---------------------------
def force_cleanup_everything():
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        except Exception:
            pass


def calculate_theoretical_flops(input_size: Tuple[int, int, int], num_classes: int = 19) -> float:
    c, h, w = input_size
    widths = [32, 64, 160, 256]
    depths = [2, 2, 2, 2]
    num_heads = [1, 2, 5, 8]
    reduction_ratios = [8, 4, 2, 1]
    patch_sizes = [7, 3, 3, 3]
    strides = [4, 2, 2, 2]
    decoder_channels = 256

    total_flops = 0
    cur_h, cur_w = h, w

    for stage in range(4):
        in_ch = c if stage == 0 else widths[stage - 1]
        out_ch = widths[stage]
        patch = patch_sizes[stage]
        stride = strides[stage]
        cur_h = math.ceil(cur_h / stride)
        cur_w = math.ceil(cur_w / stride)
        total_flops += 2 * in_ch * out_ch * (patch ** 2) * cur_h * cur_w
        for _ in range(depths[stage]):
            seq = cur_h * cur_w
            seq_kv = seq // (reduction_ratios[stage] ** 2) if reduction_ratios[stage] > 1 else seq
            total_flops += 4 * seq * out_ch * out_ch
            total_flops += 2 * num_heads[stage] * seq * seq_kv * (out_ch // num_heads[stage])
            hid = out_ch * 4
            total_flops += 2 * seq * out_ch * hid
            total_flops += hid * 9 * cur_h * cur_w
            total_flops += 2 * seq * hid * out_ch

    fh, fw = h // 4, w // 4
    for w_i in widths:
        total_flops += 2 * w_i * decoder_channels * fh * fw
    total_flops += 2 * (decoder_channels * len(widths)) * num_classes * fh * fw
    total_flops += 7 * num_classes * h * w
    return total_flops


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def measure_latency(model: nn.Module, input_size: Tuple[int, int, int], runs: int = 3, repeats: int = 3):
    device = torch.device("cpu")
    model.to(device).eval()
    x = torch.randn(1, *input_size).to(device)
    with torch.no_grad():
        for _ in range(2):
            _ = model(x)
    times = []
    with torch.no_grad():
        for _ in range(runs):
            s = time.perf_counter()
            for _ in range(repeats):
                _ = model(x)
            e = time.perf_counter()
            times.append((e - s) * 1000.0 / repeats)
    return float(sum(times) / len(times))


# ---------------------------
# ONNX export / ORT utilities
# ---------------------------

def get_model_size_mb(path: str) -> float:
    try:
        return os.path.getsize(path) / (1024 ** 2)
    except Exception:
        return 0.0


def export_to_onnx(model: nn.Module, onnx_path: str, input_size: Tuple[int, int, int], input_name='input', opset=13):
    model_cpu = model.to('cpu').eval()
    C, H, W = input_size
    dummy = torch.randn(1, C, H, W, device='cpu')
    torch.onnx.export(
        model_cpu, dummy, onnx_path, opset_version=opset,
        input_names=[input_name], output_names=['output'],
        dynamic_axes={input_name: {0: 'batch'}, 'output': {0: 'batch'}},
        do_constant_folding=True, export_params=True
    )


def run_onnxruntime(onnx_path: str, input_size: Tuple[int, int, int], warmup=3, runs=20):
    if not ORT_AVAILABLE:
        raise RuntimeError("onnxruntime not installed")
    providers = ['CPUExecutionProvider']
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    sess = ort.InferenceSession(onnx_path, sess_options=so, providers=providers)
    import numpy as _np
    C, H, W = input_size
    x = _np.random.randn(1, C, H, W).astype(_np.float32)
    feeds = {'input': x}
    for _ in range(warmup):
        _ = sess.run(None, feeds)
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        _ = sess.run(None, feeds)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)
    times = sorted(times)
    mean = float(sum(times) / len(times))
    return {
        'mean_ms': mean,
        'min_ms': float(times[0]),
        'p50_ms': float(times[len(times)//2]),
        'p95_ms': float(times[int(len(times)*0.95)-1]) if len(times) > 1 else float(times[-1]),
        'max_ms': float(times[-1]),
        'std_ms': float(math.sqrt(sum((t-mean)**2 for t in times)/len(times))) if len(times) > 1 else 0.0,
        'runs': len(times),
        'providers': providers
    }


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
        import numpy as _np
        data = _np.random.randn(1, *self.input_shape).astype(_np.float32)
        return {self.input_name: data}


def quantize_dynamic_onnx(fp32_path: str, int8_out: str):
    if not ORT_QUANT_AVAILABLE:
        raise RuntimeError("ORT quantization utilities not available")
    quantize_dynamic(model_input=fp32_path, model_output=int8_out, weight_type=QuantType.QInt8)


def quantize_static_onnx(fp32_path: str, int8_out: str, input_shape, num_calib=32, per_channel=False):
    if not ORT_QUANT_AVAILABLE:
        raise RuntimeError("ORT quantization utilities not available")
    reader = RandomCalibrationDataReader('input', input_shape, num_samples=num_calib)
    qformat = QuantFormat.QDQ if hasattr(QuantFormat, 'QDQ') else None
    quantize_static(
        model_input=fp32_path,
        model_output=int8_out,
        calibration_data_reader=reader,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        per_channel=per_channel,
        quant_format=qformat
    )


# ---------------------------
# Main profiling + ONNX flow
# ---------------------------

def human_size_mb(path: str) -> str:
    try:
        s = os.path.getsize(path) / (1024**2)
        return f"{s:.2f} MB"
    except Exception:
        return "N/A"


def profile_all(input_sizes, out_dir="onnx_models", do_quant=True):
    os.makedirs(out_dir, exist_ok=True)

    header = (" Input       | TheoFLOPs(G) | PTFLOPs(G) | Params(M) | PyTorch Lat(ms) | "
              "ONNX FP32(ms) | ONNX INT8(ms) | FP32 size | INT8 size | Act(MB)")
    print(header)
    print("-" * len(header))

    for inp in input_sizes:
        C, H, W = inp
        inp_str = f"{C}x{H}x{W}"
        # instantiate tracked model
        try:
            model = segformer_b0_with_activation(num_classes=19)
            model.eval()
        except Exception as e:
            print(f"{inp_str:12} | ERROR creating model: {e}")
            continue

        # params
        params_m = count_parameters(model) / 1e6

        # theoretical flops
        theo_g = calculate_theoretical_flops(inp, 19) / 1e9

        # ptflops (optional)
        pt_str = "N/A"
        if PTFLOPS_AVAILABLE:
            try:
                fresh = segformer_b0(num_classes=19)
                flops, _ = get_model_complexity_info(fresh, (C, H, W), as_strings=False, print_per_layer_stat=False, verbose=False)
                pt_str = f"{flops / 1e9:.2f}"
                del fresh
                force_cleanup_everything()
            except Exception:
                pt_str = "err"

        # activation measurement & pytorch latency
        try:
            with torch.no_grad():
                _ = model(torch.randn(1, C, H, W))
            act_mb = model.activation_bytes / (1024 ** 2)
            pytorch_lat = measure_latency(model, (C, H, W), runs=2, repeats=2)
        except RuntimeError as e:
            act_mb = "OOM"
            pytorch_lat = "OOM"
        except Exception:
            act_mb = "ERR"
            pytorch_lat = "ERR"

        # ONNX export + ORT benchmarking
        fp32_path = os.path.join(out_dir, f"segformer_b0_{W}x{H}_fp32.onnx")
        int8_path = os.path.join(out_dir, f"segformer_b0_{W}x{H}_int8.onnx")
        onnx_fp32_ms = "N/A"
        onnx_int8_ms = "N/A"
        fp32_size = "N/A"
        int8_size = "N/A"

        if ONNX_AVAILABLE:
            try:
                export_to_onnx(model, fp32_path, (C, H, W))
                fp32_size = human_size_mb(fp32_path)
            except Exception as e:
                fp32_path = None
                fp32_size = "ERR"
        else:
            fp32_path = None

        if fp32_path and ORT_AVAILABLE:
            try:
                ort_stats = run_onnxruntime(fp32_path, (C, H, W), warmup=3, runs=20)
                onnx_fp32_ms = f"{ort_stats['mean_ms']:.2f}"
            except Exception:
                onnx_fp32_ms = "ERR"

        # quantization: dynamic first, static fallback
        if do_quant and fp32_path and ORT_QUANT_AVAILABLE:
            try:
                try:
                    quantize_dynamic_onnx(fp32_path, int8_path)
                    int8_size = human_size_mb(int8_path)
                    if ORT_AVAILABLE:
                        int8_stats = run_onnxruntime(int8_path, (C, H, W), warmup=3, runs=20)
                        onnx_int8_ms = f"{int8_stats['mean_ms']:.2f}"
                except Exception:
                    # fallback static
                    quantize_static_onnx(fp32_path, int8_path, (C, H, W), num_calib=16)
                    int8_size = human_size_mb(int8_path)
                    if ORT_AVAILABLE:
                        int8_stats = run_onnxruntime(int8_path, (C, H, W), warmup=3, runs=20)
                        onnx_int8_ms = f"{int8_stats['mean_ms']:.2f}"
            except Exception:
                onnx_int8_ms = "ERR"
                int8_size = "ERR"
        else:
            if do_quant and not ORT_QUANT_AVAILABLE:
                onnx_int8_ms = "quant-unavail"
                int8_size = "N/A"

        # print results line
        theo_str = f"{theo_g:12.2f}"
        ptcol = pt_str if isinstance(pt_str, str) else f"{float(pt_str):12.2f}"
        params_str = f"{params_m:8.2f}"
        py_lat_str = f"{pytorch_lat:14.2f}" if isinstance(pytorch_lat, float) else f"{pytorch_lat:>14}"
        onnx_fp32_str = f"{onnx_fp32_ms:>12}"
        onnx_int8_str = f"{onnx_int8_ms:>12}"
        fp32_size_str = f"{fp32_size:>9}"
        int8_size_str = f"{int8_size:>9}"
        act_str = f"{act_mb:7.2f}" if isinstance(act_mb, (int, float)) else f"{act_mb:>7}"

        print(f" {inp_str:10} | {theo_str} | {ptcol:>10} | {params_str} | {py_lat_str} | {onnx_fp32_str} | {onnx_int8_str} | {fp32_size_str} | {int8_size_str} | {act_str}")

        # cleanup
        del model
        gc.collect()


# ---------------------------
# Entry point
# ---------------------------
if __name__ == "__main__":
    INPUTS = [
        (3, 640, 360),
        (3, 1280, 720),
        (3, 1360, 760),
        (3, 1600, 900),
        (3, 1920, 1080),
        (3, 2048, 1152),
        (3, 2560, 1440),
        (3, 3840, 2160),
    ]

    OUT_DIR = "onnx_segformer_b0"
    DO_QUANT = True  # set False to skip ONNX quantization

    print("SegFormer-B0 profiling + ONNX/ORT benchmarking")
    print(f"ONNX: {ONNX_AVAILABLE} | ORT: {ORT_AVAILABLE} | ORT-quant: {ORT_QUANT_AVAILABLE} | ptflops: {PTFLOPS_AVAILABLE}")
    print()
    profile_all(INPUTS, out_dir=OUT_DIR, do_quant=DO_QUANT)
