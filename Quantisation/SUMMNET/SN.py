#!/usr/bin/env python3
"""
SUMNet (SUMNet_all_bn) Full Profiling + ONNX/ORT Benchmarking Script
Includes INT8 activation memory estimation by dividing FP32 activations by 4.
"""

import os
import sys
import time
import math
import gc
import warnings
from typing import Tuple, List, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

try:
    from ptflops import get_model_complexity_info
    PTFLOPS_AVAILABLE = True
except Exception:
    PTFLOPS_AVAILABLE = False

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
    from onnxruntime.quantization import quantize_dynamic, QuantType
    ORT_QUANT_AVAILABLE = True
except Exception:
    ORT_QUANT_AVAILABLE = False

def calculate_conv_flops_params(input_shape: Tuple[int, int, int], output_channels: int,
                                kernel_size: int or Tuple[int, int], stride: int or Tuple[int, int],
                                padding: int or Tuple[int, int]):
    in_channels = input_shape[2]
    k_h, k_w = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
    s_h, s_w = (stride, stride) if isinstance(stride, int) else stride
    p_h, p_w = (padding, padding) if isinstance(padding, int) else padding
    out_height = (input_shape[0] - k_h + 2 * p_h) // s_h + 1
    out_width = (input_shape[1] - k_w + 2 * p_w) // s_w + 1
    if out_height <= 0 or out_width <= 0:
        warnings.warn(f"Conv produced non-positive spatial dims: {out_height}x{out_width}, returning zeros")
        return 0, 0, (max(0, out_height), max(0, out_width), output_channels)
    params = (k_h * k_w * in_channels + 1) * output_channels
    num_macs_per_output_channel = k_h * k_w * in_channels
    mults = num_macs_per_output_channel * out_height * out_width * output_channels
    adds = num_macs_per_output_channel * out_height * out_width * output_channels
    total_flops = mults + adds
    return params, total_flops, (out_height, out_width, output_channels)

def calculate_bn_flops_params(input_shape: Tuple[int, int, int]):
    num_features = input_shape[2]
    params = 2 * num_features
    total_elements = input_shape[0] * input_shape[1] * input_shape[2]
    flops = 2 * total_elements
    return params, flops, input_shape

def calculate_relu_flops(input_shape: Tuple[int, int, int]):
    total_elements = input_shape[0] * input_shape[1] * input_shape[2]
    flops = total_elements
    return 0, flops, input_shape

def calculate_pooling_flops_params(input_shape: Tuple[int,int,int], kernel_size, stride):
    k_h, k_w = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
    s_h, s_w = (stride, stride) if isinstance(stride, int) else stride
    out_height = (input_shape[0] - k_h) // s_h + 1
    out_width = (input_shape[1] - k_w) // s_w + 1
    if out_height <= 0 or out_width <= 0:
        warnings.warn("Pooling produced non-positive spatial dims")
        return 0, 0, (max(0,out_height), max(0,out_width), input_shape[2])
    return 0, 0, (out_height, out_width, input_shape[2])

def calculate_unpooling_flops_params(input_shape: Tuple[int,int,int], kernel_size, stride, output_size: Tuple[int,int,int]):
    return 0, 0, (output_size[0], output_size[1], input_shape[2])

def calculate_concat_flops_params(input_shapes: List[Tuple[int,int,int]]):
    first = input_shapes[0]
    total_c = sum(s[2] for s in input_shapes)
    return 0,0,(first[0], first[1], total_c)

def get_activation_size_mb(shape: Tuple[int,...], bytes_per_element=4):
    num = 1
    for d in shape:
        num *= d
    return (num * bytes_per_element) / (1024**2)

class SUMNet_all_bn(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super(SUMNet_all_bn, self).__init__()
        # Encoder
        self.conv1     = nn.Conv2d(in_ch, 64, 3, padding=1)
        self.bn1       = nn.BatchNorm2d(64)
        self.conv2     = nn.Conv2d(64, 128, 3, padding=1)
        self.bn2       = nn.BatchNorm2d(128)
        self.pool1     = nn.MaxPool2d(2, 2, return_indices = True)
        self.conv3a    = nn.Conv2d(128, 256, 3, padding=1)
        self.bn3a      = nn.BatchNorm2d(256)
        self.conv3b    = nn.Conv2d(256, 256, 3, padding=1)
        self.bn3b      = nn.BatchNorm2d(256)
        self.pool2     = nn.MaxPool2d(2, 2, return_indices = True)
        self.conv4a    = nn.Conv2d(256, 512, 3, padding=1)
        self.bn4a      = nn.BatchNorm2d(512)
        self.conv4b    = nn.Conv2d(512, 512, 3, padding=1)
        self.bn4b      = nn.BatchNorm2d(512)
        self.pool3     = nn.MaxPool2d(2, 2, return_indices = True)
        self.conv5a    = nn.Conv2d(512, 512, 3, padding=1)
        self.bn5a      = nn.BatchNorm2d(512)
        self.conv5b    = nn.Conv2d(512, 512, 3, padding=1)
        self.bn5b      = nn.BatchNorm2d(512)
        self.pool4     = nn.MaxPool2d(2, 2, return_indices = True)
        # Decoder
        self.unpool4   = nn.MaxUnpool2d(2, 2)
        self.donv5b    = nn.Conv2d(1024, 512, 3, padding = 1)
        self.dbn5b     = nn.BatchNorm2d(512)
        self.donv5a    = nn.Conv2d(512, 512, 3, padding = 1)
        self.dbn5a     = nn.BatchNorm2d(512)
        self.unpool3   = nn.MaxUnpool2d(2, 2)
        self.donv4b    = nn.Conv2d(1024, 512, 3, padding = 1)
        self.dbn4b     = nn.BatchNorm2d(512)
        self.donv4a    = nn.Conv2d(512, 256, 3, padding = 1)
        self.dbn4a     = nn.BatchNorm2d(256)
        self.unpool2   = nn.MaxUnpool2d(2, 2)
        self.donv3b    = nn.Conv2d(512, 256, 3, padding = 1)
        self.dbn3b     = nn.BatchNorm2d(256)
        self.donv3a    = nn.Conv2d(256, 128, 3, padding = 1)
        self.dbn3a     = nn.BatchNorm2d(128)
        self.unpool1   = nn.MaxUnpool2d(2, 2)
        self.donv2     = nn.Conv2d(128, 64, 3, padding = 1)
        self.dbn2      = nn.BatchNorm2d(64)
        self.donv1     = nn.Conv2d(128, 32, 3, padding = 1)
        self.dbn1      = nn.BatchNorm2d(32)
        self.output    = nn.Conv2d(32, out_ch, 1)
        self.activation_shapes = []
        self.activation_memories = []

    def forward(self, x):
        self.activation_shapes = []
        self.activation_memories = []
        self._track_activation(x, "input")
        conv1_out = self.conv1(x); conv1_bn = self.bn1(conv1_out); conv1 = F.relu(conv1_bn, inplace=False)
        self._track_activation(conv1, "conv1")
        conv2_out = self.conv2(conv1); conv2_bn = self.bn2(conv2_out); conv2 = F.relu(conv2_bn, inplace=False)
        self._track_activation(conv2, "conv2")
        pool1, idxs1 = self.pool1(conv2); self._track_activation(pool1, "pool1")
        conv3a_out = self.conv3a(pool1); conv3a_bn = self.bn3a(conv3a_out); conv3a = F.relu(conv3a_bn, inplace=False)
        self._track_activation(conv3a, "conv3a")
        conv3b_out = self.conv3b(conv3a); conv3b_bn = self.bn3b(conv3b_out); conv3b = F.relu(conv3b_bn, inplace=False)
        self._track_activation(conv3b, "conv3b")
        pool2, idxs2 = self.pool2(conv3b); self._track_activation(pool2, "pool2")
        conv4a_out = self.conv4a(pool2); conv4a_bn = self.bn4a(conv4a_out); conv4a = F.relu(conv4a_bn, inplace=False)
        self._track_activation(conv4a, "conv4a")
        conv4b_out = self.conv4b(conv4a); conv4b_bn = self.bn4b(conv4b_out); conv4b = F.relu(conv4b_bn, inplace=False)
        self._track_activation(conv4b, "conv4b")
        pool3, idxs3 = self.pool3(conv4b); self._track_activation(pool3, "pool3")
        conv5a_out = self.conv5a(pool3); conv5a_bn = self.bn5a(conv5a_out); conv5a = F.relu(conv5a_bn, inplace=False)
        self._track_activation(conv5a, "conv5a")
        conv5b_out = self.conv5b(conv5a); conv5b_bn = self.bn5b(conv5b_out); conv5b = F.relu(conv5b_bn, inplace=False)
        self._track_activation(conv5b, "conv5b")
        pool4, idxs4 = self.pool4(conv5b); self._track_activation(pool4, "pool4")
        unpool4 = self.unpool4(pool4, idxs4, output_size=conv5b.size()); self._track_activation(unpool4, "unpool4")
        donv5b_in = torch.cat([unpool4, conv5b], 1); self._track_activation(donv5b_in, "donv5b_in")
        donv5b_out = self.donv5b(donv5b_in); donv5b_bn = self.dbn5b(donv5b_out); donv5b = F.relu(donv5b_bn, inplace=False)
        self._track_activation(donv5b, "donv5b")
        donv5a_out = self.donv5a(donv5b); donv5a_bn = self.dbn5a(donv5a_out); donv5a = F.relu(donv5a_bn, inplace=False)
        self._track_activation(donv5a, "donv5a")
        unpool3 = self.unpool3(donv5a, idxs3, output_size=conv4b.size()); self._track_activation(unpool3, "unpool3")
        donv4b_in = torch.cat([unpool3, conv4b], 1); self._track_activation(donv4b_in, "donv4b_in")
        donv4b_out = self.donv4b(donv4b_in); donv4b_bn = self.dbn4b(donv4b_out); donv4b = F.relu(donv4b_bn, inplace=False)
        self._track_activation(donv4b, "donv4b")
        donv4a_out = self.donv4a(donv4b); donv4a_bn = self.dbn4a(donv4a_out); donv4a = F.relu(donv4a_bn, inplace=False)
        self._track_activation(donv4a, "donv4a")
        unpool2 = self.unpool2(donv4a, idxs2, output_size=conv3b.size()); self._track_activation(unpool2, "unpool2")
        donv3b_in = torch.cat([unpool2, conv3b], 1); self._track_activation(donv3b_in, "donv3b_in")
        donv3b_out = self.donv3b(donv3b_in); donv3b_bn = self.dbn3b(donv3b_out); donv3b = F.relu(donv3b_bn, inplace=False)
        self._track_activation(donv3b, "donv3b")
        donv3a_out = self.donv3a(donv3b); donv3a_bn = self.dbn3a(donv3a_out); donv3a = F.relu(donv3a_bn, inplace=False)
        self._track_activation(donv3a, "donv3a")
        unpool1 = self.unpool1(donv3a, idxs1, output_size=conv2.size()); self._track_activation(unpool1, "unpool1")
        donv2_out = self.donv2(unpool1); donv2_bn = self.dbn2(donv2_out); donv2 = F.relu(donv2_bn, inplace=False)
        self._track_activation(donv2, "donv2")
        donv1_in = torch.cat([donv2, conv1], 1); self._track_activation(donv1_in, "donv1_in")
        donv1_out = self.donv1(donv1_in); donv1_bn = self.dbn1(donv1_out); donv1 = F.relu(donv1_bn, inplace=False)
        self._track_activation(donv1, "donv1")
        output = self.output(donv1); self._track_activation(output, "output")
        return output

    def _track_activation(self, tensor: torch.Tensor, name: str):
        shape = tuple(tensor.shape)
        memory_mb = tensor.numel() * tensor.element_size() / (1024 * 1024)
        self.activation_shapes.append((name, shape))
        self.activation_memories.append((name, memory_mb))

    def get_total_activation_memory(self):
        return sum(mem for _, mem in self.activation_memories)

def total_sumnet_flops_params(input_res: Tuple[int,int], in_ch: int, out_ch: int):
    total_params = 0
    total_flops = 0
    total_activation_size_mb = 0
    curr_shape = (input_res[0], input_res[1], in_ch)
    total_activation_size_mb += get_activation_size_mb(curr_shape)
    skip_shapes = {}
    def conv_bn_relu(in_shape, out_ch, k=3, s=1, p=1):
        nonlocal total_params, total_flops, total_activation_size_mb
        p_cnt, f_cnt, out_shape = calculate_conv_flops_params(in_shape, out_ch, k, s, p)
        total_params += p_cnt; total_flops += f_cnt; total_activation_size_mb += get_activation_size_mb(out_shape)
        p_cnt, f_cnt, out_shape = calculate_bn_flops_params(out_shape)
        total_params += p_cnt; total_flops += f_cnt; total_activation_size_mb += get_activation_size_mb(out_shape)
        p_cnt, f_cnt, out_shape = calculate_relu_flops(out_shape)
        total_flops += f_cnt; total_activation_size_mb += get_activation_size_mb(out_shape)
        return out_shape
    curr_shape = conv_bn_relu(curr_shape, 64); skip_shapes['conv1'] = curr_shape
    curr_shape = conv_bn_relu(curr_shape, 128); skip_shapes['conv2'] = curr_shape
    p_cnt, f_cnt, curr_shape = calculate_pooling_flops_params(curr_shape, 2, 2)
    total_params += p_cnt; total_flops += f_cnt; total_activation_size_mb += get_activation_size_mb(curr_shape)
    curr_shape = conv_bn_relu(curr_shape, 256)
    curr_shape = conv_bn_relu(curr_shape, 256); skip_shapes['conv3b'] = curr_shape
    p_cnt, f_cnt, curr_shape = calculate_pooling_flops_params(curr_shape, 2, 2)
    total_params += p_cnt; total_flops += f_cnt; total_activation_size_mb += get_activation_size_mb(curr_shape)
    curr_shape = conv_bn_relu(curr_shape, 512)
    curr_shape = conv_bn_relu(curr_shape, 512); skip_shapes['conv4b'] = curr_shape
    p_cnt, f_cnt, curr_shape = calculate_pooling_flops_params(curr_shape, 2, 2)
    total_params += p_cnt; total_flops += f_cnt; total_activation_size_mb += get_activation_size_mb(curr_shape)
    curr_shape = conv_bn_relu(curr_shape, 512)
    curr_shape = conv_bn_relu(curr_shape, 512); skip_shapes['conv5b'] = curr_shape
    p_cnt, f_cnt, curr_shape = calculate_pooling_flops_params(curr_shape, 2, 2)
    total_params += p_cnt; total_flops += f_cnt; total_activation_size_mb += get_activation_size_mb(curr_shape)
    p_cnt, f_cnt, curr_shape = calculate_unpooling_flops_params(curr_shape, 2, 2, skip_shapes['conv5b'])
    total_params += p_cnt; total_flops += f_cnt; total_activation_size_mb += get_activation_size_mb(curr_shape)
    p_cnt, f_cnt, curr_shape = calculate_concat_flops_params([curr_shape, skip_shapes['conv5b']])
    total_params += p_cnt; total_flops += f_cnt; total_activation_size_mb += get_activation_size_mb(curr_shape)
    curr_shape = conv_bn_relu(curr_shape, 512)
    curr_shape = conv_bn_relu(curr_shape, 512)
    p_cnt, f_cnt, curr_shape = calculate_unpooling_flops_params(curr_shape, 2, 2, skip_shapes['conv4b'])
    total_params += p_cnt; total_flops += f_cnt; total_activation_size_mb += get_activation_size_mb(curr_shape)
    p_cnt, f_cnt, curr_shape = calculate_concat_flops_params([curr_shape, skip_shapes['conv4b']])
    total_params += p_cnt; total_flops += f_cnt; total_activation_size_mb += get_activation_size_mb(curr_shape)
    curr_shape = conv_bn_relu(curr_shape, 512)
    curr_shape = conv_bn_relu(curr_shape, 256)
    p_cnt, f_cnt, curr_shape = calculate_unpooling_flops_params(curr_shape, 2, 2, skip_shapes['conv3b'])
    total_params += p_cnt; total_flops += f_cnt; total_activation_size_mb += get_activation_size_mb(curr_shape)
    p_cnt, f_cnt, curr_shape = calculate_concat_flops_params([curr_shape, skip_shapes['conv3b']])
    total_params += p_cnt; total_flops += f_cnt; total_activation_size_mb += get_activation_size_mb(curr_shape)
    curr_shape = conv_bn_relu(curr_shape, 256)
    curr_shape = conv_bn_relu(curr_shape, 128)
    p_cnt, f_cnt, curr_shape = calculate_unpooling_flops_params(curr_shape, 2, 2, skip_shapes['conv2'])
    total_params += p_cnt; total_flops += f_cnt; total_activation_size_mb += get_activation_size_mb(curr_shape)
    curr_shape = conv_bn_relu(curr_shape, 64)
    p_cnt, f_cnt, curr_shape = calculate_concat_flops_params([curr_shape, skip_shapes['conv1']])
    total_params += p_cnt; total_flops += f_cnt; total_activation_size_mb += get_activation_size_mb(curr_shape)
    curr_shape = conv_bn_relu(curr_shape, 32)
    p_cnt, f_cnt, curr_shape = calculate_conv_flops_params(curr_shape, out_ch, 1, 1, 0)
    total_params += p_cnt; total_flops += f_cnt; total_activation_size_mb += get_activation_size_mb(curr_shape)
    return total_params, total_flops, total_activation_size_mb

def measure_latency_cuda(model: nn.Module, input_tensor: torch.Tensor, warmup_runs: int = 10, num_runs: int = 100):
    model.eval()
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(input_tensor)
    torch.cuda.synchronize()
    timings = []
    with torch.no_grad():
        for _ in range(num_runs):
            start = torch.cuda.Event(enable_timing=True); end = torch.cuda.Event(enable_timing=True)
            start.record()
            _ = model(input_tensor)
            end.record()
            torch.cuda.synchronize()
            timings.append(start.elapsed_time(end))
    return np.array(timings)

def measure_latency_cpu(model: nn.Module, input_tensor: torch.Tensor, warmup_runs: int = 7, num_runs: int = 25):
    model.eval()
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(input_tensor)
    timings = []
    with torch.no_grad():
        for _ in range(num_runs):
            t0 = time.perf_counter()
            _ = model(input_tensor)
            t1 = time.perf_counter()
            timings.append((t1 - t0) * 1000.0)
    return np.array(timings)

def export_to_onnx(model: nn.Module, onnx_path: str, input_size: Tuple[int,int,int], input_name='input', opset=13, verbose=False):
    model_cpu = model.to('cpu').eval()
    C, H, W = input_size
    dummy = torch.randn(1, C, H, W, device='cpu')
    torch.onnx.export(
        model_cpu, dummy, onnx_path, opset_version=opset,
        input_names=[input_name], output_names=['output'],
        dynamic_axes={input_name: {0: 'batch'}, 'output': {0: 'batch'}},
        do_constant_folding=True, export_params=True, verbose=verbose
    )
    return onnx_path

def get_file_size_mb(path: str) -> float:
    try:
        return os.path.getsize(path) / (1024.0**2)
    except Exception:
        return 0.0

def run_onnxruntime_inference(onnx_path: str, input_size: Tuple[int,int,int], warmup=3, runs=30):
    if not ORT_AVAILABLE:
        raise RuntimeError("onnxruntime not available")
    providers = ['CPUExecutionProvider']
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    sess = ort.InferenceSession(onnx_path, sess_options=so, providers=providers)
    C,H,W = input_size
    x = np.random.randn(1, C, H, W).astype(np.float32)
    feed = {'input': x}
    for _ in range(warmup):
        _ = sess.run(None, feed)
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        _ = sess.run(None, feed)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)
    times = sorted(times)
    stats = {
        'mean_ms': float(np.mean(times)),
        'min_ms': float(times[0]),
        'p50_ms': float(np.median(times)),
        'p95_ms': float(np.percentile(times, 95)),
        'max_ms': float(times[-1]),
        'runs': len(times),
    }
    return stats

def quantize_dynamic_onnx(fp32_path: str, int8_out: str):
    if not ORT_QUANT_AVAILABLE:
        raise RuntimeError("onnxruntime.quantization not available")
    quantize_dynamic(model_input=fp32_path, model_output=int8_out, weight_type=QuantType.QInt8)

def profile_sumnet_all(input_sizes: List[Tuple[int,int]],
                        in_ch: int = 3, out_ch: int = 1,
                        do_onnx: bool = True, do_quant: bool = True,
                        out_dir: str = "onnx_sumnet"):
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device} | ONNX: {ONNX_AVAILABLE} | ORT: {ORT_AVAILABLE} | ORT-quant: {ORT_QUANT_AVAILABLE} | ptflops: {PTFLOPS_AVAILABLE}")
    print("="*140)
    header = ("Resolution      | Params(M) | TheoParams(M) | TheoFLOPs(G) | ActCum(MB) | ActCumINT8(MB) | PyTorch Lat(ms) | "
              "ONNX-FP32(ms) | ONNX-INT8(ms) | FP32 size | INT8 size")
    print(header)
    print("-"*len(header))
    for (H, W) in input_sizes:
        inp_str = f"{H}x{W}"
        model = SUMNet_all_bn(in_ch, out_ch).to(device)
        model.eval()
        theo_params, theo_flops, theo_act_mb = total_sumnet_flops_params((H, W), in_ch, out_ch)
        actual_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        x = torch.randn(1, in_ch, H, W).to(device)
        try:
            if device.type == 'cuda':
                lat_arr = measure_latency_cuda(model, x, warmup_runs=7, num_runs=80)
            else:
                lat_arr = measure_latency_cpu(model, x, warmup_runs=7, num_runs=40)
            mean_lat = float(np.mean(lat_arr))
        except Exception as e:
            mean_lat = float('nan')
            print(f"[WARN] Latency measurement failed at {H}x{W}: {e}")
        try:
            with torch.no_grad():
                _ = model(x)
                actual_act_mb = model.get_total_activation_memory()
        except Exception as e:
            actual_act_mb = float('nan')
            print(f"[WARN] Activation forward failed at {H}x{W}: {e}")
        int8_act_mb = actual_act_mb / 4 if not np.isnan(actual_act_mb) else float('nan')
        pt_str = "N/A"
        if PTFLOPS_AVAILABLE:
            try:
                fresh = SUMNet_all_bn(in_ch, out_ch)
                macs, params_pt = get_model_complexity_info(fresh, (in_ch, H, W), as_strings=False, print_per_layer_stat=False, verbose=False)
                pt_str = f"{macs/1e9:.2f}"
                del fresh
                gc.collect()
            except Exception as e:
                pt_str = "err"
        fp32_path = os.path.join(out_dir, f"sumnet_{H}x{W}_fp32.onnx")
        int8_path = os.path.join(out_dir, f"sumnet_{H}x{W}_int8.onnx")
        onnx_fp32_ms = "N/A"
        onnx_int8_ms = "N/A"
        fp32_size = "N/A"
        int8_size = "N/A"
        if do_onnx and ONNX_AVAILABLE:
            try:
                export_to_onnx(model, fp32_path, (in_ch, H, W), input_name='input', opset=13)
                fp32_size = f"{get_file_size_mb(fp32_path):.2f}MB"
                if ORT_AVAILABLE:
                    try:
                        ort_stats = run_onnxruntime_inference(fp32_path, (in_ch, H, W), warmup=3, runs=30)
                        onnx_fp32_ms = f"{ort_stats['mean_ms']:.2f}"
                    except Exception as e:
                        onnx_fp32_ms = f"err:{str(e)[:40]}"
            except Exception as e:
                onnx_fp32_ms = f"export_err:{str(e)[:60]}"
                fp32_size = "ERR"
        if do_onnx and ONNX_AVAILABLE and ORT_QUANT_AVAILABLE and fp32_size != "ERR" and os.path.exists(fp32_path):
            try:
                quantize_dynamic_onnx(fp32_path, int8_path)
                int8_size = f"{get_file_size_mb(int8_path):.2f}MB"
                if ORT_AVAILABLE:
                    try:
                        int8_stats = run_onnxruntime_inference(int8_path, (in_ch, H, W), warmup=3, runs=30)
                        onnx_int8_ms = f"{int8_stats['mean_ms']:.2f}"
                    except Exception as e:
                        onnx_int8_ms = f"err:{str(e)[:40]}"
            except Exception as e:
                onnx_int8_ms = f"quant_err:{str(e)[:60]}"
                int8_size = "ERR"
        else:
            if do_quant and not ORT_QUANT_AVAILABLE:
                onnx_int8_ms = "quant-unavail"
        print(f"{inp_str:14} | {actual_params/1e6:8.2f}M | {theo_params/1e6:12.2f}M | "
              f"{theo_flops/1e9:11.2f}G | {actual_act_mb:9.2f}MB | {int8_act_mb:13.2f}MB | {mean_lat:14.2f} | "
              f"{str(onnx_fp32_ms):>11} | {str(onnx_int8_ms):>11} | {fp32_size:>8} | {int8_size:>8}")
        del model
        gc.collect()
    print("\nProfiling complete.")

if __name__ == "__main__":
    INPUT_SIZES = [
        (360, 640),
        (720, 1280),
        (760, 1360),
        (900, 1600),
        (1080, 1920),
        (1152, 2048),
        (1440, 2560),
        (2160, 3840),
    ]
    DO_ONNX = True
    DO_QUANT = True
    OUT_DIR = "onnx_sumnet"
    print("SUMNet_all_bn Full Profiling + ONNX/ORT Benchmarking (INT8 activation reporting)")
    print("="*120)
    profile_sumnet_all(INPUT_SIZES, in_ch=3, out_ch=1, do_onnx=DO_ONNX, do_quant=DO_QUANT, out_dir=OUT_DIR)
