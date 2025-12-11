#!/usr/bin/env python3
"""
UNext INT8 Quantisation & Benchmark Pipeline (CPU-optimised)
This script uses the UNext model definition and applies the exact same
profiling, ONNX export, quantization, and benchmarking pipeline as
the RTFormer-Slim and LinkNet scripts.
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

# Specific import for UNext
try:
    from timm.layers import to_2tuple, DropPath
except ImportError:
    print("Error: 'timm' library not found. Please install with: pip install timm")
    exit()

# ----------  ONNX / ORT imports (same as pipeline)  ----------
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
#  UNext model definition
# ===================================================================

class DWConv(nn.Module):
    """ Depth-wise convolution """
    def __init__(self, dim=768):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).reshape(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        return x

class shiftmlp(nn.Module):
    """ Shifted MLP """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.fc2(x)
        return x

class shiftedBlock(nn.Module):
    """ Block combining LayerNorm, MLP, and residual connection """
    def __init__(self, dim, mlp_ratio=4., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = shiftmlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, H, W):
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x

class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding with overlapping patches """
    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        stride = to_2tuple(stride)
        padding = (patch_size[0] // 2, patch_size[1] // 2) if patch_size[0] > 1 else (0,0)

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride, padding=padding)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        H_out, W_out = x.shape[-2:]
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H_out, W_out

class UNext(nn.Module):
    def __init__(self, num_classes=4, input_channels=3, depths=[1, 1], mlp_ratios=[4., 4.], drop_path_rate=0.1, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dims = [16, 32, 128, 160, 256]
        self.decoder_dims = [160, 128, 32, 16, 16]
        self.depths = depths
        self.mlp_ratios = mlp_ratios

        # Encoder
        self.encoder1 = nn.Conv2d(input_channels, self.dims[0], 3, padding=1)
        self.ebn1 = nn.BatchNorm2d(self.dims[0])
        self.encoder2 = nn.Conv2d(self.dims[0], self.dims[1], 3, padding=1)
        self.ebn2 = nn.BatchNorm2d(self.dims[1])
        self.encoder3 = nn.Conv2d(self.dims[1], self.dims[2], 3, padding=1)
        self.ebn3 = nn.BatchNorm2d(self.dims[2])
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(2, 2)

        # Transformer Encoder Stages
        self.patch_embed3 = OverlapPatchEmbed(in_chans=self.dims[2], embed_dim=self.dims[3], patch_size=3, stride=2)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths) * 2)] # Total blocks
        cur = 0
        self.block1 = nn.ModuleList([shiftedBlock(dim=self.dims[3], mlp_ratio=mlp_ratios[0], drop_path=dpr[cur + i], norm_layer=norm_layer)
                                     for i in range(depths[0])])
        self.norm3 = norm_layer(self.dims[3])
        cur += depths[0]

        self.patch_embed4 = OverlapPatchEmbed(in_chans=self.dims[3], embed_dim=self.dims[4], patch_size=3, stride=2)
        self.block2 = nn.ModuleList([shiftedBlock(dim=self.dims[4], mlp_ratio=mlp_ratios[1], drop_path=dpr[cur + i], norm_layer=norm_layer)
                                     for i in range(depths[1])])
        self.norm4 = norm_layer(self.dims[4])
        cur += depths[1]

        # Decoder
        self.decoder1 = nn.Conv2d(self.dims[4] + self.dims[3], self.decoder_dims[0], 3, padding=1)
        self.dbn1 = nn.BatchNorm2d(self.decoder_dims[0])
        self.dblock1 = nn.ModuleList([shiftedBlock(dim=self.decoder_dims[0], mlp_ratio=mlp_ratios[0], drop_path=dpr[cur + i], norm_layer=norm_layer)
                                     for i in range(depths[0])])
        self.dnorm3 = norm_layer(self.decoder_dims[0])
        cur += depths[0]
        
        self.decoder2 = nn.Conv2d(self.decoder_dims[0] + self.dims[2], self.decoder_dims[1], 3, padding=1)
        self.dbn2 = nn.BatchNorm2d(self.decoder_dims[1])
        self.dblock2 = nn.ModuleList([shiftedBlock(dim=self.decoder_dims[1], mlp_ratio=mlp_ratios[1], drop_path=dpr[cur + i], norm_layer=norm_layer)
                                     for i in range(depths[1])])
        self.dnorm4 = norm_layer(self.decoder_dims[1])
        
        self.decoder3 = nn.Conv2d(self.decoder_dims[1] + self.dims[1], self.decoder_dims[2], 3, padding=1)
        self.dbn3 = nn.BatchNorm2d(self.decoder_dims[2])
        self.decoder4 = nn.Conv2d(self.decoder_dims[2] + self.dims[0], self.decoder_dims[3], 3, padding=1)
        self.dbn4 = nn.BatchNorm2d(self.decoder_dims[3])
        self.decoder5 = nn.Conv2d(self.decoder_dims[3], self.decoder_dims[4], 3, padding=1)
        self.dbn5 = nn.BatchNorm2d(self.decoder_dims[4])
        self.final = nn.Conv2d(self.decoder_dims[4], num_classes, 1)

    def _decode_block(self, x, skip, conv, bn, blocks, norm):
        x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.relu(bn(conv(x)))
        B_dec, C_dec, H_dec, W_dec = x.shape
        x_flat = x.flatten(2).transpose(1, 2)
        for blk in blocks:
            x_flat = blk(x_flat, H_dec, W_dec)
        x = norm(x_flat).transpose(1, 2).reshape(B_dec, C_dec, H_dec, W_dec)
        return x

    def _upsample(self, x, skip, conv, bn, target_size_override=None):
        if target_size_override:
            size = target_size_override
        else:
            size = skip.shape[2:] if skip is not None else None
        
        if size is None:
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        else:
            x = F.interpolate(x, size=size, mode='bilinear', align_corners=False)
        
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        return self.relu(bn(conv(x)))

    def forward(self, x):
        input_size = x.shape[2:] 
        e1 = self.relu(self.ebn1(self.encoder1(x)))     
        e2_in = self.maxpool(e1)                        
        e2 = self.relu(self.ebn2(self.encoder2(e2_in)))
        e3_in = self.maxpool(e2)                        
        e3 = self.relu(self.ebn3(self.encoder3(e3_in)))
        
        t4_in = self.maxpool(e3) 
        t4, h4, w4 = self.patch_embed3(t4_in) 
        for blk in self.block1:
            t4 = blk(t4, h4, w4)
        t4 = self.norm3(t4)
        t4_spatial = t4.transpose(1, 2).reshape(t4.shape[0], self.dims[3], h4, w4)
        
        t5, h5, w5 = self.patch_embed4(t4_spatial) 
        for blk in self.block2:
            t5 = blk(t5, h5, w5)
        t5 = self.norm4(t5)
        t5_spatial = t5.transpose(1, 2).reshape(t5.shape[0], self.dims[4], h5, w5)
        
        d1 = self._decode_block(t5_spatial, t4_spatial, self.decoder1, self.dbn1, self.dblock1, self.dnorm3)
        d2 = self._decode_block(d1, e3, self.decoder2, self.dbn2, self.dblock2, self.dnorm4)
        d3 = self._upsample(d2, e2, self.decoder3, self.dbn3)
        d4 = self._upsample(d3, e1, self.decoder4, self.dbn4)
        d5 = self._upsample(d4, None, self.decoder5, self.dbn5, target_size_override=input_size)
        
        out = self.final(d5)
        out = F.interpolate(out, size=input_size, mode='bilinear', align_corners=False)
        return out

class UNext_S(UNext):
    """ UNext-Small variant """
    def __init__(self, num_classes=4, input_channels=3, depths=[1, 1], mlp_ratios=[4., 4.], drop_path_rate=0.1, norm_layer=nn.LayerNorm):
        super().__init__(num_classes, input_channels, depths, mlp_ratios, drop_path_rate, norm_layer)
        self.dims = [8, 16, 32, 64, 128]
        self.decoder_dims = [64, 32, 16, 8, 8]
        self.encoder1 = nn.Conv2d(input_channels, self.dims[0], 3, padding=1)
        self.ebn1 = nn.BatchNorm2d(self.dims[0])
        self.encoder2 = nn.Conv2d(self.dims[0], self.dims[1], 3, padding=1)
        self.ebn2 = nn.BatchNorm2d(self.dims[1])
        self.encoder3 = nn.Conv2d(self.dims[1], self.dims[2], 3, padding=1)
        self.ebn3 = nn.BatchNorm2d(self.dims[2])
        self.patch_embed3 = OverlapPatchEmbed(in_chans=self.dims[2], embed_dim=self.dims[3], patch_size=3, stride=2)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths) * 2)]
        cur = 0
        self.block1 = nn.ModuleList([shiftedBlock(dim=self.dims[3], mlp_ratio=mlp_ratios[0], drop_path=dpr[cur + i], norm_layer=norm_layer)
                                     for i in range(depths[0])])
        self.norm3 = norm_layer(self.dims[3])
        cur += depths[0]
        self.patch_embed4 = OverlapPatchEmbed(in_chans=self.dims[3], embed_dim=self.dims[4], patch_size=3, stride=2)
        self.block2 = nn.ModuleList([shiftedBlock(dim=self.dims[4], mlp_ratio=mlp_ratios[1], drop_path=dpr[cur + i], norm_layer=norm_layer)
                                     for i in range(depths[1])])
        self.norm4 = norm_layer(self.dims[4])
        cur += depths[1]
        self.decoder1 = nn.Conv2d(self.dims[4] + self.dims[3], self.decoder_dims[0], 3, padding=1)
        self.dbn1 = nn.BatchNorm2d(self.decoder_dims[0])
        self.dblock1 = nn.ModuleList([shiftedBlock(dim=self.decoder_dims[0], mlp_ratio=mlp_ratios[0], drop_path=dpr[cur + i], norm_layer=norm_layer)
                                     for i in range(depths[0])])
        self.dnorm3 = norm_layer(self.decoder_dims[0])
        cur += depths[0]
        self.decoder2 = nn.Conv2d(self.decoder_dims[0] + self.dims[2], self.decoder_dims[1], 3, padding=1)
        self.dbn2 = nn.BatchNorm2d(self.decoder_dims[1])
        self.dblock2 = nn.ModuleList([shiftedBlock(dim=self.decoder_dims[1], mlp_ratio=mlp_ratios[1], drop_path=dpr[cur + i], norm_layer=norm_layer)
                                     for i in range(depths[1])])
        self.dnorm4 = norm_layer(self.decoder_dims[1])
        self.decoder3 = nn.Conv2d(self.decoder_dims[1] + self.dims[1], self.decoder_dims[2], 3, padding=1)
        self.dbn3 = nn.BatchNorm2d(self.decoder_dims[2])
        self.decoder4 = nn.Conv2d(self.decoder_dims[2] + self.dims[0], self.decoder_dims[3], 3, padding=1)
        self.dbn4 = nn.BatchNorm2d(self.decoder_dims[3])
        self.decoder5 = nn.Conv2d(self.decoder_dims[3], self.decoder_dims[4], 3, padding=1)
        self.dbn5 = nn.BatchNorm2d(self.decoder_dims[4])
        self.final = nn.Conv2d(self.decoder_dims[4], num_classes, 1)

# ===================================================================
#  Profiler Classes and Utilities
# ===================================================================

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
    # Not used for INT8; left for completeness
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

def print_analysis_table(input_size_str, flops_g, params_m, total_act_mb_fp32, peak_act_mb_fp32, lat_fp32,
                        total_act_mb_int8, peak_act_mb_int8, lat_int8, model_fp32_size_mb=None, model_int8_size_mb=None):
    header = (" Input Size | FLOPs (G) | Params (M) | Total Act FP32 (MB) | Peak Act FP32 (MB) | "
              "Latency FP32 (ms) | Total Act INT8 (MB) | Peak Act INT8 (MB) | Latency INT8 (ms)")
    print(header)
    print("-" * len(header))
    print(f" {input_size_str:10s} | {flops_g:9.3f} | {params_m:10.3f} | {total_act_mb_fp32:19.2f} | "
          f"{peak_act_mb_fp32:18.2f} | {lat_fp32:17.2f} | {total_act_mb_int8:19.2f} | "
          f"{peak_act_mb_int8:18.2f} | {lat_int8:16.2f}")

def process_resolution(resolution, num_classes=4, calib_size=64, per_channel=False, warmup=10, runs=100):
    c, w, h = resolution  # (C, W, H)
    input_shape = (1, c, h, w)
    input_size_str = f"{w}x{h}"

    print(f"\n{'='*80}")
    print(f"Processing resolution: {input_size_str} (Input Shape: {input_shape})")
    print(f"{'='*80}")

    fixed_depths = [1, 1]
    fixed_mlp_ratios = [4., 4.]
    model = UNext(
        num_classes=num_classes, 
        input_channels=c, 
        depths=fixed_depths, 
        mlp_ratios=fixed_mlp_ratios
    )
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

    onnx_path = f"unext_{w}x{h}.onnx"
    quant_path = f"unext_{w}x{h}_int8.onnx"

    try:
        export_to_onnx(model, onnx_path, input_shape) 
        fp32_info = analyze_onnx_model(onnx_path)
        if fp32_info:
            print(f"    FP32 ONNX: {fp32_info['num_nodes']} nodes, {fp32_info['num_params']:,} params, {fp32_info['file_size_mb']:.2f} MB")
    except Exception as e:
        print(f"[✗] ONNX export failed: {e}")
        fp32_info = None

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

    # Estimate INT8 activation memory (divide FP32 by 4)
    int8_runtime_total_mb = total_act_mb_fp32 / 4
    int8_runtime_peak_mb = peak_act_mb_fp32 / 4

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

def main():
    print("\n" + "="*80)
    print("UNEXT INT8 QUANTIZATION & BENCHMARK PIPELINE (CPU-OPTIMIZED)")
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

    num_classes = 4
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
            print(f"[✗] Failed to process {res} (WxH: {res[1]}x{res[2]}): {e}")
            import traceback
            traceback.print_exc()
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
    print("-")

if __name__ == "__main__":
    main()
