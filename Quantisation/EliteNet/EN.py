#!/usr/bin/env python3
"""
EliteNet (UNetEncoderWithSkip) INT8 Quantisation & Benchmark Pipeline (CPU-optimised)
This script uses the EliteNet (a U-Net variant) model and applies the exact
same profiling, ONNX export, quantization, and benchmarking pipeline as
the previous scripts.
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
#  EliteNet model definition (UNetEncoderWithSkip)
# ===================================================================

class ConvBlockNoSkip(nn.Module):
    """ Convolutional block used in the EliteNet architecture. """
    def __init__(self, in_c, out_c, k_sz=3, pool=True):
        super().__init__()
        pad = (k_sz - 1) // 2
        self.out_channels = out_c
        self.pool = nn.MaxPool2d(kernel_size=2) if pool else None

        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=k_sz, padding=pad, bias=True),
            nn.ReLU(),
            nn.BatchNorm2d(out_c),
            nn.Conv2d(out_c, out_c, kernel_size=k_sz, padding=pad, bias=True),
            nn.ReLU(),
            nn.BatchNorm2d(out_c)
        )

    def forward(self, x):
        if self.pool:
            x = self.pool(x)
        return self.block(x)

class EliteNet(nn.Module):
    """
    EliteNet: A U-Net style encoder-decoder with residual-like skip connections.
    Note: Activations are stored for skip connections, but only the final output
    is returned for ONNX export, and the intermediate 'activations' list is not
    used in the main ONNX forward path.
    """
    def __init__(self, in_c, out_c, layers, k_sz=3):
        super().__init__()
        # Check if output channels match final conv layer's output
        if out_c != in_c:
             warnings.warn(f"EliteNet's final convolution hardcoded to output {in_c} channels, but 'out_c' was set to {out_c}. Using {in_c}.")
        
        self.layers = layers
        self.in_c = in_c
        self.out_c = out_c

        # First block does not pool
        self.first = ConvBlockNoSkip(in_c, layers[0], k_sz, pool=False)

        self.down_path = nn.ModuleList()
        self.up_path = nn.ModuleList()

        # Encoder Down Path
        for i in range(len(layers) - 1):
            self.down_path.append(ConvBlockNoSkip(layers[i], layers[i+1], k_sz, pool=True))
            
        # Decoder Up Path (Transposed Convolutions)
        for i in range(len(layers) - 1):
            # The transposed conv takes layers[i+1] -> layers[i]
            self.up_path.append(nn.ConvTranspose2d(layers[i+1], layers[i], kernel_size=2, stride=2, bias=True))

        # Final convolution (maps back to input channel count 'in_c' per the original code)
        self.final_conv = nn.Conv2d(layers[0], in_c, kernel_size=1, bias=True)


    def forward(self, x):
        activations = [] # Stores encoder outputs for skip connections

        # --- Encoder ---
        x1 = self.first(x)
        activations.append(x1)
        x = x1
        for down in self.down_path:
            x = down(x)
            activations.append(x) # Encoder activations: [x1, x2, x3, ..., x_bottleneck]

        # --- Decoder ---
        # x is currently the bottleneck feature map
        # Skip connections are accessed in reverse order, excluding the bottleneck
        num_down_layers = len(self.layers) - 1
        
        for i, up in enumerate(reversed(self.up_path)):
            x = up(x)
            
            # Index for skip_connection: from the second to last layer (x_n-1) down to the first (x1)
            skip_connection = activations[num_down_layers - i - 1]

            # Handle spatial size mismatch by padding (needed for ONNX export)
            diff_h = skip_connection.size(2) - x.size(2)
            diff_w = skip_connection.size(3) - x.size(3)
            
            if diff_h != 0 or diff_w != 0:
                # Pad (left, right, top, bottom)
                x = F.pad(x, [diff_w // 2, diff_w - diff_w // 2,
                              diff_h // 2, diff_h - diff_h // 2])

            x = x + skip_connection # Additive skip connection
            # activations.append(x) # Only the final output is tracked for FLOP/Act in the profiler

        # --- Final Conv ---
        x = self.final_conv(x)
        # activations.append(x) # Final output

        return x # Return only the final tensor for ONNX compatibility

# ===================================================================
#  Everything below is the **identical** pipeline from the previous scripts
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
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                Cin = module.in_channels
                kh, kw = module.kernel_size if isinstance(module.kernel_size, tuple) else (module.kernel_size, module.kernel_size)
                groups = module.groups if hasattr(module, 'groups') else 1
                # MACs: Output_pixels * Cout * Cin_per_group * kernel_size
                macs = batch * Cout * H_out * W_out * (Cin // groups) * kh * kw
                flops = 2 * macs
                self.layer_flops.append((module.__class__.__name__, flops))
            
            # Note: MaxPool, BatchNorm, ReLU, and Addition (skip connection) are typically 
            # ignored or counted separately, but we stick to Conv/Linear here for consistency.
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
                # Note: nn.ConvTranspose2d is included here
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
            # Use shape inference to get better tensor sizes
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

        # Collect information from various parts of the graph
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
def process_resolution(resolution, in_channels, out_channels, layers, calib_size=64, per_channel=False, warmup=10, runs=100):
    c, w, h = resolution  # Note: (C, W, H) format from main
    input_shape = (1, c, h, w) # PyTorch/ONNX expect (B, C, H, W)
    input_size_str = f"{w}x{h}"

    print(f"\n{'='*80}")
    print(f"Processing resolution: {input_size_str} (Input Shape: {input_shape})")
    print(f"{'='*80}")

    # *** CHANGED MODEL ***
    model = EliteNet(
        in_c=in_channels, 
        out_c=out_channels,
        layers=layers
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
    peak_act_mb_fp32 = act_prof.get_stats()['peak_activation_mb'] # Recalculate peak for safety

    # *** CHANGED PATHS ***
    onnx_path = f"elitenet_{w}x{h}.onnx"
    quant_path = f"elitenet_{w}x{h}_int8.onnx"

    try:
        # Note: input_size for export_to_onnx must match (B, C, H, W)
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
    # Calib reader also needs (B, C, H, W)
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
            # Benchmark function needs (B, C, H, W)
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
            # Benchmark function needs (B, C, H, W)
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
    print("ELITENET (UNET ENCODER-DECODER) INT8 QUANTIZATION & BENCHMARK PIPELINE")
    print("="*80)

    # Note: Format is (Channels, Width, Height) - based on your previous models
    # Your provided list of resolutions is (H, W), so I swap them back to (W, H) then (C, W, H)
    resolutions_hw = [
        (360, 640), (720, 1280), (760, 1360), (900, 1600), 
        (1080, 1920), (1152, 2048), (1440, 2560), (3840, 2160)
    ]
    # Convert to (C, W, H) format for the pipeline
    resolutions = [(3, w, h) for h, w in resolutions_hw] 

    # *** ELITENET PARAMS ***
    in_channels = 3  
    # Since your final conv maps to 'in_c', we set out_channels to in_channels for consistency.
    out_channels = in_channels
    layers = [4, 8, 16, 32] # From your model's layers definition
    
    calib_size = 64
    per_channel = False
    warmup = 10
    runs = 100

    print(f"Resolutions to test: {len(resolutions)}")
    print(f"Model layers: {layers} | Input/Output Channels: {in_channels}/{out_channels}")
    print(f"Calibration samples: {calib_size}")
    print(f"Benchmark runs: {runs} (warmup: {warmup})")
    print("="*80 + "\n")

    all_results = []
    for res in resolutions:
        try:
            # *** UPDATED ARGS ***
            result = process_resolution(
                res, 
                in_channels=in_channels, 
                out_channels=out_channels, 
                layers=layers,
                calib_size=calib_size, 
                per_channel=per_channel, 
                warmup=warmup, 
                runs=runs
            )
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
    print("  - elitenet_<width>x<height>.onnx (FP32)")
    print("  - elitenet_<width>x<height>_int8.onnx (INT8)")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()