"""ONNX export, 512-image source-train static INT8 QDQ PTQ, benchmarking."""
from pathlib import Path
import time
import numpy as np
import torch
from mangonet.models.mango_net import LogitsOnly
from mangonet.data.dataset import MangoImageDataset
from mangonet.utils.reproducibility import save_json, count_parameters


def export_onnx(model, destination, image_size=224, opset=17):
    """Export logits-only model, with dynamic batch and fixed spatial dimensions."""
    try:
        import onnx  # noqa: F401
    except ImportError as exc:
        raise ImportError("Install ONNX export extras: pip install -e '.[export]'") from exc
    dest = Path(destination)
    dest.parent.mkdir(parents=True, exist_ok=True)
    model.cpu().eval()
    wrapped = LogitsOnly(model).eval()
    sample = torch.zeros(1, 3, image_size, image_size, dtype=torch.float32)
    # Disable fused fastpath, which is not portable to older ONNX exporters.
    was_fastpath = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    try:
        torch.onnx.export(wrapped, (sample,), str(dest), opset_version=opset,
                          input_names=['images'], output_names=['logits'],
                          dynamic_axes={'images': {0: 'batch'}, 'logits': {0: 'batch'}},
                          do_constant_folding=True, dynamo=False)
    finally:
        torch.backends.mha.set_fastpath_enabled(was_fastpath)
    import onnx
    onnx.checker.check_model(str(dest))
    return dest


def quantize_onnx_static(onnx_input, onnx_output, training_dataset, limit=512):
    """Source training dataset only; per-channel signed INT8 weights/uint8 activations."""
    try:
        import onnxruntime as ort
        from onnxruntime.quantization import (quantize_static, CalibrationDataReader,
                                              QuantType, QuantFormat, CalibrationMethod)
    except ImportError as exc:
        raise ImportError("Install ONNX Runtime: pip install -e '.[export]'") from exc
    if not isinstance(training_dataset, MangoImageDataset) or training_dataset.pairs:
        raise TypeError('Expected MangoImageDataset of non-augmented source train images')
    if len(training_dataset) < 1:
        raise ValueError('Cannot calibrate on an empty source training dataset')

    class Reader(CalibrationDataReader):
        def __init__(self):
            self.index = 0
            self.last = min(limit, len(training_dataset))

        def get_next(self):
            if self.index >= self.last:
                return None
            tensor = training_dataset[self.index]['image']
            self.index += 1
            return {'images': tensor.unsqueeze(0).numpy().astype(np.float32)}

        def rewind(self):
            self.index = 0

    output = Path(onnx_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Leave LayerNorm, Softmax, GELU and unsupported attention operators in FP32.
    quantize_static(str(onnx_input), str(output), Reader(),
                    quant_format=QuantFormat.QDQ, activation_type=QuantType.QUInt8,
                    weight_type=QuantType.QInt8, per_channel=True,
                    op_types_to_quantize=['Conv', 'MatMul', 'Gemm'],
                    calibrate_method=CalibrationMethod.MinMax)
    import onnx
    ops = {}
    for op in onnx.load(str(output)).graph.node:
        ops[op.op_type] = ops.get(op.op_type, 0) + 1
    summary = {'calibration_images': min(limit, len(training_dataset)),
               'source_train_only': True,
               'fp32_onnx_bytes': Path(onnx_input).stat().st_size,
               'int8_onnx_bytes': output.stat().st_size,
               'quantized_operators_qdq_count': ops.get('QuantizeLinear', 0),
               'dequantize_operator_count': ops.get('DequantizeLinear', 0),
               'operator_counts': ops,
               'note': 'Unsupported nodes intentionally remain floating point; inspect operator counts.'}
    save_json(summary, output.parent / 'quantization_summary.json')
    return output, summary


def benchmark_torch(model, image_size=224, device='cpu', warmup=50, repeats=500, threads=4):
    old_threads = torch.get_num_threads()
    if device == 'cpu':
        torch.set_num_threads(threads)
    try:
        model.to(device).eval()
        if device == 'cuda':
            torch.cuda.reset_peak_memory_stats()
        tensor = torch.randn(1, 3, image_size, image_size, device=device)
        with torch.inference_mode():
            for _ in range(warmup):
                model(tensor)
            if device == 'cuda':
                torch.cuda.synchronize()
            measurements = []
            for _ in range(repeats):
                t0 = time.perf_counter()
                model(tensor)
                if device == 'cuda':
                    torch.cuda.synchronize()
                measurements.append((time.perf_counter() - t0) * 1000)
        median = float(np.median(measurements))
        return {'median_ms': median, 'fps_batch1': float(1000/median), 'repeats': repeats,
                'warmups': warmup, 'device': device, 'parameters': count_parameters(model),
                'peak_cuda_allocated_bytes': (int(torch.cuda.max_memory_allocated())
                                               if device == 'cuda' else None),
                'memory_note': ('Peak device allocated memory via CUDA allocator' if device == 'cuda'
                                else 'CPU peak allocation not instrumented by this benchmark')}
    finally:
        if device == 'cpu':
            torch.set_num_threads(old_threads)


def benchmark_onnx(path, image_size=224, warmup=50, repeats=500, threads=4):
    import onnxruntime as ort
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    sess = ort.InferenceSession(str(path), sess_options=options, providers=['CPUExecutionProvider'])
    x = np.random.default_rng(0).normal(size=(1,3,image_size,image_size)).astype('float32')
    for _ in range(warmup):
        sess.run(None, {'images': x})
    measurements = []
    for _ in range(repeats):
        before = time.perf_counter()
        sess.run(None, {'images': x})
        measurements.append((time.perf_counter() - before)*1000)
    median = float(np.median(measurements))
    return {'median_ms': median, 'fps_batch1': float(1000/median),
            'warmups': warmup, 'repeats': repeats,
            'backend': 'ONNXRuntime CPU 4 threads' if threads == 4 else f'ONNXRuntime CPU {threads} threads',
            'model_bytes': Path(path).stat().st_size}


def parity_check(model, onnx_path, image_size=224):
    """Maximum absolute logit discrepancy between ONNX and original FP32."""
    import onnxruntime as ort
    options = ort.SessionOptions(); options.intra_op_num_threads = 4
    sess = ort.InferenceSession(str(onnx_path), sess_options=options, providers=['CPUExecutionProvider'])
    x = np.random.default_rng(42).normal(size=(2, 3, image_size, image_size)).astype('float32')
    model.cpu().eval()
    with torch.inference_mode():
        expected = model(torch.from_numpy(x))['logits'].numpy()
    actual = sess.run(None, {'images': x})[0]
    return float(np.max(np.abs(actual - expected)))


def estimate_macs(model, image_size=224):
    """Try fvcore first; otherwise PyTorch profiler estimate (may omit fused SDPA)."""
    model = model.cpu().eval()
    sample = torch.randn(1, 3, image_size, image_size)
    wrapped = LogitsOnly(model).eval()
    try:
        from fvcore.nn import FlopCountAnalysis
        analysis = FlopCountAnalysis(wrapped, sample)
        analysis.unsupported_ops_warnings(False)
        analysis.uncalled_modules_warnings(False)
        count = int(analysis.total())
        return {'estimated_macs': count, 'method': 'fvcore FlopCountAnalysis',
                'note': 'fvcore counts multiply-add as one MAC. Unsupported operators may be excluded.'}
    except (ImportError, Exception) as error:
        try:
            from torch.profiler import profile, ProfilerActivity
            with profile(activities=[ProfilerActivity.CPU], with_flops=True) as prof:
                with torch.inference_mode():
                    wrapped(sample)
            flops = sum(int(item.flops or 0) for item in prof.key_averages())
            return {'estimated_macs': flops // 2,
                    'method': 'PyTorch profiler FLOPs/2 approximation',
                    'note': 'Lower-bound-like estimate: fused attention and other unprofiled operators may be omitted.'}
        except Exception as second:
            return {'estimated_macs': None, 'method': 'unavailable',
                    'reason': str(second)}
