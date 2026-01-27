#!/usr/bin/env python3
"""
Benchmark model performance across different platforms.

Usage:
    python scripts/benchmark.py --model-type tenn_eeg --iterations 1000
"""
import argparse
import torch
import time
import numpy as np
from pathlib import Path

from app.models.tenn_eeg import TENN_EEG
from app.models.vision_model import ObjectClassifier
from app.utils.tvm_compiler import TVMCompiler
from app.utils.device import get_device, print_device_info


MODEL_CONFIGS = {
    'tenn_eeg': {
        'class': TENN_EEG,
        'input_shape': (1, 64, 256),
        'kwargs': {'num_channels': 64, 'num_classes': 4}
    },
    'vision_classifier': {
        'class': ObjectClassifier,
        'input_shape': (1, 3, 224, 224),
        'kwargs': {'num_classes': 1000, 'in_channels': 3, 'base_channels': 32}
    }
}


def benchmark_pytorch(model, input_shape, iterations=1000, device='cpu'):
    """Benchmark PyTorch model."""
    model = model.to(device)
    model.eval()

    # Create dummy input
    dummy_input = torch.randn(*input_shape).to(device)

    # Warm up
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)

    # Benchmark
    if device == 'cuda':
        torch.cuda.synchronize()
    elif device == 'mps':
        torch.mps.synchronize()

    start_time = time.time()

    with torch.no_grad():
        for _ in range(iterations):
            _ = model(dummy_input)

    if device == 'cuda':
        torch.cuda.synchronize()
    elif device == 'mps':
        torch.mps.synchronize()

    end_time = time.time()

    avg_time = (end_time - start_time) / iterations

    return {
        'framework': 'PyTorch',
        'device': device,
        'avg_inference_time_ms': avg_time * 1000,
        'throughput': 1.0 / avg_time,
        'iterations': iterations
    }


def benchmark_tvm(model, input_shape, iterations=1000, target='llvm'):
    """Benchmark TVM compiled model."""
    print(f"Compiling model with TVM for target: {target}")

    compiler = TVMCompiler(model, input_shape)

    if not compiler.compile(target=target, opt_level=3):
        print("TVM compilation failed")
        return None

    metrics = compiler.benchmark(n_iterations=iterations)
    metrics['framework'] = 'TVM'

    return metrics


def benchmark_onnx(model_path, input_shape, iterations=1000):
    """Benchmark ONNX model."""
    try:
        import onnxruntime as ort

        session = ort.InferenceSession(model_path)

        # Create dummy input
        dummy_input = np.random.randn(*input_shape).astype(np.float32)
        input_name = session.get_inputs()[0].name

        # Warm up
        for _ in range(10):
            _ = session.run(None, {input_name: dummy_input})

        # Benchmark
        start_time = time.time()

        for _ in range(iterations):
            _ = session.run(None, {input_name: dummy_input})

        end_time = time.time()

        avg_time = (end_time - start_time) / iterations

        return {
            'framework': 'ONNX Runtime',
            'device': 'cpu',
            'avg_inference_time_ms': avg_time * 1000,
            'throughput': 1.0 / avg_time,
            'iterations': iterations
        }

    except ImportError:
        print("ONNX Runtime not installed")
        return None


def print_results(results):
    """Print benchmark results."""
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS")
    print("=" * 80)

    for result in results:
        if result is None:
            continue

        print(f"\nFramework: {result['framework']}")
        print(f"Device: {result.get('device', 'N/A')}")
        print(f"Average Inference Time: {result['avg_inference_time_ms']:.3f} ms")
        print(f"Throughput: {result['throughput']:.2f} samples/sec")
        print(f"Iterations: {result['iterations']}")

    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Benchmark model performance")
    parser.add_argument("--model-type", type=str, required=True,
                       choices=list(MODEL_CONFIGS.keys()),
                       help="Type of model to benchmark")
    parser.add_argument("--iterations", type=int, default=1000,
                       help="Number of benchmark iterations")
    parser.add_argument("--frameworks", type=str, nargs='+',
                       default=['pytorch'],
                       choices=['pytorch', 'tvm', 'onnx'],
                       help="Frameworks to benchmark")
    parser.add_argument("--device", type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda', 'mps'],
                       help="Device for PyTorch benchmarking")

    args = parser.parse_args()

    # Print device information
    print_device_info()

    # Get the best available device
    device = get_device(args.device)

    # Load model configuration
    config = MODEL_CONFIGS[args.model_type]
    model = config['class'](**config['kwargs'])
    model.eval()
    input_shape = config['input_shape']

    print(f"Benchmarking {args.model_type} model")
    print(f"Input shape: {input_shape}")
    print(f"Iterations: {args.iterations}")

    results = []

    # PyTorch benchmark
    if 'pytorch' in args.frameworks:
        print("\nBenchmarking PyTorch...")
        result = benchmark_pytorch(model, input_shape, args.iterations, device)
        results.append(result)

    # TVM benchmark
    if 'tvm' in args.frameworks:
        print("\nBenchmarking TVM...")
        result = benchmark_tvm(model, input_shape, args.iterations)
        results.append(result)

    # ONNX benchmark
    if 'onnx' in args.frameworks:
        # Need to export to ONNX first
        from app.utils.model_export import export_to_onnx

        onnx_path = "/tmp/benchmark_model.onnx"
        export_to_onnx(model, input_shape, onnx_path)

        print("\nBenchmarking ONNX Runtime...")
        result = benchmark_onnx(onnx_path, input_shape, args.iterations)
        results.append(result)

    # Print results
    print_results(results)


if __name__ == "__main__":
    main()
