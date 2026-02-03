# Benchmark Guide

## Overview

The benchmark endpoint now performs **real performance measurements** on CPU, GPU (CUDA), and MPS (Apple Silicon) devices!

## Quick Start

```bash
# Benchmark on CPU
curl "http://localhost:8000/api/v1/benchmark/tenn_eeg.onnx?device=cpu&iterations=100"

# Benchmark on CUDA GPU (if available)
curl "http://localhost:8000/api/v1/benchmark/tenn_eeg.onnx?device=cuda&iterations=100"

# Benchmark on Apple Silicon MPS (if available)
curl "http://localhost:8000/api/v1/benchmark/best_model.pt?device=mps&iterations=100"
```

## Endpoint Details

### URL
```
GET /api/v1/benchmark/{model_name}
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_name` | string (path) | required | Model name or full path |
| `iterations` | integer | 100 | Number of benchmark iterations |
| `device` | string | "cpu" | Device: `cpu`, `cuda`, or `mps` |

### Supported Devices

| Device | Description | Requirements |
|--------|-------------|--------------|
| `cpu` | ✅ CPU inference | Always available |
| `cuda` | 🎮 NVIDIA GPU | Requires CUDA-enabled PyTorch |
| `mps` | 🍎 Apple Silicon | Requires macOS with M1/M2/M3 |

## Usage Examples

### 1. CPU Benchmark (ONNX Model)

```bash
curl "http://localhost:8000/api/v1/benchmark/tenn_eeg.onnx?device=cpu&iterations=100"
```

**Response:**
```json
{
  "status": "success",
  "framework": "ONNX Runtime",
  "avg_inference_time_ms": 6.149,
  "throughput_samples_per_sec": 162.63,
  "input_shape": [1, 64, 256],
  "providers": ["CPUExecutionProvider"],
  "model_name": "tenn_eeg",
  "model_path": "models/exported/tenn_eeg.onnx",
  "iterations": 100,
  "device": "cpu"
}
```

### 2. CPU Benchmark (PyTorch Model)

```bash
curl "http://localhost:8000/api/v1/benchmark/best_model.pt?device=cpu&iterations=50"
```

**Response:**
```json
{
  "status": "success",
  "framework": "PyTorch",
  "avg_inference_time_ms": 64.319,
  "throughput_samples_per_sec": 15.55,
  "input_shape": [1, 64, 256],
  "parameters": 148644,
  "model_name": "best_model",
  "model_path": "models/trained/best_model.pt",
  "iterations": 50,
  "device": "cpu"
}
```

### 3. CUDA GPU Benchmark

```bash
# Requires CUDA-enabled PyTorch
curl "http://localhost:8000/api/v1/benchmark/tenn_eeg_final.pt?device=cuda&iterations=1000"
```

**If CUDA not available:**
```json
{
  "status": "error",
  "message": "CUDA not available. Install PyTorch with CUDA support.",
  "available_devices": ["cpu", "mps"]
}
```

### 4. Apple Silicon MPS Benchmark

```bash
# Only works on macOS with Apple Silicon (outside Docker)
curl "http://localhost:8000/api/v1/benchmark/best_model.pt?device=mps&iterations=100"
```

**Inside Docker:**
```json
{
  "status": "error",
  "message": "MPS not available. Use macOS with Apple Silicon.",
  "available_devices": ["cpu"]
}
```

### 5. Compare Multiple Devices

```bash
# Benchmark on different devices
for device in cpu cuda mps; do
  echo "=== Device: $device ==="
  curl -s "http://localhost:8000/api/v1/benchmark/tenn_eeg.onnx?device=$device&iterations=100" \
    | python3 -m json.tool
  echo
done
```

## Model Support

### ONNX Models (.onnx)
- ✅ CPU execution (always available)
- ✅ CUDA execution (with CUDAExecutionProvider)
- ⚠️ MPS not supported by ONNX Runtime

### PyTorch Models (.pt)
- ✅ CPU execution
- ✅ CUDA execution (if PyTorch built with CUDA)
- ✅ MPS execution (if PyTorch built with MPS support)

### Automatically Detected Models

The endpoint searches in order:
1. `models/exported/{model_name}`
2. `models/trained/{model_name}`
3. `models/quantized/{model_name}`

## Performance Metrics Explained

### avg_inference_time_ms
Average time per inference in milliseconds. **Lower is better**.

```
6.149 ms  = ~162 inferences per second
64.319 ms = ~15 inferences per second
```

### throughput_samples_per_sec
Number of inferences per second. **Higher is better**.

### Framework-Specific Metrics

**ONNX Runtime:**
- `providers`: Execution providers used (CPU, CUDA, etc.)
- `input_shape`: Model input dimensions

**PyTorch:**
- `parameters`: Total number of model parameters
- `input_shape`: Model input dimensions

## Performance Comparison

### Expected Performance (TENN EEG Model)

| Device | Framework | Avg Time | Throughput | Notes |
|--------|-----------|----------|------------|-------|
| CPU | ONNX | ~6ms | ~160/s | Optimized for CPU |
| CPU | PyTorch | ~60ms | ~15/s | Less optimized |
| CUDA | PyTorch | ~2ms | ~500/s | Requires GPU |
| MPS | PyTorch | ~3ms | ~330/s | Apple Silicon |

*Note: Actual performance varies by hardware*

## Advanced Usage

### Python Client

```python
import requests

BASE_URL = "http://localhost:8000/api/v1"

def benchmark_model(model_name, device='cpu', iterations=100):
    """Benchmark a model on specified device."""
    url = f"{BASE_URL}/benchmark/{model_name}"
    params = {
        'device': device,
        'iterations': iterations
    }

    response = requests.get(url, params=params)
    return response.json()

# Benchmark on different devices
models = ['tenn_eeg.onnx', 'best_model.pt']
devices = ['cpu', 'cuda', 'mps']

for model in models:
    for device in devices:
        result = benchmark_model(model, device, iterations=100)
        if result.get('status') == 'success':
            print(f"{model} on {device}: {result['avg_inference_time_ms']:.2f}ms")
        else:
            print(f"{model} on {device}: {result.get('message', 'Error')}")
```

### Multi-Device Comparison Script

```bash
#!/bin/bash
# compare_devices.sh

MODEL="tenn_eeg.onnx"
ITERATIONS=1000

echo "Benchmarking $MODEL with $ITERATIONS iterations"
echo "================================================"

for device in cpu cuda mps; do
    echo -e "\n[$device]"
    curl -s "http://localhost:8000/api/v1/benchmark/$MODEL?device=$device&iterations=$ITERATIONS" \
        | jq -r 'if .status == "success" then
            "✓ Time: \(.avg_inference_time_ms)ms, Throughput: \(.throughput_samples_per_sec)/s"
        else
            "✗ \(.message)"
        end'
done
```

## Docker Considerations

### Inside Docker Container
- ✅ CPU always works
- ❌ MPS not available (requires macOS host)
- ⚠️ CUDA requires `--gpus all` flag

### Enable GPU in Docker

```yaml
# docker-compose.yml
services:
  edge-models:
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
```

Then:
```bash
docker-compose up -d
curl "http://localhost:8000/api/v1/benchmark/model.pt?device=cuda"
```

### For MPS (Apple Silicon)
Run outside Docker directly on macOS:

```bash
# Install dependencies locally
pip install -r requirements.txt

# Run server
uvicorn app.main:app --reload

# Benchmark with MPS
curl "http://localhost:8000/api/v1/benchmark/model.pt?device=mps"
```

## Troubleshooting

### Error: "CUDA not available"

**Solution:**
1. Check PyTorch CUDA support:
   ```bash
   docker exec edge-models-dev python -c "import torch; print(torch.cuda.is_available())"
   ```

2. Install CUDA-enabled PyTorch:
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```

### Error: "MPS not available"

**Solution:**
- MPS only works on macOS with Apple Silicon (M1/M2/M3)
- Run outside Docker for MPS support
- Alternative: Use `device=cpu` in Docker

### Error: "Model not found"

**Solution:**
```bash
# List available models
ls -la models/exported/
ls -la models/trained/
ls -la models/quantized/

# Use correct model name
curl "http://localhost:8000/api/v1/benchmark/tenn_eeg.onnx?device=cpu"
```

### Slow Performance

**Tips:**
1. Increase iterations for more accurate results: `iterations=1000`
2. Use ONNX for CPU inference (faster than PyTorch)
3. Use quantized models for better CPU performance
4. Enable GPU if available

## API Integration

### Swagger UI

Interactive testing at: http://localhost:8000/docs

1. Navigate to `/benchmark/{model_name}`
2. Click "Try it out"
3. Enter:
   - `model_name`: `tenn_eeg.onnx`
   - `iterations`: `100`
   - `device`: `cpu`
4. Click "Execute"

### JavaScript/TypeScript

```javascript
async function benchmarkModel(modelName, device = 'cpu', iterations = 100) {
  const url = `http://localhost:8000/api/v1/benchmark/${modelName}`;
  const params = new URLSearchParams({ device, iterations });

  const response = await fetch(`${url}?${params}`);
  const data = await response.json();

  if (data.status === 'success') {
    console.log(`${modelName} on ${device}:`);
    console.log(`  Avg time: ${data.avg_inference_time_ms}ms`);
    console.log(`  Throughput: ${data.throughput_samples_per_sec}/s`);
  } else {
    console.error(`Error: ${data.message}`);
  }

  return data;
}

// Usage
await benchmarkModel('tenn_eeg.onnx', 'cpu', 100);
await benchmarkModel('best_model.pt', 'cuda', 1000);
```

## Summary

✅ **What Changed:**
- Added real device selection: `cpu`, `cuda`, `mps`
- Actual performance measurements (not placeholders)
- Support for both ONNX and PyTorch models
- Device availability checking
- Comprehensive error messages

✅ **How to Use:**
```bash
# Basic CPU benchmark
curl "http://localhost:8000/api/v1/benchmark/tenn_eeg.onnx?device=cpu&iterations=100"

# Compare devices
curl "http://localhost:8000/api/v1/benchmark/model.pt?device=cuda&iterations=1000"
```

📊 **Performance Insights:**
- ONNX on CPU: ~10x faster than PyTorch
- Quantized models: ~4x smaller, similar speed
- GPU: ~10-50x faster than CPU (model dependent)

For more examples, see `API_USAGE.md` and interactive docs at http://localhost:8000/docs
