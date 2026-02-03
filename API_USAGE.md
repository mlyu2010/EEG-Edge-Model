# API Usage Guide

## Base URL
```
http://localhost:8000
```

## Authentication
Currently, no authentication is required. This can be added for production use.

## Endpoints

### Health & Status

#### GET `/`
Root endpoint returning application info.

**Example:**
```bash
curl http://localhost:8000/
```

**Response:**
```json
{
  "name": "EEG-Edge-Model",
  "version": "1.0.0",
  "status": "running",
  "docs": "/docs"
}
```

#### GET `/health`
Health check endpoint.

**Example:**
```bash
curl http://localhost:8000/health
```

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0"
}
```

---

### Model Management

#### GET `/api/v1/models`
List all available models.

**Example:**
```bash
curl http://localhost:8000/api/v1/models
```

**Response:**
```json
[
  {
    "name": "tenn_eeg_v1",
    "type": "eeg_classification",
    "input_shape": [1, 64, 256],
    "output_shape": [1, 4],
    "parameters": 125000,
    "status": "ready"
  },
  ...
]
```

#### GET `/api/v1/model/{model_name}/info`
Get detailed information about a specific model.

**Path Parameter:**
- `model_name` - Can be either:
  - Simple name: `tenn_eeg_v1`
  - Full path: `models/exported/tenn_eeg.onnx`

**Examples:**
```bash
# Simple name
curl http://localhost:8000/api/v1/model/tenn_eeg_v1/info

# Full path
curl http://localhost:8000/api/v1/model/models/exported/tenn_eeg.onnx/info
```

**Response:**
```json
{
  "name": "tenn_eeg_v1",
  "type": "eeg_classification",
  "input_shape": [1, 64, 256],
  "output_shape": [1, 4],
  "parameters": 125000,
  "status": "ready"
}
```

---

### Model Operations

#### POST `/api/v1/model/{model_name}/quantize`
Quantize a model for edge deployment.

**Path Parameter:**
- `model_name` - Model name or full path

**Query Parameters:**
- `target` - Target hardware: `akida`, `tvm`, or `onnx` (default: `akida`)

**Examples:**
```bash
# Quantize for Akida
curl -X POST "http://localhost:8000/api/v1/model/tenn_eeg_v1/quantize?target=akida"

# Quantize ONNX model for ONNX Runtime
curl -X POST "http://localhost:8000/api/v1/model/models/exported/tenn_eeg.onnx/quantize?target=onnx"

# Quantize for TVM
curl -X POST "http://localhost:8000/api/v1/model/vision_classifier_v1/quantize?target=tvm"
```

**Response:**
```json
{
  "status": "success",
  "model_name": "tenn_eeg",
  "original_path": "models/exported/tenn_eeg.onnx",
  "target": "onnx",
  "quantized_model_path": "models/quantized/tenn_eeg_onnx.bin",
  "compression_ratio": 4.0,
  "message": "Model quantization completed successfully"
}
```

#### GET `/api/v1/benchmark/{model_name}`
Benchmark model performance.

**Path Parameter:**
- `model_name` - Model name or full path

**Query Parameters:**
- `iterations` - Number of iterations (default: 100)

**Examples:**
```bash
# Simple benchmark
curl "http://localhost:8000/api/v1/benchmark/tenn_eeg_v1?iterations=100"

# Benchmark with path
curl "http://localhost:8000/api/v1/benchmark/models/trained/tenn_eeg_final.pt?iterations=1000"
```

**Response:**
```json
{
  "model_name": "tenn_eeg",
  "original_path": "models/exported/tenn_eeg.onnx",
  "iterations": 100,
  "avg_inference_time_ms": 2.5,
  "throughput_samples_per_sec": 400,
  "memory_usage_mb": 125,
  "device": "cpu"
}
```

---

### Inference Endpoints

#### POST `/api/v1/predict/eeg`
Perform inference on EEG data.

**Request Body:**
```json
{
  "data": [[0.1, 0.2, ...], ...],  // 64 channels x 256 timesteps
  "model_type": "eeg"
}
```

**Example:**
```bash
curl -X POST http://localhost:8000/api/v1/predict/eeg \
  -H "Content-Type: application/json" \
  -d '{
    "data": [[0.1, 0.2], [0.3, 0.4]],
    "model_type": "eeg"
  }'
```

**Response:**
```json
{
  "predictions": [0.25, 0.35, 0.20, 0.20],
  "model_type": "eeg_classification",
  "inference_time_ms": 2.5
}
```

#### POST `/api/v1/predict/vision`
Perform inference on image data.

**Request:** Multipart form with image file

**Example:**
```bash
curl -X POST http://localhost:8000/api/v1/predict/vision \
  -F "file=@image.jpg"
```

**Response:**
```json
{
  "predictions": [0.8, 0.1, 0.05, 0.03, 0.02],
  "model_type": "image_classification",
  "inference_time_ms": 15.3
}
```

#### POST `/api/v1/predict/anomaly`
Detect anomalies in time series data.

**Request Body:**
```json
{
  "data": [[0.1, 0.2, ...]],  // Time series data
  "model_type": "anomaly"
}
```

**Example:**
```bash
curl -X POST http://localhost:8000/api/v1/predict/anomaly \
  -H "Content-Type: application/json" \
  -d '{
    "data": [[0.1, 0.2, 0.3, 0.4]],
    "model_type": "anomaly"
  }'
```

**Response:**
```json
{
  "is_anomaly": false,
  "anomaly_score": 0.15,
  "threshold": 0.5,
  "inference_time_ms": 3.2
}
```

---

## Interactive Documentation

Visit these URLs for interactive API documentation:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

These provide:
- Interactive testing interface
- Detailed schema information
- Request/response examples
- Auto-generated from code

---

## Error Responses

All endpoints return standard HTTP error codes:

### 404 Not Found
```json
{
  "detail": "Model tenn_eeg_v1 not found"
}
```

### 500 Internal Server Error
```json
{
  "detail": "Error message describing what went wrong"
}
```

---

## Python Client Example

```python
import requests

BASE_URL = "http://localhost:8000"

# List models
response = requests.get(f"{BASE_URL}/api/v1/models")
models = response.json()
print(f"Available models: {len(models)}")

# EEG prediction
data = {
    "data": [[0.1] * 256 for _ in range(64)],
    "model_type": "eeg"
}
response = requests.post(f"{BASE_URL}/api/v1/predict/eeg", json=data)
result = response.json()
print(f"Predictions: {result['predictions']}")
print(f"Inference time: {result['inference_time_ms']}ms")

# Quantize model
response = requests.post(
    f"{BASE_URL}/api/v1/model/tenn_eeg_v1/quantize",
    params={"target": "onnx"}
)
result = response.json()
print(f"Quantization status: {result['status']}")
print(f"Output path: {result['quantized_model_path']}")

# Benchmark
response = requests.get(
    f"{BASE_URL}/api/v1/benchmark/tenn_eeg_v1",
    params={"iterations": 100}
)
result = response.json()
print(f"Avg inference time: {result['avg_inference_time_ms']}ms")
```

---

## JavaScript/TypeScript Client Example

```javascript
const BASE_URL = "http://localhost:8000";

// List models
const models = await fetch(`${BASE_URL}/api/v1/models`).then(r => r.json());
console.log(`Available models: ${models.length}`);

// EEG prediction
const eegData = {
  data: Array(64).fill(Array(256).fill(0.1)),
  model_type: "eeg"
};

const prediction = await fetch(`${BASE_URL}/api/v1/predict/eeg`, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(eegData)
}).then(r => r.json());

console.log(`Predictions: ${prediction.predictions}`);
console.log(`Inference time: ${prediction.inference_time_ms}ms`);

// Quantize model
const quantize = await fetch(
  `${BASE_URL}/api/v1/model/tenn_eeg_v1/quantize?target=onnx`,
  { method: "POST" }
).then(r => r.json());

console.log(`Status: ${quantize.status}`);
```

---

## Notes

1. **Model Names**: Endpoints accept both simple names (`tenn_eeg_v1`) and full paths (`models/exported/tenn_eeg.onnx`)

2. **Placeholders**: Current implementation uses placeholder data. In production:
   - Load actual models
   - Perform real inference
   - Implement actual quantization

3. **CORS**: Enabled for all origins in development. Configure appropriately for production.

4. **Rate Limiting**: Not implemented. Consider adding for production use.

5. **Authentication**: Not implemented. Add JWT or API key authentication for production.

---

## See Also

- Full API documentation at http://localhost:8000/docs
- Project structure in `PROJECT_STRUCTURE.md`
- Setup guide in `SETUP_GUIDE.md`
