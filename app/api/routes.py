"""
API routes for model inference and management.
"""
from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import numpy as np
import torch

router = APIRouter()


class PredictionRequest(BaseModel):
    """Request model for predictions."""
    data: List[List[float]]
    model_type: str = "eeg"


class PredictionResponse(BaseModel):
    """Response model for predictions."""
    predictions: List[float]
    model_type: str
    inference_time_ms: float


class ModelInfo(BaseModel):
    """Model information."""
    name: str
    type: str
    input_shape: List[int]
    output_shape: List[int]
    parameters: int
    status: str


@router.get("/models", response_model=List[ModelInfo])
async def list_models():
    """
    List all available models.

    Returns:
        List of model information
    """
    # Placeholder - actual implementation would load from model registry
    models = [
        ModelInfo(
            name="tenn_eeg_v1",
            type="eeg_classification",
            input_shape=[1, 64, 256],
            output_shape=[1, 4],
            parameters=125000,
            status="ready"
        ),
        ModelInfo(
            name="vision_classifier_v1",
            type="image_classification",
            input_shape=[1, 3, 224, 224],
            output_shape=[1, 1000],
            parameters=2500000,
            status="ready"
        ),
        ModelInfo(
            name="segmentation_v1",
            type="semantic_segmentation",
            input_shape=[1, 3, 256, 256],
            output_shape=[1, 21, 256, 256],
            parameters=1800000,
            status="ready"
        ),
        ModelInfo(
            name="anomaly_detector_v1",
            type="anomaly_detection",
            input_shape=[1, 1, 256],
            output_shape=[1, 1, 256],
            parameters=350000,
            status="ready"
        )
    ]

    return models


@router.post("/predict/eeg", response_model=PredictionResponse)
async def predict_eeg(request: PredictionRequest):
    """
    Make predictions on EEG data.

    Args:
        request: Prediction request with EEG data

    Returns:
        Prediction results
    """
    try:
        import time

        start_time = time.time()

        # Convert input data to tensor
        data = np.array(request.data, dtype=np.float32)

        # Placeholder prediction - actual implementation would load model
        # and run inference
        predictions = [0.25, 0.35, 0.20, 0.20]  # Dummy class probabilities

        inference_time = (time.time() - start_time) * 1000

        return PredictionResponse(
            predictions=predictions,
            model_type="eeg_classification",
            inference_time_ms=inference_time
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predict/vision", response_model=PredictionResponse)
async def predict_vision(file: UploadFile = File(...)):
    """
    Make predictions on vision data.

    Args:
        file: Uploaded image file

    Returns:
        Prediction results
    """
    try:
        import time
        from PIL import Image
        import io

        start_time = time.time()

        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))

        # Placeholder prediction
        predictions = [0.8, 0.1, 0.05, 0.03, 0.02]

        inference_time = (time.time() - start_time) * 1000

        return PredictionResponse(
            predictions=predictions,
            model_type="image_classification",
            inference_time_ms=inference_time
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predict/anomaly", response_model=dict)
async def predict_anomaly(request: PredictionRequest):
    """
    Detect anomalies in time series data.

    Args:
        request: Prediction request with time series data

    Returns:
        Anomaly detection results
    """
    try:
        import time

        start_time = time.time()

        # Convert input data
        data = np.array(request.data, dtype=np.float32)

        # Placeholder anomaly detection
        anomaly_score = 0.15
        is_anomaly = anomaly_score > 0.5

        inference_time = (time.time() - start_time) * 1000

        return {
            "is_anomaly": is_anomaly,
            "anomaly_score": anomaly_score,
            "threshold": 0.5,
            "inference_time_ms": inference_time
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/model/{model_name:path}/info", response_model=ModelInfo)
async def get_model_info(model_name: str):
    """
    Get information about a specific model.

    Args:
        model_name: Name or path of the model

    Returns:
        Model information
    """
    # Placeholder - would query model registry
    models = await list_models()

    # Try to match by name or basename
    import os
    base_name = os.path.basename(model_name).replace('.onnx', '').replace('.pt', '')

    for model in models:
        if model.name == model_name or model.name == base_name:
            return model

    raise HTTPException(status_code=404, detail=f"Model {model_name} not found")


@router.post("/model/{model_name:path}/quantize")
async def quantize_model(model_name: str, target: str = "onnx"):
    """
    Quantize a model for edge deployment.

    Args:
        model_name: Name or path of the model to quantize
        target: Target hardware (onnx, akida, tvm)

    Returns:
        Quantization status
    """
    try:
        import os
        from pathlib import Path

        # Extract just the model name from path if full path provided
        base_name = os.path.basename(model_name)
        model_name_clean = base_name.replace('.onnx', '').replace('.pt', '')

        # Determine input model path
        if model_name.startswith('models/'):
            input_path = model_name
        elif os.path.exists(f"models/exported/{base_name}"):
            input_path = f"models/exported/{base_name}"
        elif os.path.exists(f"models/trained/{base_name}"):
            input_path = f"models/trained/{base_name}"
        else:
            # Try to find it
            input_path = f"models/exported/{model_name_clean}.onnx"

        # Output path
        output_path = f"models/quantized/{model_name_clean}_{target}_quantized.onnx"

        # Perform actual quantization based on target
        if target == "onnx":
            # Real ONNX quantization
            from app.utils.model_export import quantize_onnx_model

            if not os.path.exists(input_path):
                raise HTTPException(status_code=404, detail=f"Model not found: {input_path}")

            success = quantize_onnx_model(input_path, output_path, quantization_mode="dynamic")

            if not success:
                return {
                    "status": "error",
                    "message": "Quantization failed. Check logs for details.",
                    "model_name": model_name_clean,
                    "target": target
                }

            # Get file sizes for compression ratio
            original_size = os.path.getsize(input_path) / (1024 * 1024)  # MB
            quantized_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
            compression_ratio = original_size / quantized_size if quantized_size > 0 else 1.0

            return {
                "status": "success",
                "model_name": model_name_clean,
                "original_path": input_path,
                "target": target,
                "quantized_model_path": output_path,
                "original_size_mb": round(original_size, 2),
                "quantized_size_mb": round(quantized_size, 2),
                "compression_ratio": round(compression_ratio, 2),
                "message": "Model quantization completed successfully"
            }

        elif target == "akida":
            # Akida quantization requires SDK
            return {
                "status": "not_implemented",
                "message": "Akida quantization requires BrainChip Akida SDK. Please install the SDK and update app/utils/akida_quantization.py",
                "model_name": model_name_clean,
                "target": target,
                "documentation": "See INSTALL_NOTES.md for Akida SDK setup"
            }

        elif target == "tvm":
            # TVM quantization
            return {
                "status": "not_implemented",
                "message": "TVM quantization is optional. Uncomment apache-tvm in requirements.txt and update app/utils/tvm_compiler.py",
                "model_name": model_name_clean,
                "target": target
            }
        else:
            raise HTTPException(status_code=400, detail=f"Unknown target: {target}. Use 'onnx', 'akida', or 'tvm'")

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        raise HTTPException(status_code=500, detail=f"Quantization error: {str(e)}\n{traceback.format_exc()}")


@router.get("/benchmark/{model_name:path}")
async def benchmark_model(
    model_name: str,
    iterations: int = 100,
    device: str = "cpu"
):
    """
    Benchmark model performance on specified device.

    Args:
        model_name: Name or path of the model
        iterations: Number of benchmark iterations (default: 100)
        device: Device to use - 'cpu', 'cuda', 'mps' (default: 'cpu')

    Returns:
        Benchmark results including inference time, throughput, and memory usage
    """
    try:
        import os
        import torch
        import time
        import numpy as np
        from pathlib import Path

        # Validate device
        device_lower = device.lower()
        if device_lower not in ['cpu', 'cuda', 'mps']:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid device '{device}'. Use 'cpu', 'cuda', or 'mps'"
            )

        # Check device availability
        if device_lower == 'cuda' and not torch.cuda.is_available():
            return {
                "status": "error",
                "message": "CUDA not available. Install PyTorch with CUDA support.",
                "available_devices": ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])
            }

        if device_lower == 'mps' and not torch.backends.mps.is_available():
            return {
                "status": "error",
                "message": "MPS not available. Use macOS with Apple Silicon.",
                "available_devices": ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
            }

        # Extract model name
        base_name = os.path.basename(model_name)
        model_name_clean = base_name.replace('.onnx', '').replace('.pt', '')

        # Determine model path
        if model_name.startswith('models/'):
            model_path = model_name
        elif os.path.exists(f"models/exported/{base_name}"):
            model_path = f"models/exported/{base_name}"
        elif os.path.exists(f"models/trained/{base_name}"):
            model_path = f"models/trained/{base_name}"
        elif os.path.exists(f"models/quantized/{base_name}"):
            model_path = f"models/quantized/{base_name}"
        else:
            # Try to find any matching file
            model_path = None
            for dir_name in ['exported', 'trained', 'quantized']:
                search_dir = f"models/{dir_name}"
                if os.path.exists(search_dir):
                    for file in os.listdir(search_dir):
                        if model_name_clean in file and (file.endswith('.onnx') or file.endswith('.pt')):
                            model_path = f"{search_dir}/{file}"
                            break
                if model_path:
                    break

            if not model_path:
                raise HTTPException(
                    status_code=404,
                    detail=f"Model not found: {model_name}. Check models/exported/, models/trained/, or models/quantized/"
                )

        # Benchmark based on file type
        if model_path.endswith('.onnx'):
            # Benchmark ONNX model
            result = await benchmark_onnx_model(model_path, iterations, device_lower)
        elif model_path.endswith('.pt'):
            # Benchmark PyTorch model
            result = await benchmark_pytorch_model(model_path, model_name_clean, iterations, device_lower)
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported model format: {model_path}")

        result.update({
            "model_name": model_name_clean,
            "model_path": model_path,
            "iterations": iterations,
            "device": device_lower
        })

        return result

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        raise HTTPException(status_code=500, detail=f"Benchmark error: {str(e)}\n{traceback.format_exc()}")


async def benchmark_onnx_model(model_path: str, iterations: int, device: str) -> dict:
    """Benchmark ONNX model."""
    try:
        import onnxruntime as ort
        import time

        # Create inference session
        providers = ['CPUExecutionProvider']
        if device == 'cuda':
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']

        session = ort.InferenceSession(model_path, providers=providers)

        # Get input shape
        input_info = session.get_inputs()[0]
        input_shape = input_info.shape

        # Create dummy input
        dummy_input = np.random.randn(*input_shape).astype(np.float32)
        input_name = input_info.name

        # Warm up
        for _ in range(10):
            _ = session.run(None, {input_name: dummy_input})

        # Benchmark
        start_time = time.time()
        for _ in range(iterations):
            _ = session.run(None, {input_name: dummy_input})
        end_time = time.time()

        avg_time_ms = ((end_time - start_time) / iterations) * 1000
        throughput = iterations / (end_time - start_time)

        return {
            "status": "success",
            "framework": "ONNX Runtime",
            "avg_inference_time_ms": round(avg_time_ms, 3),
            "throughput_samples_per_sec": round(throughput, 2),
            "input_shape": input_shape,
            "providers": providers
        }

    except Exception as e:
        return {
            "status": "error",
            "message": f"ONNX benchmark failed: {str(e)}"
        }


async def benchmark_pytorch_model(model_path: str, model_type: str, iterations: int, device: str) -> dict:
    """Benchmark PyTorch model."""
    try:
        import torch
        import time

        # Load model architecture based on type
        from app.models.tenn_eeg import TENN_EEG
        from app.models.vision_model import ObjectClassifier
        from app.models.segmentation_model import SegmentationModel
        from app.models.anomaly_detection import AnomalyDetectionModel

        # Determine model type and create instance
        if 'eeg' in model_type.lower():
            model = TENN_EEG(num_channels=64, num_classes=4)
            input_shape = (1, 64, 256)
        elif 'vision' in model_type.lower() or 'classifier' in model_type.lower():
            model = ObjectClassifier(num_classes=1000, in_channels=3, base_channels=32)
            input_shape = (1, 3, 224, 224)
        elif 'segment' in model_type.lower():
            model = SegmentationModel(in_channels=3, num_classes=21)
            input_shape = (1, 3, 256, 256)
        elif 'anomaly' in model_type.lower():
            model = AnomalyDetectionModel(input_channels=1, sequence_length=256)
            input_shape = (1, 1, 256)
        else:
            # Default to EEG model
            model = TENN_EEG(num_channels=64, num_classes=4)
            input_shape = (1, 64, 256)

        # Load weights
        try:
            state_dict = torch.load(model_path, map_location='cpu')
            if 'model_state_dict' in state_dict:
                model.load_state_dict(state_dict['model_state_dict'])
            else:
                model.load_state_dict(state_dict)
        except Exception as e:
            # If loading fails, use untrained model for benchmark
            pass

        model.eval()
        model = model.to(device)

        # Create dummy input
        dummy_input = torch.randn(*input_shape).to(device)

        # Warm up
        with torch.no_grad():
            for _ in range(10):
                _ = model(dummy_input)

        if device == 'cuda':
            torch.cuda.synchronize()

        # Benchmark
        start_time = time.time()
        with torch.no_grad():
            for _ in range(iterations):
                _ = model(dummy_input)

        if device == 'cuda':
            torch.cuda.synchronize()

        end_time = time.time()

        avg_time_ms = ((end_time - start_time) / iterations) * 1000
        throughput = iterations / (end_time - start_time)

        # Get model size
        param_count = sum(p.numel() for p in model.parameters())

        return {
            "status": "success",
            "framework": "PyTorch",
            "avg_inference_time_ms": round(avg_time_ms, 3),
            "throughput_samples_per_sec": round(throughput, 2),
            "input_shape": list(input_shape),
            "parameters": param_count
        }

    except Exception as e:
        import traceback
        return {
            "status": "error",
            "message": f"PyTorch benchmark failed: {str(e)}\n{traceback.format_exc()}"
        }
