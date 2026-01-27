"""
Model export utilities for ONNX and TorchScript formats.
"""
import torch
import torch.onnx
from pathlib import Path
from typing import Tuple, Optional
from loguru import logger


def export_to_onnx(
    model: torch.nn.Module,
    input_shape: Tuple[int, ...],
    output_path: str,
    opset_version: int = 18,
    dynamic_axes: Optional[dict] = None
) -> bool:
    """
    Export PyTorch model to ONNX format.

    Args:
        model: PyTorch model to export
        input_shape: Shape of input tensor (including batch dimension)
        output_path: Path to save ONNX model
        opset_version: ONNX opset version
        dynamic_axes: Dictionary specifying dynamic axes for input/output

    Returns:
        True if export successful, False otherwise
    """
    try:
        model.eval()

        # Create dummy input
        dummy_input = torch.randn(*input_shape)

        # Create output directory if needed
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # Export to ONNX
        # Note: For PyTorch 2.1+, dynamic_shapes is preferred over dynamic_axes
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output']
        )

        logger.info(f"Model exported to ONNX: {output_path}")
        return True

    except Exception as e:
        logger.error(f"Failed to export model to ONNX: {e}")
        return False


def export_to_torchscript(
    model: torch.nn.Module,
    input_shape: Tuple[int, ...],
    output_path: str,
    use_trace: bool = True
) -> bool:
    """
    Export PyTorch model to TorchScript format.

    Args:
        model: PyTorch model to export
        input_shape: Shape of input tensor (including batch dimension)
        output_path: Path to save TorchScript model
        use_trace: If True, use tracing; if False, use scripting

    Returns:
        True if export successful, False otherwise
    """
    try:
        model.eval()

        # Create output directory if needed
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        if use_trace:
            # Create dummy input for tracing
            dummy_input = torch.randn(*input_shape)
            scripted_model = torch.jit.trace(model, dummy_input)
        else:
            # Use scripting (handles control flow better)
            scripted_model = torch.jit.script(model)

        # Save TorchScript model
        scripted_model.save(output_path)

        logger.info(f"Model exported to TorchScript: {output_path}")
        return True

    except Exception as e:
        logger.error(f"Failed to export model to TorchScript: {e}")
        return False


def verify_onnx_model(
    onnx_path: str,
    pytorch_model: torch.nn.Module,
    input_shape: Tuple[int, ...]
) -> bool:
    """
    Verify ONNX model produces same output as PyTorch model.

    Args:
        onnx_path: Path to ONNX model
        pytorch_model: Original PyTorch model
        input_shape: Shape of input tensor

    Returns:
        True if outputs match, False otherwise
    """
    try:
        import onnxruntime as ort
        import numpy as np

        # Create test input
        test_input = torch.randn(*input_shape)

        # Get PyTorch output
        pytorch_model.eval()
        with torch.no_grad():
            pytorch_output = pytorch_model(test_input)

        # Get ONNX output
        ort_session = ort.InferenceSession(onnx_path)
        ort_inputs = {ort_session.get_inputs()[0].name: test_input.numpy()}
        ort_output = ort_session.run(None, ort_inputs)[0]

        # Compare outputs
        if isinstance(pytorch_output, torch.Tensor):
            pytorch_output = pytorch_output.numpy()
        elif isinstance(pytorch_output, dict):
            # Handle multi-output models
            pytorch_output = list(pytorch_output.values())[0].numpy()

        # Check if outputs are close
        if np.allclose(pytorch_output, ort_output, rtol=1e-3, atol=1e-5):
            logger.info("ONNX model verification successful")
            return True
        else:
            logger.warning("ONNX model output differs from PyTorch model")
            return False

    except Exception as e:
        logger.error(f"Failed to verify ONNX model: {e}")
        return False


def quantize_onnx_model(
    model_path: str,
    output_path: str,
    quantization_mode: str = "dynamic"
) -> bool:
    """
    Quantize ONNX model for efficient inference.

    Args:
        model_path: Path to input ONNX model
        output_path: Path to save quantized model
        quantization_mode: Type of quantization ("dynamic" or "static")

    Returns:
        True if quantization successful, False otherwise
    """
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType

        # Create output directory if needed
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        if quantization_mode == "dynamic":
            quantize_dynamic(
                model_path,
                output_path,
                weight_type=QuantType.QUInt8
            )
        else:
            logger.warning("Static quantization not yet implemented")
            return False

        logger.info(f"Model quantized: {output_path}")
        return True

    except Exception as e:
        logger.error(f"Failed to quantize ONNX model: {e}")
        return False
