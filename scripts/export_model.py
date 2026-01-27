#!/usr/bin/env python3
"""
Export trained models to ONNX, TorchScript, and Akida formats.

Usage:
    python scripts/export_model.py --model-path models/trained/model.pt --format onnx
"""
import argparse
import torch
from pathlib import Path

from app.models.tenn_eeg import TENN_EEG
from app.models.vision_model import ObjectClassifier
from app.models.segmentation_model import SegmentationModel
from app.models.anomaly_detection import AnomalyDetectionModel
from app.utils.model_export import export_to_onnx, export_to_torchscript, verify_onnx_model
from app.utils.akida_quantization import AkidaQuantizer


MODEL_REGISTRY = {
    'tenn_eeg': {
        'class': TENN_EEG,
        'input_shape': (1, 64, 256),
        'kwargs': {'num_channels': 64, 'num_classes': 4}
    },
    'vision_classifier': {
        'class': ObjectClassifier,
        'input_shape': (1, 3, 224, 224),
        'kwargs': {'num_classes': 1000, 'in_channels': 3}
    },
    'segmentation': {
        'class': SegmentationModel,
        'input_shape': (1, 3, 256, 256),
        'kwargs': {'in_channels': 3, 'num_classes': 21}
    },
    'anomaly_detection': {
        'class': AnomalyDetectionModel,
        'input_shape': (1, 1, 256),
        'kwargs': {'input_channels': 1, 'sequence_length': 256}
    }
}


def load_model(model_type: str, model_path: str = None):
    """Load model from registry."""
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model type: {model_type}")

    config = MODEL_REGISTRY[model_type]
    model = config['class'](**config['kwargs'])

    if model_path and Path(model_path).exists():
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        print(f"Loaded model weights from {model_path}")

    model.eval()
    return model, config['input_shape']


def main():
    parser = argparse.ArgumentParser(description="Export PyTorch models")
    parser.add_argument("--model-type", type=str, required=True,
                       choices=list(MODEL_REGISTRY.keys()),
                       help="Type of model to export")
    parser.add_argument("--model-path", type=str, default=None,
                       help="Path to trained model weights (optional)")
    parser.add_argument("--format", type=str, required=True,
                       choices=['onnx', 'torchscript', 'akida', 'all'],
                       help="Export format")
    parser.add_argument("--output-dir", type=str, default="./models/exported",
                       help="Output directory for exported models")
    parser.add_argument("--verify", action='store_true',
                       help="Verify exported models")

    args = parser.parse_args()

    # Load model
    print(f"Loading {args.model_type} model...")
    model, input_shape = load_model(args.model_type, args.model_path)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_name = args.model_type

    # Export to requested format(s)
    if args.format == 'onnx' or args.format == 'all':
        print("\nExporting to ONNX...")
        onnx_path = str(output_dir / f"{model_name}.onnx")
        success = export_to_onnx(model, input_shape, onnx_path)

        if success and args.verify:
            print("Verifying ONNX model...")
            verify_onnx_model(onnx_path, model, input_shape)

    if args.format == 'torchscript' or args.format == 'all':
        print("\nExporting to TorchScript...")
        ts_path = str(output_dir / f"{model_name}_torchscript.pt")
        export_to_torchscript(model, input_shape, ts_path)

    if args.format == 'akida' or args.format == 'all':
        print("\nExporting for Akida hardware...")
        quantizer = AkidaQuantizer(model, input_shape)

        # First export to ONNX
        onnx_path = str(output_dir / f"{model_name}_akida.onnx")
        quantizer.export_to_onnx(onnx_path)

        # Then quantize for Akida
        akida_path = str(output_dir / f"{model_name}_akida.fbz")
        quantizer.quantize(akida_path)

    print(f"\n✓ Export complete! Models saved to {output_dir.absolute()}")


if __name__ == "__main__":
    main()
