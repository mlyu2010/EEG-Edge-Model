"""
Unit tests for utility functions.
"""
import pytest
import torch
import tempfile
from pathlib import Path
from app.models.tenn_eeg import TENN_EEG
from app.utils.model_export import export_to_onnx, export_to_torchscript
from app.utils.training import Trainer, compute_classification_metrics


class TestModelExport:
    """Tests for model export utilities."""

    def test_export_to_onnx(self):
        """Test ONNX export."""
        model = TENN_EEG(num_channels=64, num_classes=4)
        input_shape = (1, 64, 256)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = str(Path(tmpdir) / "model.onnx")
            success = export_to_onnx(model, input_shape, output_path)

            assert success
            assert Path(output_path).exists()

    def test_export_to_torchscript(self):
        """Test TorchScript export."""
        model = TENN_EEG(num_channels=64, num_classes=4)
        input_shape = (1, 64, 256)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = str(Path(tmpdir) / "model.pt")
            success = export_to_torchscript(model, input_shape, output_path)

            assert success
            assert Path(output_path).exists()


class TestTrainer:
    """Tests for training utilities."""

    def test_trainer_initialization(self):
        """Test trainer initialization."""
        model = TENN_EEG(num_channels=64, num_classes=4)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters())

        trainer = Trainer(model, criterion, optimizer)

        assert trainer is not None
        assert trainer.device == "cpu"

    def test_classification_metrics(self):
        """Test classification metrics computation."""
        predictions = torch.tensor([[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]])
        targets = torch.tensor([1, 0, 1])

        metrics = compute_classification_metrics(predictions, targets)

        assert 'accuracy' in metrics
        assert 'correct' in metrics
        assert 'total' in metrics
        assert metrics['total'] == 3
        assert 0 <= metrics['accuracy'] <= 1
