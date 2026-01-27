"""
Unit tests for model architectures.
"""
import pytest
import torch
from app.models.tenn_eeg import TENN_EEG, TemporalBlock, StateSpaceLayer
from app.models.vision_model import ObjectClassifier, ObjectDetector
from app.models.segmentation_model import SegmentationModel
from app.models.anomaly_detection import AnomalyDetectionModel


class TestTENNEEG:
    """Tests for TENN EEG model."""

    def test_model_initialization(self):
        """Test model can be initialized."""
        model = TENN_EEG(
            num_channels=64,
            num_classes=4,
            temporal_channels=[32, 64, 128],
            state_dim=64
        )
        assert model is not None
        assert model.num_classes == 4

    def test_forward_pass(self):
        """Test forward pass with dummy data."""
        model = TENN_EEG(num_channels=64, num_classes=4)
        batch_size = 2
        seq_length = 256

        x = torch.randn(batch_size, 64, seq_length)
        output = model(x)

        assert output.shape == (batch_size, 4)

    def test_feature_extraction(self):
        """Test feature extraction."""
        model = TENN_EEG(num_channels=64, num_classes=4, state_dim=64)
        x = torch.randn(2, 64, 256)

        features = model.extract_features(x)

        assert features.shape == (2, 64)

    def test_temporal_block(self):
        """Test temporal block."""
        block = TemporalBlock(in_channels=32, out_channels=64, kernel_size=3)
        x = torch.randn(2, 32, 128)

        output = block(x)

        assert output.shape[0] == 2
        assert output.shape[1] == 64

    def test_state_space_layer(self):
        """Test state space layer."""
        layer = StateSpaceLayer(input_dim=32, state_dim=64, output_dim=64)
        x = torch.randn(2, 100, 32)

        output, state = layer(x)

        assert output.shape == (2, 100, 64)
        assert state.shape == (2, 64)


class TestVisionModel:
    """Tests for vision models."""

    def test_classifier_initialization(self):
        """Test classifier initialization."""
        model = ObjectClassifier(num_classes=1000, in_channels=3)
        assert model is not None

    def test_classifier_forward(self):
        """Test classifier forward pass."""
        model = ObjectClassifier(num_classes=10, in_channels=3, base_channels=16)
        x = torch.randn(2, 3, 224, 224)

        output = model(x)

        assert output.shape == (2, 10)

    def test_detector_initialization(self):
        """Test detector initialization."""
        model = ObjectDetector(num_classes=80, in_channels=3)
        assert model is not None

    def test_detector_forward(self):
        """Test detector forward pass."""
        model = ObjectDetector(num_classes=10, in_channels=3, base_channels=16)
        x = torch.randn(2, 3, 224, 224)

        output = model(x)

        assert 'classes' in output
        assert 'boxes' in output
        assert 'objectness' in output


class TestSegmentationModel:
    """Tests for segmentation model."""

    def test_model_initialization(self):
        """Test model initialization."""
        model = SegmentationModel(in_channels=3, num_classes=21, base_channels=32)
        assert model is not None
        assert model.num_classes == 21

    def test_forward_pass(self):
        """Test forward pass."""
        model = SegmentationModel(in_channels=3, num_classes=21, base_channels=16)
        x = torch.randn(2, 3, 128, 128)

        output = model(x)

        assert output.shape == (2, 21, 128, 128)

    def test_prediction(self):
        """Test prediction method."""
        model = SegmentationModel(in_channels=3, num_classes=21, base_channels=16)
        x = torch.randn(2, 3, 128, 128)

        predictions = model.predict(x)

        assert predictions.shape == (2, 128, 128)
        assert predictions.dtype == torch.int64


class TestAnomalyDetection:
    """Tests for anomaly detection model."""

    def test_model_initialization(self):
        """Test model initialization."""
        model = AnomalyDetectionModel(
            input_channels=1,
            sequence_length=256,
            latent_dim=64
        )
        assert model is not None
        assert model.latent_dim == 64

    def test_forward_pass(self):
        """Test forward pass."""
        model = AnomalyDetectionModel(input_channels=1, sequence_length=256)
        x = torch.randn(2, 1, 256)

        reconstruction, latent = model(x)

        assert reconstruction.shape == (2, 1, 256)
        assert latent.shape == (2, 64)

    def test_encode_decode(self):
        """Test encode and decode methods."""
        model = AnomalyDetectionModel(input_channels=1, sequence_length=256)
        x = torch.randn(2, 1, 256)

        # Encode
        z = model.encode(x)
        assert z.shape == (2, 64)

        # Decode
        x_recon = model.decode(z)
        assert x_recon.shape == (2, 1, 256)

    def test_anomaly_score(self):
        """Test anomaly score computation."""
        model = AnomalyDetectionModel(input_channels=1, sequence_length=256)
        x = torch.randn(2, 1, 256)

        scores = model.compute_anomaly_score(x)

        assert scores.shape == (2,)
        assert torch.all(scores >= 0)

    def test_detect_anomalies(self):
        """Test anomaly detection."""
        model = AnomalyDetectionModel(input_channels=1, sequence_length=256)
        x = torch.randn(2, 1, 256)

        is_anomaly, scores = model.detect_anomalies(x, threshold=0.5)

        assert is_anomaly.shape == (2,)
        assert scores.shape == (2,)
        assert is_anomaly.dtype == torch.bool
