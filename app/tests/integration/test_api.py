"""
Integration tests for FastAPI endpoints.
"""
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


class TestHealthEndpoints:
    """Tests for health check endpoints."""

    def test_root_endpoint(self):
        """Test root endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "version" in data
        assert "status" in data

    def test_health_check(self):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"


class TestModelEndpoints:
    """Tests for model API endpoints."""

    def test_list_models(self):
        """Test listing available models."""
        response = client.get("/api/v1/models")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) > 0

    def test_get_model_info(self):
        """Test getting model information."""
        response = client.get("/api/v1/model/tenn_eeg_v1/info")
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "tenn_eeg_v1"
        assert "type" in data
        assert "input_shape" in data

    def test_get_nonexistent_model(self):
        """Test getting info for nonexistent model."""
        response = client.get("/api/v1/model/nonexistent/info")
        assert response.status_code == 404


class TestPredictionEndpoints:
    """Tests for prediction endpoints."""

    def test_predict_eeg(self):
        """Test EEG prediction endpoint."""
        data = {
            "data": [[0.1] * 256 for _ in range(64)],
            "model_type": "eeg"
        }
        response = client.post("/api/v1/predict/eeg", json=data)
        assert response.status_code == 200
        result = response.json()
        assert "predictions" in result
        assert "inference_time_ms" in result

    def test_predict_anomaly(self):
        """Test anomaly detection endpoint."""
        data = {
            "data": [[0.1] * 256],
            "model_type": "anomaly"
        }
        response = client.post("/api/v1/predict/anomaly", json=data)
        assert response.status_code == 200
        result = response.json()
        assert "is_anomaly" in result
        assert "anomaly_score" in result


class TestModelManagement:
    """Tests for model management endpoints."""

    def test_quantize_model(self):
        """Test model quantization endpoint."""
        response = client.post("/api/v1/model/tenn_eeg_v1/quantize?target=akida")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "quantized_model_path" in data

    def test_benchmark_model(self):
        """Test model benchmarking endpoint."""
        response = client.get("/api/v1/benchmark/tenn_eeg_v1?iterations=100")
        assert response.status_code == 200
        data = response.json()
        assert "avg_inference_time_ms" in data
        assert "throughput_samples_per_sec" in data
