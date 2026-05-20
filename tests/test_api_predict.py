"""Tests for POST /predict endpoint.

These tests exercise the full prediction pipeline, so they require
model files to be present in the model/ directory.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.main import app
from api.model_manager import model_manager

ROOT = Path(__file__).resolve().parent.parent
TEST_IMAGE = ROOT / "IM_(1).jpg"

# Models must be loaded for predict tests
needs_models = pytest.mark.skipif(
    not all(p.exists() for p in [
        ROOT / "model" / "model_binary_real_metro.h5",
        ROOT / "model" / "knn_line_model.joblib",
        ROOT / "model" / "scaler_line.joblib",
    ]),
    reason="Model files not present",
)


@pytest.fixture(scope="module")
def loaded_client() -> TestClient:
    """TestClient with models loaded (skipped if models missing)."""
    model_manager.load()
    return TestClient(app, raise_server_exceptions=False)


class TestPredictWithoutModels:
    """Test /predict behavior when models are not loaded."""

    def test_predict_returns_503_when_models_not_loaded(self) -> None:
        unloaded_client = TestClient(app, raise_server_exceptions=False)
        # Reset model state for this test
        original = model_manager._loaded
        model_manager._loaded = False
        try:
            with open(TEST_IMAGE, "rb") as f:
                response = unloaded_client.post("/predict", files={"file": f})
            assert response.status_code == 503
        finally:
            model_manager._loaded = original

    def test_predict_returns_422_without_file(self) -> None:
        client = TestClient(app, raise_server_exceptions=False)
        response = client.post("/predict")
        assert response.status_code == 422


@needs_models
class TestPredictWithModels:
    """Test /predict with real model inference."""

    def test_predict_returns_200(self, loaded_client: TestClient) -> None:
        with open(TEST_IMAGE, "rb") as f:
            response = loaded_client.post("/predict", files={"file": f})
        assert response.status_code == 200

    def test_predict_response_structure(self, loaded_client: TestClient) -> None:
        with open(TEST_IMAGE, "rb") as f:
            response = loaded_client.post("/predict", files={"file": f})
        data = response.json()
        assert "detections" in data
        assert "count" in data
        assert isinstance(data["detections"], list)
        assert data["count"] == len(data["detections"])

    def test_predict_detections_have_correct_fields(self, loaded_client: TestClient) -> None:
        with open(TEST_IMAGE, "rb") as f:
            response = loaded_client.post("/predict", files={"file": f})
        data = response.json()
        if data["count"] > 0:
            det = data["detections"][0]
            assert all(k in det for k in ("y1", "y2", "x1", "x2", "line"))

    def test_predict_with_resize_factor(self, loaded_client: TestClient) -> None:
        with open(TEST_IMAGE, "rb") as f:
            response = loaded_client.post(
                "/predict", files={"file": f}, params={"resize_factor": 0.8}
            )
        assert response.status_code == 200

    def test_health_reports_healthy_after_load(self, loaded_client: TestClient) -> None:
        response = loaded_client.get("/health")
        data = response.json()
        assert data["status"] == "healthy"
        assert data["models_loaded"] is True
