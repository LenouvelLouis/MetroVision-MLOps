"""Tests for /health and /version endpoints.

These tests use a test client WITHOUT triggering the lifespan (model loading),
so they validate the API structure independently of model availability.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app, raise_server_exceptions=False)


class TestHealthEndpoint:
    """Tests for GET /health."""

    def test_health_returns_200(self) -> None:
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_response_structure(self) -> None:
        response = client.get("/health")
        data = response.json()
        assert "status" in data
        assert "models_loaded" in data
        assert isinstance(data["models_loaded"], bool)

    def test_health_without_lifespan_reports_unhealthy(self) -> None:
        """Without lifespan, models are not loaded, so status should be unhealthy."""
        response = client.get("/health")
        data = response.json()
        assert data["status"] == "unhealthy"
        assert data["models_loaded"] is False


class TestVersionEndpoint:
    """Tests for GET /version."""

    def test_version_returns_200(self) -> None:
        response = client.get("/version")
        assert response.status_code == 200

    def test_version_response_structure(self) -> None:
        response = client.get("/version")
        data = response.json()
        assert "version" in data
        assert "python_version" in data
        assert "models" in data
        assert data["version"] == "0.1.0"

    def test_version_models_lists_expected_keys(self) -> None:
        response = client.get("/version")
        models = response.json()["models"]
        assert "cnn" in models
        assert "knn" in models
        assert "scaler" in models
