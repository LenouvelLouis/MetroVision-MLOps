"""API-key authentication tests for /predict."""

from __future__ import annotations

import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from api.main import app


def _valid_jpeg() -> bytes:
    img = Image.new("RGB", (16, 16), color=(0, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture
def client() -> TestClient:
    return TestClient(app, raise_server_exceptions=False)


class TestAuthRequired:
    def test_missing_header_returns_401(self, client: TestClient) -> None:
        response = client.post(
            "/predict",
            files={"file": ("t.jpg", _valid_jpeg(), "image/jpeg")},
        )
        assert response.status_code == 401
        assert "Missing" in response.json()["detail"]

    def test_invalid_key_returns_401(self, client: TestClient) -> None:
        response = client.post(
            "/predict",
            files={"file": ("t.jpg", _valid_jpeg(), "image/jpeg")},
            headers={"X-API-Key": "definitely-not-the-right-key"},
        )
        assert response.status_code == 401
        assert "Invalid" in response.json()["detail"]

    def test_valid_key_passes_auth(
        self, client: TestClient, api_key_header: dict[str, str]
    ) -> None:
        # Models aren't loaded in this client → 503 confirms auth passed.
        # The point is: NOT 401.
        response = client.post(
            "/predict",
            files={"file": ("t.jpg", _valid_jpeg(), "image/jpeg")},
            headers=api_key_header,
        )
        assert response.status_code != 401


class TestPublicEndpointsUnaffected:
    """Auth is scoped to /predict — infrastructure endpoints stay open."""

    @pytest.mark.parametrize("endpoint", ["/health", "/version", "/metrics"])
    def test_no_auth_required(self, client: TestClient, endpoint: str) -> None:
        response = client.get(endpoint)
        assert response.status_code == 200
