"""Security-focused tests: upload size, file-type validation, headers, rate limit."""

from __future__ import annotations

import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from api.main import app


def _valid_jpeg(size: tuple[int, int] = (32, 32)) -> bytes:
    """Return a minimal valid JPEG payload."""
    img = Image.new("RGB", size, color=(128, 128, 128))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture
def client() -> TestClient:
    return TestClient(app, raise_server_exceptions=False)


class TestUploadSizeLimit:
    def test_oversized_upload_returns_413(self, client: TestClient) -> None:
        # Blocked by middleware BEFORE auth/endpoint — no header needed.
        big_payload = b"\xff\xd8\xff" + b"x" * (11 * 1024 * 1024)
        response = client.post(
            "/predict",
            files={"file": ("big.jpg", big_payload, "image/jpeg")},
        )
        assert response.status_code == 413


class TestFileTypeValidation:
    def test_non_image_payload_rejected_with_415(
        self, client: TestClient, api_key_header: dict[str, str]
    ) -> None:
        response = client.post(
            "/predict",
            files={"file": ("fake.jpg", b"this is definitely not an image", "image/jpeg")},
            headers=api_key_header,
        )
        assert response.status_code in {415, 503}
        if response.status_code == 415:
            assert "Unsupported" in response.json()["detail"]

    def test_png_magic_bytes_accepted(
        self, client: TestClient, api_key_header: dict[str, str]
    ) -> None:
        img = Image.new("RGB", (16, 16), color=(0, 0, 0))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        response = client.post(
            "/predict",
            files={"file": ("ok.png", buf.getvalue(), "image/png")},
            headers=api_key_header,
        )
        assert response.status_code in {200, 503}


class TestSecurityHeaders:
    @pytest.mark.parametrize("endpoint", ["/health", "/version", "/metrics"])
    def test_headers_present(self, client: TestClient, endpoint: str) -> None:
        response = client.get(endpoint)
        assert response.headers["X-Content-Type-Options"] == "nosniff"
        assert response.headers["X-Frame-Options"] == "DENY"
        assert response.headers["Referrer-Policy"] == "no-referrer"
        assert "Strict-Transport-Security" in response.headers


class TestRateLimit:
    def test_excess_requests_get_429(
        self, client: TestClient, api_key_header: dict[str, str]
    ) -> None:
        valid = _valid_jpeg()
        statuses = []
        for _ in range(12):
            r = client.post(
                "/predict",
                files={"file": ("t.jpg", valid, "image/jpeg")},
                headers=api_key_header,
            )
            statuses.append(r.status_code)

        assert 429 in statuses, f"Expected at least one 429 in {statuses}"

    def test_429_increments_error_counter(
        self, client: TestClient, api_key_header: dict[str, str]
    ) -> None:
        valid = _valid_jpeg()
        for _ in range(12):
            client.post(
                "/predict",
                files={"file": ("t.jpg", valid, "image/jpeg")},
                headers=api_key_header,
            )

        metrics_body = client.get("/metrics").text
        assert 'metrovision_errors_total{reason="rate_limited"}' in metrics_body
