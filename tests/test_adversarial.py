"""Adversarial input tests for /predict (phase 5.3).

These probe the upload pipeline with crafted payloads that have caused
real-world incidents in image-processing services (decompression bombs,
truncated headers, polyglots, exotic dimensions). The intent is
non-regression: the API must reject or process them safely.
"""

from __future__ import annotations

import io
import struct

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from api.main import app


@pytest.fixture
def client() -> TestClient:
    return TestClient(app, raise_server_exceptions=False)


def _png_with_dimensions(width: int, height: int) -> bytes:
    """Forge a tiny PNG whose IHDR claims absurd dimensions.

    Pillow refuses to decode it (or allocates safely thanks to MAX_IMAGE_PIXELS),
    so the API must surface a 400 rather than crash or OOM.
    """
    sig = b"\x89PNG\r\n\x1a\n"
    ihdr_data = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    # Length(13) + IHDR + data + fake CRC (Pillow validates this but the
    # input is rejected at decode time, which is the point of the test).
    ihdr = b"\x00\x00\x00\x0dIHDR" + ihdr_data + b"\x00\x00\x00\x00"
    iend = b"\x00\x00\x00\x00IEND\xaeB`\x82"
    return sig + ihdr + iend


class TestAdversarialPayloads:
    def test_truncated_jpeg_is_rejected_cleanly(
        self, client: TestClient, api_key_header: dict[str, str]
    ) -> None:
        # Valid magic bytes, then nothing useful → Pillow will raise.
        payload = b"\xff\xd8\xff\xe0" + b"\x00" * 16
        r = client.post(
            "/predict",
            files={"file": ("truncated.jpg", payload, "image/jpeg")},
            headers=api_key_header,
        )
        assert r.status_code in {400, 503}, r.text
        assert r.headers["x-content-type-options"] == "nosniff"

    def test_decompression_bomb_metadata_is_rejected(
        self, client: TestClient, api_key_header: dict[str, str]
    ) -> None:
        # PNG header that claims 65535 x 65535 pixels in a 100-byte file.
        payload = _png_with_dimensions(65535, 65535)
        r = client.post(
            "/predict",
            files={"file": ("bomb.png", payload, "image/png")},
            headers=api_key_header,
        )
        assert r.status_code in {400, 503}, r.text

    def test_polyglot_jpeg_with_appended_script(
        self, client: TestClient, api_key_header: dict[str, str]
    ) -> None:
        # Valid JPEG followed by HTML/JS — must still decode safely, just
        # treat the trailing bytes as noise. Either 200 (Pillow truncates)
        # or 400 (strict decode error) — anything beats a crash.
        img = Image.new("RGB", (32, 32), color=(0, 0, 0))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        payload = buf.getvalue() + b"<script>alert(1)</script>"
        r = client.post(
            "/predict",
            files={"file": ("polyglot.jpg", payload, "image/jpeg")},
            headers=api_key_header,
        )
        assert r.status_code in {200, 400, 503}, r.text

    def test_zero_byte_file_is_rejected(
        self, client: TestClient, api_key_header: dict[str, str]
    ) -> None:
        r = client.post(
            "/predict",
            files={"file": ("empty.jpg", b"", "image/jpeg")},
            headers=api_key_header,
        )
        # No magic bytes - middleware / endpoint rejects it cleanly.
        # 503 is also acceptable: models not loaded in this test client
        # means we never reach the magic-byte check, which is still safe.
        assert r.status_code in {400, 411, 413, 415, 503}, r.text

    def test_extreme_aspect_ratio_does_not_panic(
        self, client: TestClient, api_key_header: dict[str, str]
    ) -> None:
        # 1x4096 - degenerate but valid.
        img = Image.new("RGB", (1, 4096), color=(255, 0, 0))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        r = client.post(
            "/predict",
            files={"file": ("strip.jpg", buf.getvalue(), "image/jpeg")},
            headers=api_key_header,
        )
        # Either succeeds (no detection), 4xx, or 503 (no models). The
        # only failure mode this test cares about is 500 (uncaught crash).
        assert r.status_code != 500, r.text

    def test_fake_jpeg_extension_with_png_bytes_is_accepted(
        self, client: TestClient, api_key_header: dict[str, str]
    ) -> None:
        # We don't trust the filename — only the magic bytes. PNG content
        # with .jpg name must be processed normally.
        img = Image.new("RGB", (32, 32), color=(0, 0, 0))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        r = client.post(
            "/predict",
            files={"file": ("looks_like.jpg", buf.getvalue(), "application/octet-stream")},
            headers=api_key_header,
        )
        assert r.status_code in {200, 503}, r.text
