"""Shared pytest fixtures: API-key provisioning + rate-limit reset."""

from __future__ import annotations

import contextlib
import hashlib
from collections.abc import Iterator
from pathlib import Path

import pytest

import api.auth as auth_mod
from api.rate_limit import limiter

TEST_RAW_KEY = "test-key-for-pytest"


@pytest.fixture(autouse=True)
def _setup_api_keys(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[None]:
    """Each test runs against a fresh API-keys file containing one known key."""
    keys_file = tmp_path / "api-keys"
    keys_file.write_text(
        hashlib.sha256(TEST_RAW_KEY.encode()).hexdigest() + "\n"
    )
    monkeypatch.setenv("METROVISION_API_KEYS_FILE", str(keys_file))
    auth_mod.reload_api_keys()
    yield
    auth_mod.reload_api_keys()


@pytest.fixture(autouse=True)
def _reset_rate_limiter() -> Iterator[None]:
    """Each test starts with a clean rate-limit state."""
    with contextlib.suppress(Exception):
        limiter.reset()
    yield
    with contextlib.suppress(Exception):
        limiter.reset()


@pytest.fixture
def api_key_header() -> dict[str, str]:
    return {"X-API-Key": TEST_RAW_KEY}
