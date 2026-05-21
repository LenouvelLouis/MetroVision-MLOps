"""API-key authentication for protected endpoints.

Server stores SHA-256 hashes of valid keys in a flat file (one hex hash
per line, comments allowed). Clients send the raw key in `X-API-Key`.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import os
from pathlib import Path

from fastapi import Header, HTTPException

logger = logging.getLogger(__name__)

_DEFAULT_PATH = "/run/secrets/api-keys"


def _keys_file() -> Path:
    return Path(os.getenv("METROVISION_API_KEYS_FILE", _DEFAULT_PATH))


def hash_key(raw: str) -> str:
    """Return the lowercase SHA-256 hex digest of a raw key."""
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _load_hashes() -> set[str]:
    path = _keys_file()
    if not path.exists():
        logger.warning("api_keys_file_missing", extra={"path": str(path)})
        return set()
    hashes: set[str] = set()
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line and not line.startswith("#"):
            hashes.add(line.lower())
    return hashes


_cached_hashes: set[str] | None = None


def _get_hashes() -> set[str]:
    global _cached_hashes
    if _cached_hashes is None:
        _cached_hashes = _load_hashes()
    return _cached_hashes


def reload_api_keys() -> None:
    """Force-reload the hash set from disk (mainly for tests / SIGHUP-style refresh)."""
    global _cached_hashes
    _cached_hashes = _load_hashes()


def require_api_key(
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> None:
    """FastAPI dependency: 401 unless `X-API-Key` matches a known hash."""
    if x_api_key is None:
        raise HTTPException(status_code=401, detail="Missing X-API-Key header")

    provided = hash_key(x_api_key)
    valid = _get_hashes()
    if not valid:
        raise HTTPException(
            status_code=503,
            detail="API is not configured: no API keys loaded",
        )

    if not any(hmac.compare_digest(provided, candidate) for candidate in valid):
        raise HTTPException(status_code=401, detail="Invalid API key")
