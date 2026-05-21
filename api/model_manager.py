"""Singleton wrapper around the academic model loading code.

Verifies that the on-disk model artefacts match the SHA-256 digests
declared in `model/manifest.json` BEFORE calling into `joblib.load`
or `keras.load_model`. The manifest is committed; any deliberate model
rotation requires updating it (which goes through code review).
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).resolve().parent.parent / "model"
MANIFEST_PATH = MODEL_DIR / "manifest.json"


class ModelIntegrityError(RuntimeError):
    """Raised when an on-disk model artefact does not match its manifest digest."""


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_manifest() -> dict[str, dict[str, str]]:
    """Read and validate the model manifest.

    Returns a dict {logical_name: {"file": str, "sha256": str}}.
    """
    if not MANIFEST_PATH.exists():
        raise ModelIntegrityError(
            f"Model manifest is missing at {MANIFEST_PATH}. Refusing to load any model."
        )
    try:
        raw = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ModelIntegrityError(f"Model manifest is not valid JSON: {exc}") from exc

    models = raw.get("models")
    if not isinstance(models, dict) or not models:
        raise ModelIntegrityError("Model manifest must declare a non-empty 'models' object.")
    return models


def _verify_artefacts(models: dict[str, dict[str, str]]) -> None:
    """Compute the digest of each declared file and compare with the manifest."""
    for name, entry in models.items():
        filename = entry.get("file")
        expected = entry.get("sha256", "").lower()
        if not filename or not expected:
            raise ModelIntegrityError(
                f"Manifest entry '{name}' must define both 'file' and 'sha256'."
            )

        path = MODEL_DIR / filename
        if not path.exists():
            raise ModelIntegrityError(f"Model file missing: {path}")

        actual = _sha256(path)
        if not hmac.compare_digest(actual.lower(), expected):
            raise ModelIntegrityError(
                f"Integrity check failed for '{name}' ({filename}): "
                f"expected sha256={expected}, got {actual}. Refusing to load."
            )

        logger.info(
            "model_integrity_ok",
            extra={"model_name": name, "file": filename, "sha256": expected[:12]},
        )


class ModelManager:
    """Manages the lifecycle of ML models used by the inference pipeline."""

    def __init__(self) -> None:
        self._loaded = False
        self._manifest: dict[str, dict[str, str]] | None = None

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def load(self) -> None:
        """Load all models into memory via the academic code.

        Raises:
            ModelIntegrityError: if the manifest is missing, malformed, or
                if any artefact's SHA-256 does not match the declared digest.
        """
        if self._loaded:
            logger.info("Models already loaded, skipping.")
            return

        self._manifest = _load_manifest()
        _verify_artefacts(self._manifest)

        from api.safe_loader import load_models_safe

        load_models_safe(MODEL_DIR)
        self._loaded = True
        logger.info("All models loaded successfully.")

    def model_info(self) -> dict[str, str]:
        """Return metadata about model files (logical name → file name)."""
        manifest = self._manifest or _load_manifest()
        return {name: entry["file"] for name, entry in manifest.items()}


# Module-level singleton
model_manager = ModelManager()
