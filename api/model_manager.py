"""Singleton wrapper around the academic model loading code.

Provides state tracking (loaded/not) without modifying the original
myMetroProcessing module.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).resolve().parent.parent / "model"

EXPECTED_FILES = {
    "cnn": MODEL_DIR / "model_binary_real_metro.h5",
    "knn": MODEL_DIR / "knn_line_model.joblib",
    "scaler": MODEL_DIR / "scaler_line.joblib",
}


class ModelManager:
    """Manages the lifecycle of ML models used by the inference pipeline."""

    def __init__(self) -> None:
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def load(self) -> None:
        """Load all models into memory via the academic code."""
        if self._loaded:
            logger.info("Models already loaded, skipping.")
            return

        missing = [name for name, path in EXPECTED_FILES.items() if not path.exists()]
        if missing:
            raise FileNotFoundError(
                f"Missing model files: {', '.join(missing)}. Expected in {MODEL_DIR}"
            )

        from myMetroProcessing import load_models

        load_models()
        self._loaded = True
        logger.info("All models loaded successfully.")

    def model_info(self) -> dict[str, str]:
        """Return metadata about model files."""
        return {name: str(path.name) for name, path in EXPECTED_FILES.items()}


# Module-level singleton
model_manager = ModelManager()
