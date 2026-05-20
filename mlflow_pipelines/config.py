"""Shared MLflow configuration constants."""

from __future__ import annotations

import os

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")

EXPERIMENT_CNN = "metrovision-cnn-binary"
EXPERIMENT_KNN = "metrovision-knn-line"

REGISTERED_MODEL_CNN = "metrovision-cnn"
REGISTERED_MODEL_KNN = "metrovision-knn"

IMG_SIZE = 64

HOG_PARAMS = dict(
    orientations=9,
    pixels_per_cell=(8, 8),
    cells_per_block=(2, 2),
    block_norm="L2-Hys",
    transform_sqrt=False,
    feature_vector=True,
)
