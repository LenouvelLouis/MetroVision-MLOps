"""Load model artefacts without invoking pickle.

The legacy `myMetroProcessing.load_models()` used `tf.keras.models.load_model`
on a `.h5` file and `joblib.load` on `.joblib` files — both of which call
`pickle.load` under the hood and would execute arbitrary code embedded in
a tampered artefact.

This module loads the post-conversion artefacts (`cnn.keras`, `knn.npz`,
`scaler.npz` produced by `mlflow_pipelines.convert_to_safe`) using only
formats that contain no executable code:

  - `cnn.keras` is a Keras 3 zip (JSON config + .npy weights);
    `load_model(..., safe_mode=True)` refuses pickled lambdas.
  - `knn.npz` and `scaler.npz` are `numpy.savez` archives — plain arrays.

The reconstructed objects are injected into `myMetroProcessing`'s module
namespace so the unchanged `processOneMetroImage` keeps working.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


def _load_cnn(path: Path):
    return tf.keras.models.load_model(str(path), safe_mode=True)


def _load_knn(path: Path) -> KNeighborsClassifier:
    """Rebuild a KNeighborsClassifier from its stored training arrays.

    k-NN's ``fit`` is just memorisation; replaying it on the same arrays
    yields a classifier that is bit-identical at prediction time.
    """
    data = np.load(path)
    knn = KNeighborsClassifier(
        n_neighbors=int(data["n_neighbors"]),
        metric=str(data["metric"]),
    )
    knn.fit(data["X_train"], data["y_train"])
    return knn


def _load_scaler(path: Path) -> StandardScaler:
    """Rebuild a StandardScaler from its fitted attributes (mean/scale/var)."""
    data = np.load(path)
    scaler = StandardScaler()
    scaler.mean_ = data["mean"]
    scaler.scale_ = data["scale"]
    scaler.var_ = data["var"]
    scaler.n_features_in_ = int(data["n_features_in"])
    return scaler


def load_models_safe(model_dir: Path) -> None:
    """Load all artefacts and inject them into `myMetroProcessing`.

    Mutates the academic module's globals (``model_bin``, ``knn``,
    ``scaler_line``) in place so the existing ``processOneMetroImage``
    pipeline keeps working without modification.
    """
    import myMetroProcessing as mmp

    mmp.model_bin = _load_cnn(model_dir / "cnn.keras")
    mmp.knn = _load_knn(model_dir / "knn.npz")
    mmp.scaler_line = _load_scaler(model_dir / "scaler.npz")
    logger.info("safe_models_loaded")
