"""One-shot converter: pickle-based artefacts -> pickle-free artefacts.

Loads the legacy ``.h5`` / ``.joblib`` files **once**, in a controlled
local environment, and re-saves them in formats that contain no
executable code:

    model_binary_real_metro.h5  -> cnn.keras   (Keras 3 zip, safe_mode-loadable)
    knn_line_model.joblib       -> knn.npz     (training data + n_neighbors)
    scaler_line.joblib          -> scaler.npz  (fitted attributes only)

After running this, the runtime image stops shipping the .h5/.joblib
files and the API loads exclusively via :mod:`api.safe_loader` — no
``pickle.load`` in the prediction path.

Usage:
    python -m mlflow_pipelines.convert_to_safe --model-dir model/
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import joblib
import numpy as np
import tensorflow as tf

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def convert_cnn(src: Path, dst: Path) -> None:
    """.h5 -> .keras (zip format, no pickle on safe_mode load)."""
    model = tf.keras.models.load_model(str(src))
    model.save(str(dst))  # Keras 3 picks the .keras zip format from the extension.
    logger.info("CNN converted: %s -> %s", src.name, dst.name)


def convert_knn(src: Path, dst: Path) -> None:
    """.joblib -> .npz containing the training arrays and hyperparameters.

    A k-NN's "fit" is just a memorisation of (X, y). We must recover the
    *original* labels (not sklearn's internal `_y` index into `classes_`),
    otherwise predictions get shifted by the class encoding.
    """
    knn = joblib.load(src)
    metric = knn.metric or "minkowski"
    # knn._y holds indices into knn.classes_; mapping back yields the labels
    # originally passed to fit(). Storing those is what makes the rebuilt
    # classifier predict identically.
    y_original = np.asarray(knn.classes_)[np.asarray(knn._y)]
    np.savez_compressed(
        dst,
        X_train=np.asarray(knn._fit_X),
        y_train=y_original,
        n_neighbors=np.int64(knn.n_neighbors),
        metric=np.array(str(metric), dtype="U32"),
    )
    logger.info("KNN converted: %s -> %s", src.name, dst.name)


def convert_scaler(src: Path, dst: Path) -> None:
    """.joblib -> .npz containing only the fitted numerical attributes."""
    scaler = joblib.load(src)
    np.savez_compressed(
        dst,
        mean=np.asarray(scaler.mean_),
        scale=np.asarray(scaler.scale_),
        var=np.asarray(scaler.var_),
        n_features_in=np.int64(scaler.n_features_in_),
    )
    logger.info("Scaler converted: %s -> %s", src.name, dst.name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("model"),
        help="Directory containing the legacy artefacts and where safe outputs land.",
    )
    args = parser.parse_args()

    md: Path = args.model_dir
    convert_cnn(md / "model_binary_real_metro.h5", md / "cnn.keras")
    convert_knn(md / "knn_line_model.joblib", md / "knn.npz")
    convert_scaler(md / "scaler_line.joblib", md / "scaler.npz")
    logger.info("Conversion complete. Update model/manifest.json with the new SHAs.")


if __name__ == "__main__":
    main()
