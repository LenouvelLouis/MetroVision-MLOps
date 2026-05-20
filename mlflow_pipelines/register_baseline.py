"""Register existing pre-trained models as a baseline run in MLflow.

This script logs the current model/ artifacts into MLflow without retraining,
creating a reference point for future experiments.

Usage:
    python -m mlflow_pipelines.register_baseline
    python -m mlflow_pipelines.register_baseline --model-dir model/
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import mlflow

from mlflow_pipelines.config import (
    EXPERIMENT_CNN,
    EXPERIMENT_KNN,
    REGISTERED_MODEL_CNN,
    REGISTERED_MODEL_KNN,
    TRACKING_URI,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MODEL_FILES = {
    "cnn": {
        "file": "model_binary_real_metro.h5",
        "experiment": EXPERIMENT_CNN,
        "registered_name": REGISTERED_MODEL_CNN,
    },
    "knn": {
        "files": ["knn_line_model.joblib", "scaler_line.joblib"],
        "experiment": EXPERIMENT_KNN,
        "registered_name": REGISTERED_MODEL_KNN,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Register baseline models in MLflow")
    parser.add_argument(
        "--model-dir", default="model", help="Directory containing pre-trained model files"
    )
    return parser.parse_args()


def register_cnn_baseline(model_dir: Path) -> None:
    """Register the CNN model as a baseline run."""
    mlflow.set_experiment(MODEL_FILES["cnn"]["experiment"])
    cnn_path = model_dir / MODEL_FILES["cnn"]["file"]

    if not cnn_path.exists():
        logger.warning("CNN model not found at %s, skipping.", cnn_path)
        return

    with mlflow.start_run(run_name="baseline-cnn") as run:
        mlflow.log_params({
            "source": "pre-trained-baseline",
            "model_file": MODEL_FILES["cnn"]["file"],
            "img_size": 64,
            "architecture": "Sequential(Conv2D-32, Conv2D-64, Dense-64, Dense-1)",
        })
        mlflow.log_artifact(str(cnn_path), artifact_path="model")

        client = mlflow.tracking.MlflowClient()
        model_name = MODEL_FILES["cnn"]["registered_name"]
        try:
            client.create_registered_model(model_name)
        except mlflow.exceptions.MlflowException:
            pass  # Already exists
        client.create_model_version(
            name=model_name,
            source=f"{run.info.artifact_uri}/model",
            run_id=run.info.run_id,
        )
        logger.info("CNN baseline registered. Run ID: %s", run.info.run_id)


def register_knn_baseline(model_dir: Path) -> None:
    """Register the k-NN model + scaler as a baseline run."""
    mlflow.set_experiment(MODEL_FILES["knn"]["experiment"])
    files = [model_dir / f for f in MODEL_FILES["knn"]["files"]]
    missing = [f for f in files if not f.exists()]

    if missing:
        logger.warning("Missing k-NN files: %s, skipping.", [str(f) for f in missing])
        return

    with mlflow.start_run(run_name="baseline-knn") as run:
        mlflow.log_params({
            "source": "pre-trained-baseline",
            "n_neighbors": 3,
            "metric": "euclidean",
            "hog_orientations": 9,
            "hog_pixels_per_cell": "(8, 8)",
            "hog_cells_per_block": "(2, 2)",
        })
        for f in files:
            mlflow.log_artifact(str(f), artifact_path="model")

        client = mlflow.tracking.MlflowClient()
        model_name = MODEL_FILES["knn"]["registered_name"]
        try:
            client.create_registered_model(model_name)
        except mlflow.exceptions.MlflowException:
            pass  # Already exists
        client.create_model_version(
            name=model_name,
            source=f"{run.info.artifact_uri}/model",
            run_id=run.info.run_id,
        )
        logger.info("k-NN baseline registered. Run ID: %s", run.info.run_id)


def main() -> None:
    args = parse_args()
    model_dir = Path(args.model_dir)

    mlflow.set_tracking_uri(TRACKING_URI)
    logger.info("Registering baseline models from %s", model_dir)

    register_cnn_baseline(model_dir)
    register_knn_baseline(model_dir)

    logger.info("Baseline registration complete.")


if __name__ == "__main__":
    main()
