"""MLflow-tracked k-NN line classifier training.

Wraps the logic from train_knn_scaler.py with full experiment tracking:
params, accuracy metrics, confusion matrix, and model/scaler artifacts.

Usage:
    python -m mlflow_pipelines.train_knn_mlflow \
        --data-xlsx path/to/Apprentissage_VF.xlsx \
        --image-dir path/to/BD_METRO \
        --output-dir model/
"""

from __future__ import annotations

import argparse
import json
import logging
import os

import cv2
import joblib
import mlflow
import numpy as np
import pandas as pd
from skimage.feature import hog
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from mlflow_pipelines.config import (
    EXPERIMENT_KNN,
    HOG_PARAMS,
    IMG_SIZE,
    REGISTERED_MODEL_KNN,
    TRACKING_URI,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train k-NN line classifier with MLflow tracking")
    parser.add_argument("--data-xlsx", required=True, help="Path to training Excel file")
    parser.add_argument("--image-dir", required=True, help="Path to image directory")
    parser.add_argument("--output-dir", default="model", help="Directory to save model and scaler")
    parser.add_argument("--n-neighbors", type=int, default=3)
    parser.add_argument("--metric", default="euclidean")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--img-size", type=int, default=IMG_SIZE)
    return parser.parse_args()


def load_line_data(
    xlsx_path: str, image_dir: str, img_size: int
) -> tuple[np.ndarray, np.ndarray]:
    """Load ROIs with HYP > 0 and extract HOG features."""
    df = pd.read_excel(xlsx_path)
    df_pos = df[df["HYP"] > 0]

    rois, labels = [], []
    for _, row in df_pos.iterrows():
        img_path = os.path.join(image_dir, f"{row['NOM']}.JPG")
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        x1, x2 = int(row["x1"]), int(row["x2"])
        y1, y2 = int(row["y1"]), int(row["y2"])
        roi = img[y1:y2, x1:x2]
        if roi.size == 0:
            continue
        roi_r = cv2.resize(roi, (img_size, img_size), cv2.INTER_AREA)
        rois.append(roi_r)
        labels.append(int(row["HYP"]))

    x_hog = np.vstack([hog(r, **HOG_PARAMS) for r in rois])
    y = np.array(labels, dtype="int32")
    return x_hog, y


def main() -> None:
    args = parse_args()

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_KNN)

    logger.info("Loading and extracting HOG features from %s", args.data_xlsx)
    x_hog, y = load_line_data(args.data_xlsx, args.image_dir, args.img_size)
    logger.info("Loaded %d samples, %d unique classes", len(y), len(np.unique(y)))

    # Split
    x_train, x_test, y_train, y_test = train_test_split(
        x_hog, y, test_size=args.test_size, stratify=y, random_state=args.random_state
    )

    # Scale
    scaler = StandardScaler().fit(x_train)
    x_train_scaled = scaler.transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    with mlflow.start_run(run_name="knn-line-training") as run:
        # Log parameters
        mlflow.log_params({
            "n_neighbors": args.n_neighbors,
            "metric": args.metric,
            "test_size": args.test_size,
            "random_state": args.random_state,
            "img_size": args.img_size,
            "n_samples": len(y),
            "n_classes": len(np.unique(y)),
            "hog_orientations": HOG_PARAMS["orientations"],
            "hog_pixels_per_cell": str(HOG_PARAMS["pixels_per_cell"]),
            "hog_cells_per_block": str(HOG_PARAMS["cells_per_block"]),
            "hog_block_norm": HOG_PARAMS["block_norm"],
            "feature_dim": x_hog.shape[1],
        })

        # Train
        knn = KNeighborsClassifier(n_neighbors=args.n_neighbors, metric=args.metric)
        knn.fit(x_train_scaled, y_train)

        # Evaluate
        y_pred = knn.predict(x_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred, zero_division=0, output_dict=True)

        mlflow.log_metric("test_accuracy", accuracy)
        mlflow.log_metric("weighted_f1", report["weighted avg"]["f1-score"])
        mlflow.log_metric("weighted_precision", report["weighted avg"]["precision"])
        mlflow.log_metric("weighted_recall", report["weighted avg"]["recall"])

        # Log classification report as artifact
        os.makedirs(args.output_dir, exist_ok=True)
        report_path = os.path.join(args.output_dir, "classification_report.json")
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        mlflow.log_artifact(report_path, artifact_path="metrics")

        # Log confusion matrix as artifact
        cm_path = os.path.join(args.output_dir, "confusion_matrix.npy")
        np.save(cm_path, cm)
        mlflow.log_artifact(cm_path, artifact_path="metrics")

        # Save and log model + scaler
        knn_path = os.path.join(args.output_dir, "knn_line_model.joblib")
        scaler_path = os.path.join(args.output_dir, "scaler_line.joblib")
        joblib.dump(knn, knn_path)
        joblib.dump(scaler, scaler_path)
        mlflow.log_artifact(knn_path, artifact_path="model")
        mlflow.log_artifact(scaler_path, artifact_path="model")

        # Register model
        model_uri = f"runs:/{run.info.run_id}/model"
        mlflow.register_model(model_uri, REGISTERED_MODEL_KNN)

        logger.info(
            "Training complete. Run ID: %s | Accuracy: %.4f | Weighted F1: %.4f",
            run.info.run_id, accuracy, report["weighted avg"]["f1-score"],
        )


if __name__ == "__main__":
    main()
