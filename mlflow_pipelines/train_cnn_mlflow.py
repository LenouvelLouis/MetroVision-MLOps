"""MLflow-tracked CNN binary classifier training.

Wraps the logic from train_cnn.py with full experiment tracking:
params, per-epoch metrics, and model artifact logging.

Usage:
    python -m mlflow_pipelines.train_cnn_mlflow \
        --data-xlsx path/to/Apprentissage_VF.xlsx \
        --image-dir path/to/BD_METRO \
        --output-dir model/
"""

from __future__ import annotations

import argparse
import logging
import os

import cv2
import mlflow
import numpy as np
import pandas as pd
from keras import layers, models
from tensorflow import keras

from mlflow_pipelines.config import (
    EXPERIMENT_CNN,
    IMG_SIZE,
    REGISTERED_MODEL_CNN,
    TRACKING_URI,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CNN binary classifier with MLflow tracking")
    parser.add_argument("--data-xlsx", required=True, help="Path to training Excel file")
    parser.add_argument("--image-dir", required=True, help="Path to image directory")
    parser.add_argument("--output-dir", default="model", help="Directory to save the trained model")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--validation-split", type=float, default=0.2)
    parser.add_argument("--img-size", type=int, default=IMG_SIZE)
    return parser.parse_args()


def load_training_data(
    xlsx_path: str, image_dir: str, img_size: int
) -> tuple[np.ndarray, np.ndarray]:
    """Load and preprocess training data from Excel + image directory."""
    df = pd.read_excel(xlsx_path)
    x_list, y_list = [], []

    for _, row in df.iterrows():
        img_path = os.path.join(image_dir, f"{row['NOM']}.JPG")
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        x1, x2 = int(row["x1"]), int(row["x2"])
        y1, y2 = int(row["y1"]), int(row["y2"])
        ymin, ymax = max(0, y1), min(img.shape[0], y2)
        xmin, xmax = max(0, x1), min(img.shape[1], x2)
        if ymin >= ymax or xmin >= xmax:
            continue
        roi = img[ymin:ymax, xmin:xmax]
        if roi.size == 0:
            continue
        roi_resized = cv2.resize(roi, (img_size, img_size), cv2.INTER_AREA)
        x_list.append(roi_resized)
        y_list.append(1 if int(row["HYP"]) > 0 else 0)

    x_arr = np.array(x_list, dtype="float32") / 255.0
    x_arr = x_arr[..., np.newaxis]
    y_arr = np.array(y_list, dtype="float32")
    return x_arr, y_arr


def build_model(img_size: int) -> keras.Model:
    """Build the binary CNN architecture (same as original train_cnn.py)."""
    return models.Sequential(
        [
            layers.Input((img_size, img_size, 1)),
            layers.Conv2D(32, 3, activation="relu", padding="same"),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, activation="relu", padding="same"),
            layers.MaxPooling2D(),
            layers.Flatten(),
            layers.Dense(64, activation="relu"),
            layers.Dense(1, activation="sigmoid"),
        ]
    )


def main() -> None:
    args = parse_args()

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_CNN)

    logger.info("Loading training data from %s", args.data_xlsx)
    x_train, y_train = load_training_data(args.data_xlsx, args.image_dir, args.img_size)
    n_positive = int(y_train.sum())
    n_negative = len(y_train) - n_positive
    logger.info(
        "Loaded %d samples (positive: %d, negative: %d)", len(y_train), n_positive, n_negative
    )

    with mlflow.start_run(run_name="cnn-binary-training") as run:
        # Log parameters
        mlflow.log_params(
            {
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "validation_split": args.validation_split,
                "img_size": args.img_size,
                "optimizer": "adam",
                "loss": "binary_crossentropy",
                "n_samples": len(y_train),
                "n_positive": n_positive,
                "n_negative": n_negative,
            }
        )

        # Build and train
        model = build_model(args.img_size)
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

        history = model.fit(
            x_train,
            y_train,
            validation_split=args.validation_split,
            epochs=args.epochs,
            batch_size=args.batch_size,
        )

        # Log per-epoch metrics
        for epoch_idx in range(args.epochs):
            mlflow.log_metrics(
                {
                    "train_loss": history.history["loss"][epoch_idx],
                    "train_accuracy": history.history["accuracy"][epoch_idx],
                    "val_loss": history.history["val_loss"][epoch_idx],
                    "val_accuracy": history.history["val_accuracy"][epoch_idx],
                },
                step=epoch_idx + 1,
            )

        # Log final metrics
        final_val_acc = history.history["val_accuracy"][-1]
        final_val_loss = history.history["val_loss"][-1]
        mlflow.log_metrics(
            {
                "final_val_accuracy": final_val_acc,
                "final_val_loss": final_val_loss,
            }
        )

        # Save and log model
        os.makedirs(args.output_dir, exist_ok=True)
        model_path = os.path.join(args.output_dir, "model_binary_real_metro.h5")
        model.save(model_path)
        mlflow.log_artifact(model_path, artifact_path="model")

        # Register model
        model_uri = f"runs:/{run.info.run_id}/model"
        mlflow.register_model(model_uri, REGISTERED_MODEL_CNN)

        logger.info(
            "Training complete. Run ID: %s | Val accuracy: %.4f",
            run.info.run_id,
            final_val_acc,
        )


if __name__ == "__main__":
    main()
