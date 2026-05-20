"""Evidently AI drift report for HOG feature distributions.

Compares HOG feature vectors extracted from a reference dataset against
new prediction data to detect data drift.

Note: Requires Python >= 3.11.1 due to an Evidently/typing compatibility
issue in 3.11.0. The script will work correctly in Docker (python:3.11-slim
ships 3.11.x where x >= 1) and in CI runners.

Usage:
    python -m monitoring.drift.drift_report \
        --reference-csv data/reference_hog_features.csv \
        --current-csv data/current_hog_features.csv \
        --output monitoring/drift/report.html

    # Or generate from images directly:
    python -m monitoring.drift.drift_report \
        --reference-dir images/reference/ \
        --current-dir images/current/ \
        --output monitoring/drift/report.html
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import cv2
import pandas as pd
from skimage.feature import hog

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

IMG_SIZE = 64

HOG_PARAMS = dict(
    orientations=9,
    pixels_per_cell=(8, 8),
    cells_per_block=(2, 2),
    block_norm="L2-Hys",
    transform_sqrt=False,
    feature_vector=True,
)


def _import_evidently():
    """Lazy-import evidently to avoid import-time crash on Python 3.11.0."""
    try:
        from evidently import ColumnMapping
        from evidently.metric_preset import DataDriftPreset
        from evidently.report import Report

        return ColumnMapping, DataDriftPreset, Report
    except (KeyError, TypeError) as exc:
        raise ImportError(
            "Evidently is incompatible with Python 3.11.0 due to a typing bug. "
            "Please use Python >= 3.11.1 or run inside Docker (python:3.11-slim)."
        ) from exc


def extract_hog_features_from_dir(image_dir: Path) -> pd.DataFrame:
    """Extract HOG feature vectors from all images in a directory.

    Args:
        image_dir: Directory containing image files (jpg/png).

    Returns:
        DataFrame with one row per image, columns are HOG feature dimensions.
    """
    features = []
    extensions = {".jpg", ".jpeg", ".png", ".bmp"}

    for img_path in sorted(image_dir.iterdir()):
        if img_path.suffix.lower() not in extensions:
            continue
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            logger.warning("Could not read %s, skipping.", img_path)
            continue
        img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE), cv2.INTER_AREA)
        feat = hog(img_resized, **HOG_PARAMS)
        features.append(feat)

    if not features:
        raise ValueError(f"No valid images found in {image_dir}")

    columns = [f"hog_{i}" for i in range(len(features[0]))]
    return pd.DataFrame(features, columns=columns)


def generate_drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    output_path: Path,
):
    """Generate an Evidently data drift report comparing two datasets.

    Args:
        reference: Reference HOG feature DataFrame.
        current: Current HOG feature DataFrame.
        output_path: Path to save the HTML report.

    Returns:
        The generated Evidently Report object.
    """
    ColumnMapping, DataDriftPreset, Report = _import_evidently()

    column_mapping = ColumnMapping(
        numerical_features=list(reference.columns),
    )

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference, current_data=current, column_mapping=column_mapping)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    report.save_html(str(output_path))
    logger.info("Drift report saved to %s", output_path)

    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate HOG feature drift report")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--reference-csv", type=Path, help="CSV with reference HOG features")
    group.add_argument("--reference-dir", type=Path, help="Directory with reference images")

    parser.add_argument("--current-csv", type=Path, help="CSV with current HOG features")
    parser.add_argument("--current-dir", type=Path, help="Directory with current images")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("monitoring/drift/report.html"),
        help="Output HTML report path",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Load reference data
    if args.reference_csv:
        reference = pd.read_csv(args.reference_csv)
        logger.info("Loaded reference from CSV: %d samples", len(reference))
    else:
        reference = extract_hog_features_from_dir(args.reference_dir)
        logger.info("Extracted reference HOG features: %d samples", len(reference))

    # Load current data
    if args.current_csv:
        current = pd.read_csv(args.current_csv)
        logger.info("Loaded current from CSV: %d samples", len(current))
    elif args.current_dir:
        current = extract_hog_features_from_dir(args.current_dir)
        logger.info("Extracted current HOG features: %d samples", len(current))
    else:
        raise ValueError("Provide either --current-csv or --current-dir")

    generate_drift_report(reference, current, args.output)


if __name__ == "__main__":
    main()
