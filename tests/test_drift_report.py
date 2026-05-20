"""Tests for the Evidently drift report module."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def _can_import_evidently() -> bool:
    """Check if evidently's drift report machinery is importable."""
    try:
        from monitoring.drift.drift_report import _import_evidently

        _import_evidently()
        return True
    except (ImportError, KeyError, TypeError):
        return False


needs_evidently = pytest.mark.skipif(
    not _can_import_evidently(),
    reason="Evidently not importable (version or Python compatibility issue)",
)


class TestDriftReportImport:
    """Verify drift report module is importable (lazy evidently import)."""

    def test_import_module(self) -> None:
        import monitoring.drift.drift_report  # noqa: F401

    def test_import_extract_function(self) -> None:
        from monitoring.drift.drift_report import extract_hog_features_from_dir  # noqa: F401

    def test_import_generate_function(self) -> None:
        from monitoring.drift.drift_report import generate_drift_report  # noqa: F401


class TestHogExtraction:
    """Test HOG feature extraction (no evidently dependency)."""

    def test_extract_from_test_images(self) -> None:
        from monitoring.drift.drift_report import extract_hog_features_from_dir

        root = Path(__file__).resolve().parent.parent
        # Use the repo root which has IM_*.jpg files
        df = extract_hog_features_from_dir(root)
        assert len(df) > 0
        assert all(col.startswith("hog_") for col in df.columns)


@needs_evidently
class TestDriftReportGeneration:
    """Test drift report generation with synthetic HOG-like data."""

    @pytest.fixture
    def synthetic_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Create synthetic reference and current DataFrames mimicking HOG features."""
        rng = np.random.default_rng(42)
        n_features = 36
        columns = [f"hog_{i}" for i in range(n_features)]

        reference = pd.DataFrame(
            rng.normal(0, 1, size=(100, n_features)),
            columns=columns,
        )
        current = pd.DataFrame(
            rng.normal(0.3, 1.2, size=(50, n_features)),
            columns=columns,
        )
        return reference, current

    def test_report_generates_html(
        self, synthetic_data: tuple[pd.DataFrame, pd.DataFrame], tmp_path: Path
    ) -> None:
        from monitoring.drift.drift_report import generate_drift_report

        reference, current = synthetic_data
        output = tmp_path / "drift_report.html"

        report = generate_drift_report(reference, current, output)

        assert output.exists()
        assert output.stat().st_size > 0
        assert report is not None

    def test_report_html_contains_drift_info(
        self, synthetic_data: tuple[pd.DataFrame, pd.DataFrame], tmp_path: Path
    ) -> None:
        from monitoring.drift.drift_report import generate_drift_report

        reference, current = synthetic_data
        output = tmp_path / "drift_report.html"

        generate_drift_report(reference, current, output)

        html_content = output.read_text(encoding="utf-8")
        assert len(html_content) > 1000
