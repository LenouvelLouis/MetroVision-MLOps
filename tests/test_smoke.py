"""Smoke tests for Phase 1 — project skeleton validation."""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

ROOT = Path(__file__).resolve().parent.parent


class TestDirectorySkeleton:
    """Verify that the target architecture directories exist."""

    EXPECTED_DIRS: ClassVar[list[str]] = [
        "api",
        "docker",
        "k8s",
        "monitoring",
        "mlflow_pipelines",
        "tests",
        "docs",
        "scripts",
        ".github/workflows",
        "model",
    ]

    def test_directories_exist(self) -> None:
        for d in self.EXPECTED_DIRS:
            assert (ROOT / d).is_dir(), f"Missing directory: {d}"

    def test_pyproject_toml_exists(self) -> None:
        assert (ROOT / "pyproject.toml").is_file()

    def test_readme_exists(self) -> None:
        assert (ROOT / "README.md").is_file()

    def test_license_exists(self) -> None:
        assert (ROOT / "LICENSE").is_file()


class TestPackagesImportable:
    """Verify that Python packages created in Phase 1 are importable."""

    def test_import_api(self) -> None:
        import api  # noqa: F401

    def test_import_mlflow_pipelines(self) -> None:
        import mlflow_pipelines  # noqa: F401
