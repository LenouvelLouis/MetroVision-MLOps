"""Tests for MLflow pipeline configuration and module importability."""

from __future__ import annotations


class TestMlflowConfig:
    """Verify config constants are properly defined."""

    def test_tracking_uri_is_string(self) -> None:
        from mlflow_pipelines.config import TRACKING_URI

        assert isinstance(TRACKING_URI, str)
        assert TRACKING_URI.startswith("http")

    def test_experiment_names_defined(self) -> None:
        from mlflow_pipelines.config import EXPERIMENT_CNN, EXPERIMENT_KNN

        assert EXPERIMENT_CNN
        assert EXPERIMENT_KNN
        assert EXPERIMENT_CNN != EXPERIMENT_KNN

    def test_registered_model_names_defined(self) -> None:
        from mlflow_pipelines.config import REGISTERED_MODEL_CNN, REGISTERED_MODEL_KNN

        assert REGISTERED_MODEL_CNN
        assert REGISTERED_MODEL_KNN

    def test_hog_params_has_expected_keys(self) -> None:
        from mlflow_pipelines.config import HOG_PARAMS

        expected = {"orientations", "pixels_per_cell", "cells_per_block", "block_norm"}
        assert expected.issubset(set(HOG_PARAMS.keys()))


class TestModuleImports:
    """Verify pipeline modules are importable without side effects."""

    def test_import_config(self) -> None:
        import mlflow_pipelines.config  # noqa: F401

    def test_import_train_cnn_mlflow(self) -> None:
        import mlflow_pipelines.train_cnn_mlflow  # noqa: F401

    def test_import_train_knn_mlflow(self) -> None:
        import mlflow_pipelines.train_knn_mlflow  # noqa: F401

    def test_import_register_baseline(self) -> None:
        import mlflow_pipelines.register_baseline  # noqa: F401
