"""Tests for the model-manifest integrity check (phase 5.1)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import api.model_manager as mm_module
from api.model_manager import ModelIntegrityError, ModelManager

ROOT = Path(__file__).resolve().parent.parent
MODELS = [
    "cnn.keras",
    "knn.npz",
    "scaler.npz",
]


def _build_real_manifest(target_dir: Path) -> dict:
    """Compute true SHA-256 of each model and assemble a fresh manifest."""
    digests = {}
    for fname in MODELS:
        src = ROOT / "model" / fname
        h = hashlib.sha256()
        with src.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 16), b""):
                h.update(chunk)
        digests[fname] = h.hexdigest()
    return {
        "version": "1",
        "models": {
            "cnn": {"file": MODELS[0], "sha256": digests[MODELS[0]]},
            "knn": {"file": MODELS[1], "sha256": digests[MODELS[1]]},
            "scaler": {"file": MODELS[2], "sha256": digests[MODELS[2]]},
        },
    }


@pytest.fixture
def isolated_model_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Copy model files into a tmp dir + ship a valid manifest. Redirect MODEL_DIR there."""
    target = tmp_path / "model"
    target.mkdir()
    for fname in MODELS:
        (target / fname).write_bytes((ROOT / "model" / fname).read_bytes())
    manifest = _build_real_manifest(target)
    (target / "manifest.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(mm_module, "MODEL_DIR", target)
    monkeypatch.setattr(mm_module, "MANIFEST_PATH", target / "manifest.json")
    return target


class TestManifestPresence:
    def test_missing_manifest_refuses_load(
        self, isolated_model_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (isolated_model_dir / "manifest.json").unlink()
        with pytest.raises(ModelIntegrityError, match="manifest is missing"):
            ModelManager().load()

    def test_malformed_manifest_refuses_load(self, isolated_model_dir: Path) -> None:
        (isolated_model_dir / "manifest.json").write_text("{ not json")
        with pytest.raises(ModelIntegrityError, match="not valid JSON"):
            ModelManager().load()

    def test_empty_models_block_refuses_load(self, isolated_model_dir: Path) -> None:
        (isolated_model_dir / "manifest.json").write_text('{"models": {}}')
        with pytest.raises(ModelIntegrityError, match="non-empty 'models'"):
            ModelManager().load()


class TestArtefactIntegrity:
    def test_tampered_npz_blocks_startup(self, isolated_model_dir: Path) -> None:
        path = isolated_model_dir / "knn.npz"
        original = path.read_bytes()
        path.write_bytes(original + b"\x00")  # single byte appended -> digest changes
        try:
            with pytest.raises(ModelIntegrityError, match="Integrity check failed"):
                ModelManager().load()
        finally:
            path.write_bytes(original)

    def test_tampered_keras_blocks_startup(self, isolated_model_dir: Path) -> None:
        path = isolated_model_dir / "cnn.keras"
        original = path.read_bytes()
        mutated = bytearray(original)
        mutated[100] ^= 0xFF
        path.write_bytes(bytes(mutated))
        try:
            with pytest.raises(ModelIntegrityError, match="Integrity check failed"):
                ModelManager().load()
        finally:
            path.write_bytes(original)

    def test_missing_artefact_blocks_startup(self, isolated_model_dir: Path) -> None:
        (isolated_model_dir / "scaler.npz").unlink()
        with pytest.raises(ModelIntegrityError, match="Model file missing"):
            ModelManager().load()


class TestHappyPath:
    def test_unmodified_artefacts_pass_check(self, isolated_model_dir: Path) -> None:
        mgr = ModelManager()
        from api.model_manager import _load_manifest, _verify_artefacts

        manifest = _load_manifest()
        _verify_artefacts(manifest)  # should not raise
        assert mgr.model_info()["cnn"] == "cnn.keras"
