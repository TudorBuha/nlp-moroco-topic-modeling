"""Tests for `src.paths`. Just structural / no-side-effects checks."""

from __future__ import annotations

from src import paths


def test_root_contains_src_and_data():
    assert (paths.ROOT / "src").is_dir()
    assert (paths.ROOT / "data").is_dir()


def test_paths_are_under_root():
    for p in (
        paths.DATA_PROCESSED,
        paths.RESULTS_BERTOPIC,
        paths.EMBEDDINGS_TRAIN,
        paths.MODEL_DIR_MAIN,
    ):
        assert paths.ROOT in p.parents or paths.ROOT == p


def test_ensure_dirs_is_idempotent(tmp_path, monkeypatch):
    paths.ensure_dirs()
    paths.ensure_dirs()
    assert paths.DATA_PROCESSED.is_dir()
    assert paths.RESULTS_BERTOPIC.is_dir()
