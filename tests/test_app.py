"""Smoke tests for the Streamlit `app/` package.

We don't actually run the Streamlit server (that needs `streamlit run`).
Instead we import every tab module to catch syntax/import errors and
exercise the artifact-loading helpers in `app.utils` against missing
files (the pre-data-handoff state the demo must survive).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


def test_app_modules_import():
    import app  # noqa: F401
    from app import streamlit_app, utils  # noqa: F401
    from app.tabs import (  # noqa: F401
        bertopic_tab,
        comparison,
        home,
        lda,
        stability,
        try_it,
    )


def test_utils_returns_none_for_missing_files(tmp_path: Path):
    from app.utils import (
        file_status,
        load_csv_or_none,
        load_html_or_none,
        load_json_or_none,
    )

    missing = tmp_path / "nope.csv"
    assert load_csv_or_none(missing) is None
    assert load_json_or_none(missing) is None
    assert load_html_or_none(missing) is None

    status = file_status(missing)
    assert status == {"name": "nope.csv", "exists": False, "size_kb": 0}


def test_utils_loads_csv_and_json(tmp_path: Path):
    import pandas as pd

    from app.utils import file_status, load_csv_or_none, load_json_or_none

    csv_path = tmp_path / "ok.csv"
    # Padded so the rounded `size_kb` (1 decimal) doesn't hit 0.
    pd.DataFrame({"a": list(range(50)), "b": [f"row_{i}" for i in range(50)]}).to_csv(
        csv_path, index=False
    )
    df = load_csv_or_none(csv_path)
    assert df is not None
    assert list(df.columns) == ["a", "b"]
    assert len(df) == 50

    json_path = tmp_path / "ok.json"
    json_path.write_text(json.dumps({"x": 1, "y": [1, 2, 3]}), encoding="utf-8")
    payload = load_json_or_none(json_path)
    assert payload == {"x": 1, "y": [1, 2, 3]}

    status = file_status(csv_path)
    assert status["exists"] is True
    assert status["size_kb"] > 0
    assert status["name"] == "ok.csv"


def test_streamlit_app_main_callable():
    """The main entry point must be callable (not invoked here, just checked)."""
    from app.streamlit_app import main

    assert callable(main)


@pytest.mark.parametrize(
    "tab_module",
    [
        "app.tabs.home",
        "app.tabs.try_it",
        "app.tabs.lda",
        "app.tabs.bertopic_tab",
        "app.tabs.comparison",
        "app.tabs.stability",
    ],
)
def test_tab_module_exposes_render(tab_module: str):
    import importlib

    mod = importlib.import_module(tab_module)
    assert hasattr(mod, "render"), f"{tab_module} must expose a render() function"
    assert callable(mod.render)
