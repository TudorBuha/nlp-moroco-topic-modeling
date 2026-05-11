"""Render presentation diagrams from pre-built SVG assets."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

STATIC_DIR = Path(__file__).resolve().parent / "static"


def render_mermaid(diagram_key: str, height: int = 420) -> None:
    """Show a scrollable, pre-rendered Mermaid SVG (no live CDN render)."""
    svg_path = STATIC_DIR / f"{diagram_key}.svg"
    if not svg_path.exists():
        st.warning(
            f"Diagram `{diagram_key}.svg` is missing under `app/static/`. "
            "Re-run `python scripts/render_diagrams.py`."
        )
        return

    svg = svg_path.read_text(encoding="utf-8")
    st.markdown(
        f'<div style="overflow:auto; max-height:{height}px; width:100%;">{svg}</div>',
        unsafe_allow_html=True,
    )
