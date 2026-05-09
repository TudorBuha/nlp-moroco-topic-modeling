"""Helpers shared by every Streamlit tab.

All artifact-loading goes through here so we have a single place to:
- check for missing files (the app must never crash before the pipeline runs)
- format the "run X first" hint message consistently
- cache loads with `st.cache_data` / `st.cache_resource`
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

try:  # streamlit is optional at import time so unit tests can run without it
    import streamlit as st
except ImportError:  # pragma: no cover
    st = None  # type: ignore[assignment]


def _cache_data(func):
    if st is not None:
        return st.cache_data(show_spinner=False)(func)
    return func


def _cache_resource(func):
    if st is not None:
        return st.cache_resource(show_spinner=False)(func)
    return func


def file_status(path: Path) -> dict[str, Any]:
    """Lightweight introspection used by the Home tab's status table."""
    p = Path(path)
    return {
        "name": p.name,
        "exists": p.exists(),
        "size_kb": round(p.stat().st_size / 1024, 1) if p.exists() else 0,
    }


@_cache_data
def load_csv_or_none(path: str | Path) -> pd.DataFrame | None:
    p = Path(path)
    return pd.read_csv(p) if p.exists() else None


@_cache_data
def load_json_or_none(path: str | Path) -> dict | None:
    p = Path(path)
    if not p.exists():
        return None
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_html_or_none(path: str | Path) -> str | None:
    p = Path(path)
    if not p.exists():
        return None
    return p.read_text(encoding="utf-8")


def missing_artifact(path: str | Path, suggestion: str) -> None:
    """Render a friendly card explaining how to produce the missing file."""
    if st is None:  # pragma: no cover
        return
    st.info(
        f"📄 `{Path(path).name}` not found yet.\n\n"
        f"Run **{suggestion}** to generate it."
    )


@_cache_resource
def get_bertopic_model(model_dir: str | Path):
    """Load the saved BERTopic model. Cached for the lifetime of the app."""
    from bertopic import BERTopic

    p = Path(model_dir)
    if not p.exists():
        return None
    return BERTopic.load(str(p))


@_cache_resource
def get_romanian_embedder():
    """Load the RoBERT mean-pool encoder (only when Try-it-live needs it)."""
    from src.bertopic.embeddings import RomanianEmbedder

    return RomanianEmbedder()


@_cache_resource
def get_lda_artifacts(model_path: str | Path, dictionary_path: str | Path):
    """Load LDA model + dictionary (Mihai's outputs) if they exist."""
    model_p = Path(model_path)
    dict_p = Path(dictionary_path)
    if not (model_p.exists() and dict_p.exists()):
        return None
    from gensim.corpora import Dictionary
    from gensim.models import LdaModel

    return {
        "model": LdaModel.load(str(model_p)),
        "dictionary": Dictionary.load(str(dict_p)),
    }


@_cache_resource
def get_spacy_ro():
    """Load spaCy ro_core_news_sm for LDA-side preprocessing in Try-it-live."""
    import spacy

    try:
        return spacy.load("ro_core_news_sm", disable=["parser", "ner"])
    except OSError:
        return None


def section_header(title: str, subtitle: str | None = None) -> None:
    if st is None:  # pragma: no cover
        return
    st.header(title)
    if subtitle:
        st.caption(subtitle)
