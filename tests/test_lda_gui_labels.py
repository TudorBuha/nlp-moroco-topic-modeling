"""Tests for LDA GUI topic labels."""

from __future__ import annotations

from src.lda.gui_labels import DEFAULT_LDA_GUI_LABELS, default_label


def test_default_labels_cover_k15_topics():
    assert set(DEFAULT_LDA_GUI_LABELS) == set(range(15))


def test_default_label_finance_topic():
    assert "Finance" in (default_label(1) or "")


def test_app_lda_topic_display_name_matches_builtin():
    """Streamlit utils merge CSV overrides with builtins — topic 1 stays finance."""
    from app.utils import lda_topic_display_name

    assert lda_topic_display_name(1) == DEFAULT_LDA_GUI_LABELS[1]
    assert lda_topic_display_name(999) == "Topic 999"
