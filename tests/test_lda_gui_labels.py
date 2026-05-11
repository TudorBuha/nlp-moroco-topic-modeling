"""Tests for LDA + BERTopic GUI topic labels."""

from __future__ import annotations

from src.bertopic.gui_labels import (
    DEFAULT_BERTOPIC_GUI_LABELS,
    default_label as bertopic_default_label,
)
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


def test_bertopic_default_labels_include_outlier_and_topics():
    assert -1 in DEFAULT_BERTOPIC_GUI_LABELS
    assert "Outlier" in DEFAULT_BERTOPIC_GUI_LABELS[-1]
    assert "Sports" in (bertopic_default_label(1) or "")
    assert "Weather" in (bertopic_default_label(17) or "")


def test_app_bertopic_topic_display_name_falls_back():
    from app.utils import bertopic_topic_display_name

    assert bertopic_topic_display_name(1) == DEFAULT_BERTOPIC_GUI_LABELS[1]
    assert "Outlier" in bertopic_topic_display_name(-1)
    assert bertopic_topic_display_name(999) == "Topic 999"
