"""Unit tests for `src.bertopic.inspection` (no real BERTopic required)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd

from src.bertopic.inspection import (
    attach_manual_labels,
    n_topics,
    outlier_proportion,
    topic_keyword_table,
)


def _fake_topic_model(topics: dict[int, list[tuple[str, float]]]) -> MagicMock:
    """Build a stub object with the BERTopic methods we use."""
    info_rows = [
        {"Topic": tid, "Count": idx + 1, "Name": f"{tid}_x"}
        for idx, tid in enumerate(topics)
    ]
    info_df = pd.DataFrame(info_rows)

    fake = MagicMock()
    fake.get_topic_info.return_value = info_df
    fake.get_topic.side_effect = lambda tid: topics.get(tid, [])
    return fake


class TestOutlierProportion:
    def test_all_outliers(self):
        assert outlier_proportion([-1, -1, -1]) == 1.0

    def test_no_outliers(self):
        assert outlier_proportion([0, 1, 2]) == 0.0

    def test_mixed(self):
        assert outlier_proportion([-1, 0, -1, 1]) == 0.5

    def test_empty(self):
        assert outlier_proportion([]) == 0.0


class TestNTopics:
    def test_excludes_outlier(self):
        model = _fake_topic_model({-1: [], 0: [("x", 1.0)], 1: [("y", 1.0)]})
        assert n_topics(model, exclude_outlier=True) == 2

    def test_includes_outlier(self):
        model = _fake_topic_model({-1: [], 0: [("x", 1.0)]})
        assert n_topics(model, exclude_outlier=False) == 2


class TestTopicKeywordTable:
    def test_basic(self):
        model = _fake_topic_model(
            {
                -1: [],
                0: [("politica", 0.9), ("guvern", 0.7), ("lege", 0.5)],
                1: [("fotbal", 0.8), ("meci", 0.6)],
            }
        )
        df = topic_keyword_table(model, top_k=10)
        assert list(df["topic_id"]) == [-1, 0, 1]
        row0 = df[df["topic_id"] == 0].iloc[0]
        assert "politica" in row0["top_keywords"]
        assert "guvern" in row0["top_keywords"]

    def test_respects_top_k(self):
        model = _fake_topic_model(
            {
                0: [(f"w{i}", 1.0 / (i + 1)) for i in range(20)],
            }
        )
        df = topic_keyword_table(model, top_k=3)
        kws = df.iloc[0]["top_keywords"].split(", ")
        assert kws == ["w0", "w1", "w2"]


class TestAttachManualLabels:
    def test_known_and_unknown(self):
        df = pd.DataFrame(
            {
                "topic_id": [0, 1, 2],
                "top_keywords": ["a", "b", "c"],
            }
        )
        out = attach_manual_labels(df, {0: "politics", 2: "sports"})
        assert out.loc[out["topic_id"] == 0, "manual_label"].iloc[0] == "politics"
        assert out.loc[out["topic_id"] == 1, "manual_label"].iloc[0] == ""
        assert out.loc[out["topic_id"] == 2, "manual_label"].iloc[0] == "sports"
