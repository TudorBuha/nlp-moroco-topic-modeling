"""Topic inspection utilities (Tudor, Step T3).

Helpers around `BERTopic.get_topic_info()` / `get_topic()` for producing
keyword tables, counting outliers, and exporting CSVs."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pandas as pd


def topic_keyword_table(topic_model, top_k: int = 10) -> pd.DataFrame:
    """Return a DataFrame with one row per topic and top-k keywords joined.

    Columns: topic_id, count, name (BERTopic's auto-name), top_keywords.
    The outlier topic (-1) is included so you can decide whether to drop it.
    """
    info = topic_model.get_topic_info().copy()
    rows: list[dict] = []
    for tid in info["Topic"].tolist():
        keywords = topic_model.get_topic(tid)
        if not keywords:  # outlier in some configs
            kw_str = ""
        else:
            kw_str = ", ".join(w for w, _ in list(keywords)[:top_k])
        count = int(info.loc[info["Topic"] == tid, "Count"].iloc[0])
        name = str(info.loc[info["Topic"] == tid, "Name"].iloc[0])
        rows.append(
            {
                "topic_id": int(tid),
                "count": count,
                "name": name,
                "top_keywords": kw_str,
            }
        )
    return pd.DataFrame(rows).sort_values("topic_id").reset_index(drop=True)


def outlier_proportion(topics: Sequence[int]) -> float:
    """Fraction of documents BERTopic put into the outlier cluster (-1)."""
    if len(topics) == 0:
        return 0.0
    return float(sum(1 for t in topics if t == -1) / len(topics))


def n_topics(topic_model, exclude_outlier: bool = True) -> int:
    """Number of topics discovered (optionally excluding the outlier topic)."""
    ids = list(topic_model.get_topic_info()["Topic"])
    if exclude_outlier:
        ids = [i for i in ids if i != -1]
    return len(ids)


def save_topic_table(topic_model, out_path: Path | str, top_k: int = 10) -> Path:
    """Build the keyword table and write it to CSV."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = topic_keyword_table(topic_model, top_k=top_k)
    df.to_csv(out_path, index=False, encoding="utf-8")
    return out_path


def attach_manual_labels(
    topic_table: pd.DataFrame,
    labels: dict[int, str],
) -> pd.DataFrame:
    """Add a `manual_label` column based on a dict {topic_id: label}.
    Topics not in the dict get an empty string."""
    out = topic_table.copy()
    out["manual_label"] = out["topic_id"].map(labels).fillna("")
    return out
