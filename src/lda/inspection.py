"""Topic-keyword tables + manual labels for LDA models."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from gensim.models import LdaModel


def topic_keyword_table(model: LdaModel, top_n: int = 10) -> pd.DataFrame:
    """Return a tidy DataFrame with one row per LDA topic.

    Columns:
        topic_id, top_keywords (comma-separated), top_weights
    """
    rows: list[dict] = []
    for topic_id in range(model.num_topics):
        terms = model.show_topic(topic_id, topn=top_n)
        rows.append(
            {
                "topic_id": int(topic_id),
                "top_keywords": ", ".join(w for w, _ in terms),
                "top_weights": ", ".join(f"{w:.4f}" for _, w in terms),
            }
        )
    return pd.DataFrame(rows)


def attach_manual_labels(
    table: pd.DataFrame,
    labels: dict[int, str],
) -> pd.DataFrame:
    """Add a `label` column with human-readable topic names.

    Topics not present in `labels` get an empty string so the column is
    always populated and easy to scan in the Streamlit table.
    """
    out = table.copy()
    out["label"] = out["topic_id"].map(lambda tid: labels.get(int(tid), ""))
    return out


def save_topic_table(table: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)
    return path
