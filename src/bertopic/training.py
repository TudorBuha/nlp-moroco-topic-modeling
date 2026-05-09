"""High-level BERTopic training (Step T2).

This module is data-agnostic: it accepts already-loaded docs + embeddings
and returns a fitted model. The CLI wrapper in `scripts/fit_bertopic.py`
deals with reading parquet / saving embeddings.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .model import BERTopicConfig, build_bertopic_model


def fit_bertopic(
    docs: list[str],
    embeddings: np.ndarray,
    cfg: BERTopicConfig | None = None,
):
    """Fit BERTopic on `docs` using pre-computed `embeddings`.

    Returns: (topic_model, topics, probs).
    """
    cfg = cfg or BERTopicConfig()
    topic_model = build_bertopic_model(cfg)
    topics, probs = topic_model.fit_transform(docs, embeddings)
    return topic_model, topics, probs


def save_model(topic_model, out_dir: Path | str) -> Path:
    """Persist a fitted BERTopic model to disk (safetensors format)."""
    out_dir = Path(out_dir)
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    topic_model.save(
        str(out_dir),
        serialization="safetensors",
        save_ctfidf=True,
    )
    return out_dir


def load_model(model_dir: Path | str):
    """Load a previously saved BERTopic model."""
    from bertopic import BERTopic

    return BERTopic.load(str(model_dir))
