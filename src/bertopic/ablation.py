"""Embedding ablation: RoBERT vs. multilingual MPNet (Step T5).

The MPNet model `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`
produces 768-d sentence embeddings already, so we use the
`sentence-transformers` API directly for that path; for RoBERT we keep using
our own mean-pool wrapper.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from ..evaluation.metrics import nmi_score, purity_score
from .coherence import compute_cv_coherence
from .embeddings import RomanianEmbedder
from .inspection import n_topics, outlier_proportion
from .model import BERTopicConfig
from .training import fit_bertopic

ROBERT_NAME = "readerbench/robert-base"
MPNET_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"


def encode_with_sbert(docs: Sequence[str], model_name: str) -> np.ndarray:
    """Encode using sentence-transformers (used for the MPNet baseline)."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    return model.encode(
        list(docs),
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
    ).astype(np.float32)


def run_one(
    name: str,
    embeddings_train: np.ndarray,
    embeddings_test: np.ndarray,
    docs_train: list[str],
    docs_test: list[str],
    y_test: Sequence,
    cfg: BERTopicConfig,
) -> dict:
    """Fit BERTopic with a given embedding source and evaluate on the test set.

    Wraps the fit in a try/except so a single embedding's failure (e.g.
    c-TF-IDF vocabulary collapse with `min_df=5` on a tiny topic group)
    doesn't kill the whole ablation table.
    """
    try:
        model, _topics_train, _ = fit_bertopic(docs_train, embeddings_train, cfg)
        topics_test, _ = model.transform(docs_test, embeddings_test)
        return {
            "embedding": name,
            "n_topics": n_topics(model, exclude_outlier=True),
            "outlier_pct_test": outlier_proportion(topics_test),
            "c_v_train": compute_cv_coherence(model, docs_train),
            "nmi_test": nmi_score(y_test, topics_test),
            "purity_test": purity_score(y_test, topics_test),
            "ok": True,
            "error": "",
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "embedding": name,
            "n_topics": -1,
            "outlier_pct_test": float("nan"),
            "c_v_train": float("nan"),
            "nmi_test": float("nan"),
            "purity_test": float("nan"),
            "ok": False,
            "error": str(exc),
        }


def run_ablation(
    docs_train: list[str],
    docs_test: list[str],
    y_test: Sequence,
    embeddings_robert_train: np.ndarray,
    embeddings_robert_test: np.ndarray,
    embeddings_mpnet_train: np.ndarray | None = None,
    embeddings_mpnet_test: np.ndarray | None = None,
    cfg: BERTopicConfig | None = None,
) -> pd.DataFrame:
    """Compare RoBERT vs. MPNet on identical docs/test split.

    If MPNet embeddings aren't provided they are computed on the fly.

    The ablation uses a slightly relaxed `min_df=2` so the comparison doesn't
    fail on encoders whose clustering happens to produce small topics with
    sparse vocabulary; the structural pipeline (UMAP / HDBSCAN / c-TF-IDF)
    is otherwise identical between the two runs.
    """
    cfg = cfg or BERTopicConfig()
    cfg = BERTopicConfig(
        n_neighbors=cfg.n_neighbors,
        n_components=cfg.n_components,
        min_dist=cfg.min_dist,
        min_cluster_size=cfg.min_cluster_size,
        min_df=2,
        ngram_range=cfg.ngram_range,
        language=cfg.language,
        calculate_probabilities=cfg.calculate_probabilities,
        random_state=cfg.random_state,
    )

    if embeddings_mpnet_train is None:
        embeddings_mpnet_train = encode_with_sbert(docs_train, MPNET_NAME)
    if embeddings_mpnet_test is None:
        embeddings_mpnet_test = encode_with_sbert(docs_test, MPNET_NAME)

    rows = [
        run_one(
            "robert-base",
            embeddings_robert_train,
            embeddings_robert_test,
            docs_train,
            docs_test,
            y_test,
            cfg,
        ),
        run_one(
            "mpnet-multilingual",
            embeddings_mpnet_train,
            embeddings_mpnet_test,
            docs_train,
            docs_test,
            y_test,
            cfg,
        ),
    ]
    return pd.DataFrame(rows)


def save_ablation(df: pd.DataFrame, out_path: Path | str) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return out_path
