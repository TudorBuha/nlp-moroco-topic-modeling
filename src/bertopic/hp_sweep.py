"""Hyperparameter sensitivity sweep (Tudor, Step T4).

Sweeps `min_cluster_size ∈ {10, 20, 30, 50}` and `n_neighbors ∈ {5, 15, 30}`
as required by the task spec, recording for each config:
  - number of topics discovered (excluding outliers)
  - outlier proportion
  - C_v coherence
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from .coherence import compute_cv_coherence
from .inspection import n_topics, outlier_proportion
from .model import BERTopicConfig
from .training import fit_bertopic

DEFAULT_MIN_CLUSTER_SIZES = (10, 20, 30, 50)
DEFAULT_N_NEIGHBORS = (5, 15, 30)


def sweep(
    docs: list[str],
    embeddings: np.ndarray,
    min_cluster_sizes: Iterable[int] = DEFAULT_MIN_CLUSTER_SIZES,
    n_neighbors_list: Iterable[int] = DEFAULT_N_NEIGHBORS,
    coherence_top_n: int = 10,
) -> pd.DataFrame:
    """Run BERTopic for the cartesian product of the two grids.

    Returns a DataFrame with one row per config. The 'best' row by
    coherence × (1 - outlier_pct) is convenient for picking T4 winner.
    """
    rows: list[dict] = []
    grid = [(mcs, nn) for mcs in min_cluster_sizes for nn in n_neighbors_list]

    for mcs, nn in tqdm(grid, desc="hp sweep"):
        cfg = BERTopicConfig(min_cluster_size=mcs, n_neighbors=nn)
        try:
            model, topics, _ = fit_bertopic(docs, embeddings, cfg)
            cv = compute_cv_coherence(model, docs, top_n=coherence_top_n)
            row = {
                "min_cluster_size": mcs,
                "n_neighbors": nn,
                "n_topics": n_topics(model, exclude_outlier=True),
                "outlier_pct": outlier_proportion(topics),
                "c_v": cv,
                "ok": True,
                "error": "",
            }
        except Exception as exc:  # pragma: no cover
            row = {
                "min_cluster_size": mcs,
                "n_neighbors": nn,
                "n_topics": -1,
                "outlier_pct": float("nan"),
                "c_v": float("nan"),
                "ok": False,
                "error": str(exc),
            }
        rows.append(row)

    df = pd.DataFrame(rows)
    df["score"] = df["c_v"] * (1.0 - df["outlier_pct"])
    return df.sort_values("score", ascending=False).reset_index(drop=True)


def save_sweep(df: pd.DataFrame, out_path: Path | str) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return out_path
