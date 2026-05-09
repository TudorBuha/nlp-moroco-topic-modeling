"""Multi-seed stability analysis for BERTopic.

Re-fits BERTopic N times with different UMAP random seeds (UMAP is the
main source of stochasticity in the pipeline) on the same train embeddings,
and aggregates per-run metrics so we can report mean ± std for the report.

Why this matters: with a single random seed you can't tell whether a "good"
NMI is reproducible or just one lucky run. The course recommendation to
"run multiple times" is operationalized here.

Pairs nicely with `src.evaluation.metrics.bootstrap_metric`, which addresses
a different question (CI on a single test number) — see the report for
how to interpret each.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from ..evaluation.metrics import nmi_score, purity_score
from .coherence import compute_cv_coherence
from .inspection import n_topics, outlier_proportion
from .model import BERTopicConfig
from .training import fit_bertopic

DEFAULT_SEEDS: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)


def run_multi_seed(
    docs_train: list[str],
    embeddings_train: np.ndarray,
    docs_test: Sequence[str] | None = None,
    embeddings_test: np.ndarray | None = None,
    y_test: Sequence | None = None,
    seeds: Iterable[int] = DEFAULT_SEEDS,
    base_cfg: BERTopicConfig | None = None,
    coherence_top_n: int = 10,
) -> pd.DataFrame:
    """Fit BERTopic once per seed, return per-run metrics.

    Returns a DataFrame with columns:
        seed, n_topics, outlier_pct_train, c_v_train,
        outlier_pct_test, nmi_test, purity_test (NaN if test data missing).

    Test metrics are only filled in when `docs_test`, `embeddings_test`, and
    `y_test` are all supplied.
    """
    base_cfg = base_cfg or BERTopicConfig()

    has_test = (
        docs_test is not None
        and embeddings_test is not None
        and y_test is not None
    )

    rows: list[dict] = []
    seeds = list(seeds)
    for seed in tqdm(seeds, desc="multi-seed"):
        cfg = replace(base_cfg, random_state=seed)
        try:
            model, topics_train, _ = fit_bertopic(docs_train, embeddings_train, cfg)
        except Exception as exc:  # pragma: no cover
            rows.append(
                {
                    "seed": seed,
                    "n_topics": -1,
                    "outlier_pct_train": float("nan"),
                    "c_v_train": float("nan"),
                    "outlier_pct_test": float("nan"),
                    "nmi_test": float("nan"),
                    "purity_test": float("nan"),
                    "ok": False,
                    "error": str(exc),
                }
            )
            continue

        row: dict = {
            "seed": seed,
            "n_topics": n_topics(model, exclude_outlier=True),
            "outlier_pct_train": outlier_proportion(topics_train),
            "c_v_train": compute_cv_coherence(model, docs_train, top_n=coherence_top_n),
            "outlier_pct_test": float("nan"),
            "nmi_test": float("nan"),
            "purity_test": float("nan"),
            "ok": True,
            "error": "",
        }

        if has_test:
            topics_pred, _ = model.transform(list(docs_test), embeddings_test)
            row["outlier_pct_test"] = outlier_proportion(topics_pred)
            row["nmi_test"] = nmi_score(y_test, topics_pred)
            row["purity_test"] = purity_score(y_test, topics_pred)

        rows.append(row)

    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Return a 1-column-per-metric mean ± std summary for the report."""
    metric_cols = [
        "n_topics",
        "outlier_pct_train",
        "c_v_train",
        "outlier_pct_test",
        "nmi_test",
        "purity_test",
    ]
    available = [c for c in metric_cols if c in df.columns and df[c].notna().any()]

    rows: list[dict] = []
    for col in available:
        s = df[col].dropna()
        rows.append(
            {
                "metric": col,
                "mean": float(s.mean()),
                "std": float(s.std(ddof=1)) if len(s) > 1 else 0.0,
                "min": float(s.min()),
                "max": float(s.max()),
                "n_runs": int(len(s)),
            }
        )
    return pd.DataFrame(rows)


def save_stability(
    runs: pd.DataFrame,
    summary: pd.DataFrame,
    out_runs: Path | str,
    out_summary: Path | str,
) -> tuple[Path, Path]:
    out_runs = Path(out_runs)
    out_summary = Path(out_summary)
    out_runs.parent.mkdir(parents=True, exist_ok=True)
    runs.to_csv(out_runs, index=False)
    summary.to_csv(out_summary, index=False)
    return out_runs, out_summary
