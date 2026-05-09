"""Topic-vs-ground-truth metrics shared between LDA and BERTopic.

References:
- Manning et al., Introduction to Information Retrieval, Ch. 16 (Purity)
- sklearn `normalized_mutual_info_score`
- Efron & Tibshirani (1993), An Introduction to the Bootstrap (percentile CI)
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import normalized_mutual_info_score


def purity_score(y_true: Sequence, y_pred: Sequence) -> float:
    """Fraction of correctly assigned documents under the best label-cluster
    matching, i.e. each cluster takes the majority gold label."""
    cm = pd.crosstab(pd.Series(y_pred, name="cluster"), pd.Series(y_true, name="label"))
    return float(cm.max(axis=1).sum() / cm.values.sum())


def nmi_score(y_true: Sequence, y_pred: Sequence) -> float:
    """Normalized mutual information (arithmetic average)."""
    return float(normalized_mutual_info_score(y_true, y_pred))


def build_confusion_matrix(
    y_true: Sequence,
    y_pred: Sequence,
    drop_outliers: bool = False,
) -> pd.DataFrame:
    """Cluster-id (rows) × true-label (cols) matrix.

    If `drop_outliers` is True, the BERTopic outlier cluster (-1) is removed.
    """
    cm = pd.crosstab(
        pd.Series(y_pred, name="cluster"),
        pd.Series(y_true, name="label"),
    )
    if drop_outliers and -1 in cm.index:
        cm = cm.drop(index=-1)
    return cm


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals
# ---------------------------------------------------------------------------


def bootstrap_metric(
    y_true: Sequence,
    y_pred: Sequence,
    metric_fn: Callable[[Sequence, Sequence], float],
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    random_state: int | None = 42,
) -> dict[str, float | list[float]]:
    """Percentile bootstrap CI for a clustering / classification metric.

    Resamples `(y_true, y_pred)` indices with replacement `n_bootstrap` times
    and recomputes `metric_fn(y_true_resampled, y_pred_resampled)` each time.

    Returns a dict with `mean`, `std`, `ci_low`, `ci_high`, plus the full
    `samples` list for plotting.

    Parameters
    ----------
    y_true, y_pred
        Aligned sequences of length N (ground-truth labels and predicted
        cluster ids).
    metric_fn
        Any callable `(y_true, y_pred) -> float`, e.g. `nmi_score`.
    n_bootstrap
        Number of bootstrap resamples. 1000 is the textbook default; 500 is
        often enough for a stable percentile CI.
    ci
        Confidence level in (0, 1), e.g. 0.95 for a 95% CI.
    random_state
        Seed for reproducibility. Pass None to use the global RNG.
    """
    if not 0.0 < ci < 1.0:
        raise ValueError(f"ci must be in (0, 1), got {ci}")
    y_true_arr = np.asarray(list(y_true))
    y_pred_arr = np.asarray(list(y_pred))
    if y_true_arr.shape[0] != y_pred_arr.shape[0]:
        raise ValueError(
            f"y_true and y_pred must align: got {y_true_arr.shape[0]} vs "
            f"{y_pred_arr.shape[0]}"
        )

    n = y_true_arr.shape[0]
    rng = np.random.default_rng(random_state)
    samples: list[float] = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        try:
            samples.append(float(metric_fn(y_true_arr[idx], y_pred_arr[idx])))
        except ValueError:
            # Some metrics (e.g. NMI) can fail on degenerate resamples
            # (e.g. all labels identical). Skip and continue.
            continue

    if not samples:
        raise RuntimeError("All bootstrap resamples failed to produce a metric.")

    arr = np.asarray(samples, dtype=np.float64)
    alpha = 1.0 - ci
    ci_low, ci_high = np.quantile(arr, [alpha / 2.0, 1.0 - alpha / 2.0])

    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "n_bootstrap": int(arr.size),
        "ci": float(ci),
        "samples": samples,
    }
