"""Topic-vs-ground-truth metrics shared between LDA and BERTopic.

References:
- Manning et al., Introduction to Information Retrieval, Ch. 16 (Purity)
- sklearn `normalized_mutual_info_score`
"""

from __future__ import annotations

from typing import Sequence

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
