"""Shared evaluation metrics — used by both LDA and BERTopic."""

from .metrics import (
    bootstrap_metric,
    build_confusion_matrix,
    nmi_score,
    purity_score,
)

__all__ = [
    "purity_score",
    "nmi_score",
    "build_confusion_matrix",
    "bootstrap_metric",
]
