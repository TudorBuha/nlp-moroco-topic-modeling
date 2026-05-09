"""Test-set evaluation for the chosen BERTopic model (Tudor, Step T6)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from ..evaluation.metrics import (
    build_confusion_matrix,
    nmi_score,
    purity_score,
)
from .inspection import outlier_proportion


def evaluate_on_test(
    topic_model,
    docs_test: Sequence[str],
    embeddings_test: np.ndarray,
    y_test: Sequence,
) -> dict:
    """Run inference on the test split and compute alignment metrics.

    Returns a dict with NMI, Purity, outlier_pct_test, plus the predicted
    topic ids in `topics_pred` (so the caller can save a confusion matrix).
    """
    topics_pred, _ = topic_model.transform(list(docs_test), embeddings_test)
    return {
        "n_test_docs": int(len(y_test)),
        "outlier_pct_test": outlier_proportion(topics_pred),
        "nmi_test": nmi_score(y_test, topics_pred),
        "purity_test": purity_score(y_test, topics_pred),
        "topics_pred": [int(t) for t in topics_pred],
    }


def save_evaluation(
    metrics: dict,
    y_test: Sequence,
    out_json: Path | str,
    out_confusion_csv: Path | str,
) -> tuple[Path, Path]:
    """Persist a JSON of scalar metrics + a CSV confusion matrix."""
    out_json = Path(out_json)
    out_confusion_csv = Path(out_confusion_csv)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    cm = build_confusion_matrix(y_test, metrics["topics_pred"])
    cm.to_csv(out_confusion_csv)

    serializable = {k: v for k, v in metrics.items() if k != "topics_pred"}
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2)

    return out_json, out_confusion_csv


def confusion_matrix_df(
    y_true: Sequence, topics_pred: Sequence[int]
) -> pd.DataFrame:
    return build_confusion_matrix(y_true, topics_pred)
