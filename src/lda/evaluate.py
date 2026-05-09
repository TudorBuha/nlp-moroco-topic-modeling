"""LDA evaluation against the MOROCO test set.

Topic ↔ category alignment is computed by majority-voting each LDA topic to
the most frequent gold category among documents whose dominant topic is that
LDA topic. We then report standard external metrics (NMI, Purity) plus a
confusion matrix.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from gensim.corpora import Dictionary
from gensim.models import LdaModel

from src.evaluation.metrics import build_confusion_matrix, nmi_score, purity_score


def predict_dominant_topics(
    model: LdaModel,
    corpus: Sequence,
) -> np.ndarray:
    """Return an array of dominant topic ids (highest prob per doc)."""
    out = np.empty(len(corpus), dtype=np.int64)
    for i, bow in enumerate(corpus):
        if not bow:
            out[i] = -1
            continue
        topics = model.get_document_topics(bow, minimum_probability=0.0)
        out[i] = int(max(topics, key=lambda x: x[1])[0])
    return out


def evaluate_lda_on_test(
    model: LdaModel,
    dictionary: Dictionary,
    test_corpus: Sequence,
    test_labels: Sequence[str],
) -> dict:
    """Return NMI, Purity, n_topics, predicted-vs-true labels.

    Documents whose BoW is empty (every token filtered out) get topic id `-1`
    in the predictions; they're excluded from NMI/Purity to avoid biasing the
    score. They are still counted in the confusion matrix as an "empty" row.
    """
    preds = predict_dominant_topics(model, test_corpus)
    mask = preds >= 0
    keep = preds[mask]
    truth = np.asarray(test_labels)[mask]

    return {
        "n_topics": int(model.num_topics),
        "n_test": int(len(test_corpus)),
        "n_evaluated": int(mask.sum()),
        "n_empty_bow": int((~mask).sum()),
        "nmi_test": float(nmi_score(truth, keep)),
        "purity_test": float(purity_score(truth, keep)),
        "predictions": preds.tolist(),
        "labels": list(test_labels),
        "_dict_size": int(len(dictionary)),
    }


def save_evaluation(
    result: dict,
    json_path: str | Path,
    confusion_csv_path: str | Path,
) -> None:
    json_path = Path(json_path)
    confusion_csv_path = Path(confusion_csv_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    confusion_csv_path.parent.mkdir(parents=True, exist_ok=True)

    preds = np.asarray(result["predictions"])
    labels = np.asarray(result["labels"])
    mask = preds >= 0
    cm = build_confusion_matrix(labels[mask], preds[mask])
    pd.DataFrame(cm).to_csv(confusion_csv_path)

    payload = {k: v for k, v in result.items() if k not in ("predictions", "labels")}
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
