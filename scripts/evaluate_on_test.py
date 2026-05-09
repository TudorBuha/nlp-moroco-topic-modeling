"""Step T6 — Test-set evaluation of the chosen BERTopic model.

Computes NMI, Purity, outlier proportion, and writes:
  - `results/bertopic/test_evaluation.json`  (scalar metrics)
  - `results/bertopic/test_confusion_matrix.csv` (cluster_id x true_label)

Usage:
    python scripts/evaluate_on_test.py
    python scripts/evaluate_on_test.py --label-col topic
"""

from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401

import numpy as np
import pandas as pd

from src import paths
from src.bertopic.evaluate import evaluate_on_test, save_evaluation
from src.bertopic.training import load_model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--text-col", default="text")
    p.add_argument("--label-col", default="topic")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    paths.ensure_dirs()

    if not paths.MODEL_DIR_MAIN.exists():
        raise SystemExit(
            f"Model not found at {paths.MODEL_DIR_MAIN}. "
            "Run scripts/fit_bertopic.py first."
        )
    if not paths.TEST_PARQUET.exists() or not paths.EMBEDDINGS_TEST.exists():
        raise SystemExit("Need test.parquet + embeddings_test.npy.")

    test_df = pd.read_parquet(paths.TEST_PARQUET)
    docs_test = test_df[args.text_col].tolist()
    y_test = test_df[args.label_col].tolist()
    embeddings_test = np.load(paths.EMBEDDINGS_TEST)

    print(f"loaded {len(docs_test):,} test docs, embeddings {embeddings_test.shape}")

    model = load_model(paths.MODEL_DIR_MAIN)
    metrics = evaluate_on_test(model, docs_test, embeddings_test, y_test)

    print("\nTest metrics:")
    for k, v in metrics.items():
        if k == "topics_pred":
            continue
        print(f"  {k:>20s} = {v}")

    json_path, csv_path = save_evaluation(
        metrics,
        y_test,
        out_json=paths.TEST_EVAL_JSON,
        out_confusion_csv=paths.TEST_CONFUSION_CSV,
    )
    print(f"\nsaved -> {json_path}")
    print(f"saved -> {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
