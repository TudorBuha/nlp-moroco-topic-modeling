"""Method A — Step M4: evaluate LDA on the held-out test set.

Loads the saved LDA model + dictionary + test corpus and computes:
  - dominant-topic predictions for every test doc
  - NMI, Purity vs the gold MOROCO topics
  - confusion matrix (cluster id × gold label)

Outputs:
    results/lda/test_evaluation.json
    results/lda/test_confusion_matrix.csv
"""

from __future__ import annotations

import sys

import _bootstrap  # noqa: F401
import pandas as pd

from src import paths
from src.lda.corpus import load_corpus
from src.lda.evaluate import evaluate_lda_on_test, save_evaluation
from src.lda.training import load_lda


def main() -> int:
    if not paths.LDA_MODEL_MAIN.exists():
        print(
            f"ERROR: {paths.LDA_MODEL_MAIN} not found. "
            "Run `python scripts/train_lda.py` first.",
            file=sys.stderr,
        )
        return 1

    print("Loading model + corpus…")
    model = load_lda(paths.LDA_MODEL_MAIN)
    bundle = load_corpus(paths.RESULTS_LDA)
    if bundle.get("test_corpus") is None:
        print("ERROR: test corpus missing.", file=sys.stderr)
        return 2

    print("Loading gold test labels…")
    test = pd.read_parquet(paths.TEST_PARQUET)
    if len(test) != len(bundle["test_corpus"]):
        print(
            f"WARN: row count mismatch — parquet has {len(test)}, "
            f"corpus has {len(bundle['test_corpus'])}",
            file=sys.stderr,
        )

    print("Evaluating…")
    result = evaluate_lda_on_test(
        model, bundle["dictionary"], bundle["test_corpus"], test["topic"].tolist()
    )

    save_evaluation(
        result, paths.LDA_TEST_EVAL_JSON, paths.LDA_TEST_CONFUSION_CSV
    )
    print(f"NMI = {result['nmi_test']:.4f}")
    print(f"Purity = {result['purity_test']:.4f}")
    print(f"# topics = {result['n_topics']}")
    print(f"Empty BoW (skipped) = {result['n_empty_bow']}")
    print(f"-> {paths.LDA_TEST_EVAL_JSON}")
    print(f"-> {paths.LDA_TEST_CONFUSION_CSV}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
