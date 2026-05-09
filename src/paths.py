"""Shared filesystem paths. Importing this gives every module the same view
of where data and results live, so CLI scripts and the notebook agree."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA = ROOT / "data"
DATA_RAW = DATA / "raw"
DATA_PROCESSED = DATA / "processed"
TRAIN_PARQUET = DATA_PROCESSED / "train.parquet"
TEST_PARQUET = DATA_PROCESSED / "test.parquet"

RESULTS = ROOT / "results"
RESULTS_LDA = RESULTS / "lda"
RESULTS_BERTOPIC = RESULTS / "bertopic"

# BERTopic artifacts
EMBEDDINGS_TRAIN = RESULTS_BERTOPIC / "embeddings_train.npy"
EMBEDDINGS_TEST = RESULTS_BERTOPIC / "embeddings_test.npy"
EMBEDDINGS_TRAIN_MPNET = RESULTS_BERTOPIC / "embeddings_train_mpnet.npy"
EMBEDDINGS_TEST_MPNET = RESULTS_BERTOPIC / "embeddings_test_mpnet.npy"
MODEL_DIR_MAIN = RESULTS_BERTOPIC / "model_main"
MODEL_DIR_MPNET = RESULTS_BERTOPIC / "model_mpnet"
TOPIC_TABLE_CSV = RESULTS_BERTOPIC / "topic_keywords.csv"
HP_SWEEP_CSV = RESULTS_BERTOPIC / "hp_sweep.csv"
ABLATION_CSV = RESULTS_BERTOPIC / "ablation.csv"
TEST_EVAL_JSON = RESULTS_BERTOPIC / "test_evaluation.json"
TEST_CONFUSION_CSV = RESULTS_BERTOPIC / "test_confusion_matrix.csv"

# Stability / bootstrap (Step T8)
STABILITY_RUNS_CSV = RESULTS_BERTOPIC / "stability_runs.csv"
STABILITY_SUMMARY_CSV = RESULTS_BERTOPIC / "stability_summary.csv"
BOOTSTRAP_NMI_JSON = RESULTS_BERTOPIC / "bootstrap_nmi.json"
BOOTSTRAP_PURITY_JSON = RESULTS_BERTOPIC / "bootstrap_purity.json"


def ensure_dirs() -> None:
    """Create all output directories if missing (idempotent)."""
    for d in (DATA_RAW, DATA_PROCESSED, RESULTS_LDA, RESULTS_BERTOPIC):
        d.mkdir(parents=True, exist_ok=True)
