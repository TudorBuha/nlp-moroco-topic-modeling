"""Step T8 — Stability analysis: multi-seed runs + bootstrap CIs.

Runs the BERTopic pipeline N times with N different UMAP seeds (multi-seed
stability), then for the *fitted main model* computes percentile bootstrap
95% confidence intervals for NMI and Purity on the test split.

Saves:
  - results/bertopic/stability_runs.csv      (one row per seed)
  - results/bertopic/stability_summary.csv   (mean ± std per metric)
  - results/bertopic/bootstrap_nmi.json      (mean, std, 95% CI, samples)
  - results/bertopic/bootstrap_purity.json

Usage:
    python scripts/stability_analysis.py
    python scripts/stability_analysis.py --seeds 0 1 2 --n-bootstrap 500
    python scripts/stability_analysis.py --skip-multi-seed
    python scripts/stability_analysis.py --skip-bootstrap
"""

from __future__ import annotations

import argparse
import json

import _bootstrap  # noqa: F401

import numpy as np
import pandas as pd

from src import paths
from src.bertopic.evaluate import evaluate_on_test
from src.bertopic.model import BERTopicConfig
from src.bertopic.stability import (
    DEFAULT_SEEDS,
    run_multi_seed,
    save_stability,
    summarize,
)
from src.bertopic.training import load_model
from src.evaluation.metrics import bootstrap_metric, nmi_score, purity_score


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_SEEDS),
        help="Random seeds to use for multi-seed runs (default: 0..9).",
    )
    p.add_argument("--n-bootstrap", type=int, default=1000)
    p.add_argument("--bootstrap-seed", type=int, default=42)
    p.add_argument("--ci", type=float, default=0.95)
    p.add_argument(
        "--min-cluster-size",
        type=int,
        default=30,
        help="Base BERTopic config (matches scripts/fit_bertopic.py).",
    )
    p.add_argument("--n-neighbors", type=int, default=15)
    p.add_argument("--text-col", default="text")
    p.add_argument("--label-col", default="topic")
    p.add_argument("--skip-multi-seed", action="store_true")
    p.add_argument("--skip-bootstrap", action="store_true")
    return p.parse_args()


def _load_data(args):
    if not paths.TRAIN_PARQUET.exists() or not paths.EMBEDDINGS_TRAIN.exists():
        raise SystemExit("Need train.parquet + embeddings_train.npy.")
    train_df = pd.read_parquet(paths.TRAIN_PARQUET)
    docs_train = train_df[args.text_col].tolist()
    e_train = np.load(paths.EMBEDDINGS_TRAIN)

    docs_test = e_test = y_test = None
    if paths.TEST_PARQUET.exists() and paths.EMBEDDINGS_TEST.exists():
        test_df = pd.read_parquet(paths.TEST_PARQUET)
        docs_test = test_df[args.text_col].tolist()
        e_test = np.load(paths.EMBEDDINGS_TEST)
        if args.label_col in test_df.columns:
            y_test = test_df[args.label_col].tolist()

    return docs_train, e_train, docs_test, e_test, y_test


def _multi_seed(args, docs_train, e_train, docs_test, e_test, y_test):
    cfg = BERTopicConfig(
        min_cluster_size=args.min_cluster_size,
        n_neighbors=args.n_neighbors,
    )
    runs = run_multi_seed(
        docs_train=docs_train,
        embeddings_train=e_train,
        docs_test=docs_test,
        embeddings_test=e_test,
        y_test=y_test,
        seeds=args.seeds,
        base_cfg=cfg,
    )
    summary = summarize(runs)

    print("\nPer-run results:")
    print(runs.to_string(index=False))
    print("\nSummary (mean +/- std across runs):")
    print(summary.to_string(index=False))

    save_stability(
        runs,
        summary,
        out_runs=paths.STABILITY_RUNS_CSV,
        out_summary=paths.STABILITY_SUMMARY_CSV,
    )
    print(f"\nsaved -> {paths.STABILITY_RUNS_CSV}")
    print(f"saved -> {paths.STABILITY_SUMMARY_CSV}")


def _bootstrap(args, docs_test, e_test, y_test):
    if not paths.MODEL_DIR_MAIN.exists():
        raise SystemExit(
            f"Model not found at {paths.MODEL_DIR_MAIN}. "
            "Run scripts/fit_bertopic.py first."
        )
    if y_test is None:
        raise SystemExit("Need test.parquet with label column for bootstrap.")

    print("\nLoading saved model + computing test predictions once...")
    model = load_model(paths.MODEL_DIR_MAIN)
    metrics = evaluate_on_test(model, docs_test, e_test, y_test)
    topics_pred = metrics["topics_pred"]

    print(f"\nBootstrapping NMI with B={args.n_bootstrap}...")
    nmi_boot = bootstrap_metric(
        y_true=y_test,
        y_pred=topics_pred,
        metric_fn=nmi_score,
        n_bootstrap=args.n_bootstrap,
        ci=args.ci,
        random_state=args.bootstrap_seed,
    )
    print(
        f"NMI    mean={nmi_boot['mean']:.4f}  "
        f"std={nmi_boot['std']:.4f}  "
        f"{int(args.ci * 100)}% CI=[{nmi_boot['ci_low']:.4f}, "
        f"{nmi_boot['ci_high']:.4f}]"
    )

    print(f"\nBootstrapping Purity with B={args.n_bootstrap}...")
    purity_boot = bootstrap_metric(
        y_true=y_test,
        y_pred=topics_pred,
        metric_fn=purity_score,
        n_bootstrap=args.n_bootstrap,
        ci=args.ci,
        random_state=args.bootstrap_seed,
    )
    print(
        f"Purity mean={purity_boot['mean']:.4f}  "
        f"std={purity_boot['std']:.4f}  "
        f"{int(args.ci * 100)}% CI=[{purity_boot['ci_low']:.4f}, "
        f"{purity_boot['ci_high']:.4f}]"
    )

    paths.RESULTS_BERTOPIC.mkdir(parents=True, exist_ok=True)
    with paths.BOOTSTRAP_NMI_JSON.open("w", encoding="utf-8") as f:
        json.dump(nmi_boot, f, indent=2)
    with paths.BOOTSTRAP_PURITY_JSON.open("w", encoding="utf-8") as f:
        json.dump(purity_boot, f, indent=2)
    print(f"\nsaved -> {paths.BOOTSTRAP_NMI_JSON}")
    print(f"saved -> {paths.BOOTSTRAP_PURITY_JSON}")


def main() -> int:
    args = parse_args()
    paths.ensure_dirs()
    docs_train, e_train, docs_test, e_test, y_test = _load_data(args)

    if not args.skip_multi_seed:
        _multi_seed(args, docs_train, e_train, docs_test, e_test, y_test)

    if not args.skip_bootstrap:
        _bootstrap(args, docs_test, e_test, y_test)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
