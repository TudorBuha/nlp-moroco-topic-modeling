"""Method A — Steps M1 + M2 + M3.5: build corpus, sweep K, save best LDA + viz.

Inputs:  data/processed/{train,test}.parquet (from `scripts/preprocess.py`)
Outputs: results/lda/dictionary.dict + train/test_corpus.mm + train/test_tokens.pkl
         results/lda/coherence_sweep.csv + coherence_curve.png
         results/lda/model_main.gensim
         results/lda/ldavis.html

Usage:
    python scripts/train_lda.py
    python scripts/train_lda.py --k-values 5 6 8 10 12 --passes 5
"""

from __future__ import annotations

import argparse
import sys

import _bootstrap  # noqa: F401
import pandas as pd

from src import paths
from src.lda.corpus import build_corpus, save_corpus
from src.lda.training import (
    DEFAULT_K_VALUES,
    save_lda,
    sweep_k,
    write_sweep_csv,
)
from src.lda.visualizations import save_coherence_curve, save_pyldavis_html


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--k-values",
        type=int,
        nargs="+",
        default=list(DEFAULT_K_VALUES),
        help="K values to sweep over (default: 5 6 8 10 12 15 20).",
    )
    p.add_argument("--passes", type=int, default=10)
    p.add_argument("--iterations", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--no-pyldavis",
        action="store_true",
        help="Skip the pyLDAvis HTML export (it can be slow on large corpora).",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    paths.ensure_dirs()

    print(f"[1/5] Loading splits from {paths.DATA_PROCESSED}…")
    if not paths.TRAIN_PARQUET.exists() or not paths.TEST_PARQUET.exists():
        print(
            "ERROR: train/test parquets not found. "
            "Run `python scripts/preprocess.py` first.",
            file=sys.stderr,
        )
        return 1
    train = pd.read_parquet(paths.TRAIN_PARQUET)
    test = pd.read_parquet(paths.TEST_PARQUET)
    print(f"        train={len(train):,}, test={len(test):,}")

    print("[2/5] Tokenizing + building Gensim Dictionary + BoW corpus "
          "(this is the slow part)…")
    bundle = build_corpus(train["text_clean"], test["text_clean"])
    print(f"        dictionary size: {len(bundle['dictionary']):,}")
    save_corpus(bundle, paths.RESULTS_LDA)
    print(f"        saved corpus artifacts -> {paths.RESULTS_LDA}")

    print(f"[3/5] Sweeping K over {args.k_values} (passes={args.passes}, "
          f"iterations={args.iterations})…")
    sweep_df, models = sweep_k(
        bundle["train_corpus"],
        bundle["train_tokens"],
        bundle["dictionary"],
        k_values=args.k_values,
        passes=args.passes,
        iterations=args.iterations,
        random_state=args.seed,
    )
    write_sweep_csv(sweep_df, paths.LDA_SWEEP_CSV)
    save_coherence_curve(sweep_df, paths.LDA_COHERENCE_PNG)
    print(f"        sweep CSV -> {paths.LDA_SWEEP_CSV}")
    print(f"        coherence curve -> {paths.LDA_COHERENCE_PNG}")

    best_k = int(sweep_df.iloc[0]["k"])
    print(f"[4/5] Best K = {best_k} (C_v = {sweep_df.iloc[0]['c_v']:.4f}). "
          f"Saving model…")
    best_model = models[best_k]
    save_lda(best_model, paths.LDA_MODEL_MAIN)
    print(f"        model -> {paths.LDA_MODEL_MAIN}")

    if args.no_pyldavis:
        print("[5/5] Skipping pyLDAvis (--no-pyldavis).")
    else:
        print("[5/5] Rendering pyLDAvis HTML…")
        save_pyldavis_html(
            best_model,
            bundle["train_corpus"],
            bundle["dictionary"],
            paths.LDA_LDAVIS_HTML,
        )
        print(f"        ldavis -> {paths.LDA_LDAVIS_HTML}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
