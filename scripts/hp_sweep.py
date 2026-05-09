"""Step T4 — BERTopic hyperparameter sensitivity sweep.

Runs the cartesian product of {min_cluster_size} x {n_neighbors} on the
TRAINING set with cached embeddings, and saves the result table to
`results/bertopic/hp_sweep.csv`.

Usage:
    python scripts/hp_sweep.py
    python scripts/hp_sweep.py --min-cluster-sizes 20 30 50 --n-neighbors 5 15
"""

from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401

import numpy as np
import pandas as pd

from src import paths
from src.bertopic.hp_sweep import (
    DEFAULT_MIN_CLUSTER_SIZES,
    DEFAULT_N_NEIGHBORS,
    save_sweep,
    sweep,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--min-cluster-sizes",
        type=int,
        nargs="+",
        default=list(DEFAULT_MIN_CLUSTER_SIZES),
    )
    p.add_argument(
        "--n-neighbors",
        type=int,
        nargs="+",
        default=list(DEFAULT_N_NEIGHBORS),
    )
    p.add_argument("--text-col", default="text")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    paths.ensure_dirs()

    if not paths.TRAIN_PARQUET.exists() or not paths.EMBEDDINGS_TRAIN.exists():
        raise SystemExit(
            "Need train.parquet + embeddings_train.npy before running the sweep."
        )

    docs = pd.read_parquet(paths.TRAIN_PARQUET)[args.text_col].tolist()
    embeddings = np.load(paths.EMBEDDINGS_TRAIN)

    df = sweep(
        docs=docs,
        embeddings=embeddings,
        min_cluster_sizes=args.min_cluster_sizes,
        n_neighbors_list=args.n_neighbors,
    )
    print("\nResults (sorted by score = c_v * (1 - outlier_pct)):")
    print(df.to_string(index=False))

    out_path = save_sweep(df, paths.HP_SWEEP_CSV)
    print(f"\nsaved -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
