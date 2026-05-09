"""Step T2 — Fit BERTopic on the training set with the chosen config.

Reads `train.parquet` + `embeddings_train.npy`, fits BERTopic,
saves the model under `results/bertopic/model_main/`.

Usage:
    python scripts/fit_bertopic.py
    python scripts/fit_bertopic.py --min-cluster-size 20 --n-neighbors 15
"""

from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401

import numpy as np
import pandas as pd

from src import paths
from src.bertopic.model import BERTopicConfig
from src.bertopic.training import fit_bertopic, save_model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--min-cluster-size", type=int, default=30)
    p.add_argument("--n-neighbors", type=int, default=15)
    p.add_argument("--n-components", type=int, default=5)
    p.add_argument("--text-col", default="text")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    paths.ensure_dirs()

    if not paths.TRAIN_PARQUET.exists():
        raise SystemExit(
            f"Missing {paths.TRAIN_PARQUET} — Mihai's hand-off not ready."
        )
    if not paths.EMBEDDINGS_TRAIN.exists():
        raise SystemExit(
            f"Missing {paths.EMBEDDINGS_TRAIN} — run scripts/encode_docs.py first."
        )

    docs = pd.read_parquet(paths.TRAIN_PARQUET)[args.text_col].tolist()
    embeddings = np.load(paths.EMBEDDINGS_TRAIN)
    print(f"loaded {len(docs):,} docs and embeddings {embeddings.shape}")

    cfg = BERTopicConfig(
        min_cluster_size=args.min_cluster_size,
        n_neighbors=args.n_neighbors,
        n_components=args.n_components,
    )
    model, topics, _ = fit_bertopic(docs, embeddings, cfg)
    n_outliers = sum(1 for t in topics if t == -1)
    print(
        f"fit complete: {len(set(topics))} unique topic ids, "
        f"{n_outliers:,} outliers ({n_outliers / len(topics):.1%})"
    )

    save_model(model, paths.MODEL_DIR_MAIN)
    print(f"saved model -> {paths.MODEL_DIR_MAIN}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
