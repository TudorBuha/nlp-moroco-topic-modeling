"""Step T5 — Embedding ablation: RoBERT vs. multilingual MPNet.

Re-runs BERTopic with `paraphrase-multilingual-mpnet-base-v2` on the same
docs, and compares to the RoBERT version on coherence + NMI/Purity vs MOROCO
labels.

Saves results table to `results/bertopic/ablation.csv`.

Usage:
    python scripts/embedding_ablation.py
    python scripts/embedding_ablation.py --label-col topic
"""

from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401

import numpy as np
import pandas as pd

from src import paths
from src.bertopic.ablation import (
    encode_with_sbert,
    MPNET_NAME,
    run_ablation,
    save_ablation,
)
from src.bertopic.model import BERTopicConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--text-col", default="text")
    p.add_argument(
        "--label-col",
        default="topic",
        help="Column in test.parquet with the MOROCO ground-truth label.",
    )
    p.add_argument("--min-cluster-size", type=int, default=30)
    p.add_argument("--n-neighbors", type=int, default=15)
    return p.parse_args()


def _load_or_encode_mpnet(parquet_path, npy_path, text_col):
    if npy_path.exists():
        print(f"using cached MPNet embeddings: {npy_path}")
        return np.load(npy_path)
    docs = pd.read_parquet(parquet_path)[text_col].tolist()
    vecs = encode_with_sbert(docs, MPNET_NAME)
    npy_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(npy_path, vecs)
    print(f"saved MPNet embeddings -> {npy_path} ({vecs.shape})")
    return vecs


def main() -> int:
    args = parse_args()
    paths.ensure_dirs()

    for required in (
        paths.TRAIN_PARQUET,
        paths.TEST_PARQUET,
        paths.EMBEDDINGS_TRAIN,
        paths.EMBEDDINGS_TEST,
    ):
        if not required.exists():
            raise SystemExit(f"Missing {required}.")

    train_df = pd.read_parquet(paths.TRAIN_PARQUET)
    test_df = pd.read_parquet(paths.TEST_PARQUET)
    docs_train = train_df[args.text_col].tolist()
    docs_test = test_df[args.text_col].tolist()
    y_test = test_df[args.label_col].tolist()

    e_robert_train = np.load(paths.EMBEDDINGS_TRAIN)
    e_robert_test = np.load(paths.EMBEDDINGS_TEST)
    e_mpnet_train = _load_or_encode_mpnet(
        paths.TRAIN_PARQUET, paths.EMBEDDINGS_TRAIN_MPNET, args.text_col
    )
    e_mpnet_test = _load_or_encode_mpnet(
        paths.TEST_PARQUET, paths.EMBEDDINGS_TEST_MPNET, args.text_col
    )

    cfg = BERTopicConfig(
        min_cluster_size=args.min_cluster_size,
        n_neighbors=args.n_neighbors,
    )

    df = run_ablation(
        docs_train=docs_train,
        docs_test=docs_test,
        y_test=y_test,
        embeddings_robert_train=e_robert_train,
        embeddings_robert_test=e_robert_test,
        embeddings_mpnet_train=e_mpnet_train,
        embeddings_mpnet_test=e_mpnet_test,
        cfg=cfg,
    )
    print("\nAblation results:")
    print(df.to_string(index=False))

    out_path = save_ablation(df, paths.ABLATION_CSV)
    print(f"\nsaved -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
