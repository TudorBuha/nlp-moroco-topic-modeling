"""Step T1 — Encode train + test documents with `readerbench/robert-base`.

Reads `data/processed/{train,test}.parquet` (produced by `scripts/preprocess.py`)
and writes `results/bertopic/embeddings_{train,test}.npy`.

Usage:
    python scripts/encode_docs.py
    python scripts/encode_docs.py --batch-size 16 --max-length 256
    python scripts/encode_docs.py --skip-test
"""

from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401  (must come before src.* imports)

import pandas as pd

from src import paths
from src.bertopic.embeddings import (
    DEFAULT_MAX_LENGTH,
    DEFAULT_MODEL,
    encode_documents,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=DEFAULT_MODEL, help="HF model id")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    p.add_argument("--text-col", default="text")
    p.add_argument("--skip-train", action="store_true")
    p.add_argument("--skip-test", action="store_true")
    return p.parse_args()


def _encode_split(parquet_path, out_path, text_col, model, batch_size, max_length):
    if not parquet_path.exists():
        raise SystemExit(
            f"Missing {parquet_path}.\n"
            "Run `python scripts/preprocess.py` first."
        )
    df = pd.read_parquet(parquet_path)
    print(f"loaded {len(df):,} rows from {parquet_path}")
    encode_documents(
        docs=df[text_col].tolist(),
        out_path=out_path,
        model_name=model,
        batch_size=batch_size,
        max_length=max_length,
    )


def main() -> int:
    args = parse_args()
    paths.ensure_dirs()

    if not args.skip_train:
        _encode_split(
            paths.TRAIN_PARQUET,
            paths.EMBEDDINGS_TRAIN,
            args.text_col,
            args.model,
            args.batch_size,
            args.max_length,
        )

    if not args.skip_test:
        _encode_split(
            paths.TEST_PARQUET,
            paths.EMBEDDINGS_TEST,
            args.text_col,
            args.model,
            args.batch_size,
            args.max_length,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
