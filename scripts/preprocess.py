"""Phase 1 driver — load MOROCO, clean, optionally subset, then split.

Outputs:
    data/processed/train.parquet
    data/processed/test.parquet
    data/processed/dataset_stats.csv      (per-topic / per-dialect counts)

Usage:
    python scripts/preprocess.py                       # full corpus (~33k)
    python scripts/preprocess.py --max-per-class 1000  # demo subset (6k)
    python scripts/preprocess.py --max-per-class -1    # equivalent to "no cap"
"""

from __future__ import annotations

import argparse
import sys

import _bootstrap  # noqa: F401
import pandas as pd

from src import paths
from src.preprocessing.clean import clean_text
from src.preprocessing.load import load_moroco
from src.preprocessing.split import save_splits, train_test_split


def stratified_subset(
    df: pd.DataFrame,
    max_per_class: int,
    seed: int,
) -> pd.DataFrame:
    if max_per_class <= 0 or max_per_class >= df["topic"].value_counts().max():
        return df.reset_index(drop=True)
    parts: list[pd.DataFrame] = []
    for topic, group in df.groupby("topic", sort=False):
        n = min(len(group), max_per_class)
        parts.append(group.sample(n=n, random_state=seed))
    out = pd.concat(parts, ignore_index=True)
    return out.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--max-per-class",
        type=int,
        default=-1,
        help="Cap docs per topic class (default: no cap = full corpus). "
        "Use 1000 for the demo subset.",
    )
    p.add_argument(
        "--test-size", type=float, default=0.2, help="Proportion held out for test."
    )
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    paths.ensure_dirs()

    print(f"[1/4] Loading MOROCO from {paths.DATA_RAW}…")
    df = load_moroco(paths.DATA_RAW)
    print(f"        {len(df):,} samples, "
          f"{df['topic'].nunique()} topics, {df['dialect'].nunique()} dialects.")

    print(f"[2/4] Subsetting (max_per_class={args.max_per_class})…")
    df = stratified_subset(df, args.max_per_class, args.seed)
    print(f"        kept {len(df):,} samples after subsetting.")

    print("[3/4] Cleaning text (normalize diacritics, lowercase, "
          "strip punct/digits)…")
    df["text_clean"] = df["text"].astype(str).map(clean_text)
    df = df[df["text_clean"].str.split().str.len() >= 5].reset_index(drop=True)
    print(f"        {len(df):,} samples remain after dropping ones with <5 tokens.")

    print(f"[4/4] Stratified split (test_size={args.test_size}, "
          f"seed={args.seed})…")
    train, test = train_test_split(
        df, test_size=args.test_size, stratify_col="topic", random_state=args.seed
    )

    save_splits(train, test, paths.DATA_PROCESSED)

    stats = pd.DataFrame(
        {
            "train": train["topic"].value_counts(),
            "test": test["topic"].value_counts(),
        }
    ).fillna(0).astype(int).sort_index()
    stats["total"] = stats["train"] + stats["test"]
    stats_path = paths.DATA_PROCESSED / "dataset_stats.csv"
    stats.to_csv(stats_path)
    print(f"\nSaved dataset stats -> {stats_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
