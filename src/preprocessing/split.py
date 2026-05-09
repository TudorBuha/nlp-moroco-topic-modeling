"""Train/test split with stratification on topic (Phase 1.6)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split as _sk_split

DEFAULT_PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"


def train_test_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    stratify_col: str = "topic",
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified train/test split returning two DataFrames."""
    train, test = _sk_split(
        df,
        test_size=test_size,
        stratify=df[stratify_col],
        random_state=random_state,
    )
    return train.reset_index(drop=True), test.reset_index(drop=True)


def save_splits(
    train: pd.DataFrame,
    test: pd.DataFrame,
    out_dir: Path | str = DEFAULT_PROCESSED_DIR,
) -> None:
    """Persist splits as parquet and print per-topic counts."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_path = out_dir / "train.parquet"
    test_path = out_dir / "test.parquet"
    train.to_parquet(train_path, index=False)
    test.to_parquet(test_path, index=False)

    print(f"Saved {len(train):>6,} rows -> {train_path}")
    print(f"Saved {len(test):>6,} rows -> {test_path}")
    print("\nPer-topic counts (train | test):")
    counts = pd.DataFrame(
        {
            "train": train["topic"].value_counts(),
            "test": test["topic"].value_counts(),
        }
    ).fillna(0).astype(int)
    print(counts)
