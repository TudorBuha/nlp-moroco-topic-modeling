"""Step T3 — Print top keywords, save the topic table CSV, and write
the three required HTML visualizations.

Usage:
    python scripts/inspect_topics.py
    python scripts/inspect_topics.py --top-k 15
"""

from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401

from src import paths
from src.bertopic.inspection import save_topic_table, topic_keyword_table
from src.bertopic.training import load_model
from src.bertopic.visualizations import save_default_visualizations


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--top-k", type=int, default=10)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not paths.MODEL_DIR_MAIN.exists():
        raise SystemExit(
            f"Model not found at {paths.MODEL_DIR_MAIN}. "
            "Run scripts/fit_bertopic.py first."
        )

    model = load_model(paths.MODEL_DIR_MAIN)

    table = topic_keyword_table(model, top_k=args.top_k)
    print("\nTopic keyword table:")
    print(table.to_string(index=False))

    out_csv = save_topic_table(model, paths.TOPIC_TABLE_CSV, top_k=args.top_k)
    print(f"\nsaved topic table -> {out_csv}")

    saved = save_default_visualizations(model, paths.RESULTS_BERTOPIC)
    print(f"\nvisualizations: {sorted(saved.keys())}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
