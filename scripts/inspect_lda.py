"""Method A — Step M3: build the LDA topic-keyword table.

Saves results/lda/topic_keywords.csv. If the user has decided on manual labels,
they can be added here too via the same `attach_manual_labels` helper used by
the BERTopic side.
"""

from __future__ import annotations

import sys

import _bootstrap  # noqa: F401

from src import paths
from src.lda.gui_labels import DEFAULT_LDA_GUI_LABELS
from src.lda.inspection import (
    attach_manual_labels,
    save_topic_table,
    topic_keyword_table,
)
from src.lda.training import load_lda


def main() -> int:
    if not paths.LDA_MODEL_MAIN.exists():
        print(
            f"ERROR: {paths.LDA_MODEL_MAIN} not found. "
            "Run `python scripts/train_lda.py` first.",
            file=sys.stderr,
        )
        return 1

    model = load_lda(paths.LDA_MODEL_MAIN)
    table = topic_keyword_table(model, top_n=10)
    save_topic_table(table, paths.LDA_TOPIC_TABLE_CSV)
    save_topic_table(
        attach_manual_labels(table, labels=DEFAULT_LDA_GUI_LABELS),
        paths.LDA_TOPIC_TABLE_LABELED_CSV,
    )
    print(f"Saved {len(table)} topic rows -> {paths.LDA_TOPIC_TABLE_CSV}")
    print(f"Editable labeled copy   -> {paths.LDA_TOPIC_TABLE_LABELED_CSV}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
