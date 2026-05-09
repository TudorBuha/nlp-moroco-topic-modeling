"""Tests for `src.preprocessing.load`.

Builds a tiny synthetic MOROCO directory in `tmp_path` so the test does not
depend on the actual ~33k-row corpus being present.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.preprocessing.load import (
    CATEGORY_NAMES,
    DIALECT_NAMES,
    load_moroco,
)


def _make_split(split_dir: Path, rows: list[tuple[str, str, int, int]]) -> None:
    """Each row is `(id, text, dialect_int, category_int)`."""
    split_dir.mkdir(parents=True, exist_ok=True)
    with (split_dir / "samples.txt").open("w", encoding="utf-8") as f:
        for doc_id, text, _, _ in rows:
            f.write(f"{doc_id}\t{text}\n")
    with (split_dir / "dialect_labels.txt").open("w", encoding="utf-8") as f:
        for doc_id, _, dialect, _ in rows:
            f.write(f"{doc_id}\t{dialect}\n")
    with (split_dir / "category_labels.txt").open("w", encoding="utf-8") as f:
        for doc_id, _, _, category in rows:
            f.write(f"{doc_id}\t{category}\n")


def _make_corpus(root: Path) -> Path:
    base = root / "MOROCO" / "MOROCO" / "preprocessed"
    _make_split(
        base / "train",
        [("1", "text ro politics", 2, 3), ("2", "text md sports", 1, 5)],
    )
    _make_split(base / "validation", [("3", "validation tech", 2, 6)])
    _make_split(base / "test", [("4", "test culture", 1, 1)])
    return root


def test_load_moroco_concatenates_three_splits(tmp_path: Path):
    raw_dir = _make_corpus(tmp_path)
    df = load_moroco(raw_dir)

    assert len(df) == 4
    assert set(df.columns) >= {"id", "text", "dialect", "topic", "moroco_split"}
    assert set(df["moroco_split"]) == {"train", "validation", "test"}


def test_load_moroco_decodes_label_integers(tmp_path: Path):
    raw_dir = _make_corpus(tmp_path)
    df = load_moroco(raw_dir)
    by_id = {row["id"]: row for _, row in df.iterrows()}

    assert by_id["1"]["dialect"] == "ro"
    assert by_id["1"]["topic"] == "politics"
    assert by_id["2"]["dialect"] == "md"
    assert by_id["2"]["topic"] == "sports"
    assert by_id["3"]["topic"] == "tech"
    assert by_id["4"]["topic"] == "culture"


def test_load_moroco_strips_named_entity_markers(tmp_path: Path):
    base = tmp_path / "MOROCO" / "MOROCO" / "preprocessed"
    for split in ("train", "validation", "test"):
        _make_split(
            base / split,
            [(f"id{split}", "text $NE$ with marker", 2, 3)],
        )
    df = load_moroco(tmp_path)
    assert "$NE$" not in df["text"].iloc[0]


def test_load_moroco_raises_when_corpus_missing(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="MOROCO not found"):
        load_moroco(tmp_path / "does-not-exist")


def test_label_dicts_match_moroco_readme():
    """Mappings come from the upstream README; regression-pin them."""
    assert CATEGORY_NAMES == {
        1: "culture",
        2: "finance",
        3: "politics",
        4: "science",
        5: "sports",
        6: "tech",
    }
    assert DIALECT_NAMES == {1: "md", 2: "ro"}
