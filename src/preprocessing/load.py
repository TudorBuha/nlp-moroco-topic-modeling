"""MOROCO loader (Phase 1.3).

The official MOROCO release ships pre-tokenized train/validation/test splits as
plain text under (note the nested `MOROCO/MOROCO/...` after cloning):

    data/raw/MOROCO/MOROCO/preprocessed/{train,validation,test}/
        samples.txt          # "<id>\\t<text>" per line
        dialect_labels.txt   # "<id>\\t<int>" per line — 0=Romanian, 1=Moldavian
        category_labels.txt  # "<id>\\t<int>" per line — 0..5 (see CATEGORY_NAMES)

`load_moroco` concatenates all three splits into a single DataFrame so we can
re-split with our own stratified `train_test_split` (deterministic on seed 42).
Labels are joined to samples by doc id so any line ordering differences in the
upstream files don't silently misalign the labels with the texts.

Output schema:
    id (str)        - MOROCO doc id
    text (str)      - raw sample (UTF-8, diacritics preserved)
    dialect (str)   - "ro" | "md"
    topic (str)     - "culture" | "finance" | "politics" | "science" | "sports" | "tech"
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

DEFAULT_RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"

# Mapping from the MOROCO README (1-indexed for both dialect and category):
#   dialect_labels.txt — 1 => Moldavian, 2 => Romanian
#   category_labels.txt — 1 culture, 2 finance, 3 politics, 4 science, 5 sports, 6 tech
CATEGORY_NAMES: dict[int, str] = {
    1: "culture",
    2: "finance",
    3: "politics",
    4: "science",
    5: "sports",
    6: "tech",
}

DIALECT_NAMES: dict[int, str] = {
    1: "md",  # Moldavian
    2: "ro",  # Romanian
}

_SPLITS = ("train", "validation", "test")


def _read_id_value_file(path: Path) -> dict[str, str]:
    """Parse a `<id>\\t<value>` file into a dict, ignoring blank lines."""
    out: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if not line:
                continue
            doc_id, _, value = line.partition("\t")
            out[doc_id] = value
    return out


def _read_split(split_dir: Path) -> pd.DataFrame:
    samples_path = split_dir / "samples.txt"
    dialects_path = split_dir / "dialect_labels.txt"
    categories_path = split_dir / "category_labels.txt"

    for p in (samples_path, dialects_path, categories_path):
        if not p.exists():
            raise FileNotFoundError(f"Missing MOROCO file: {p}")

    samples = _read_id_value_file(samples_path)
    dialects_raw = _read_id_value_file(dialects_path)
    categories_raw = _read_id_value_file(categories_path)

    rows: list[dict] = []
    for doc_id, text in samples.items():
        if doc_id not in dialects_raw or doc_id not in categories_raw:
            continue
        rows.append(
            {
                "id": doc_id,
                "text": text,
                "dialect": DIALECT_NAMES[int(dialects_raw[doc_id])],
                "topic": CATEGORY_NAMES[int(categories_raw[doc_id])],
                "moroco_split": split_dir.name,
            }
        )

    return pd.DataFrame(rows)


def load_moroco(raw_dir: Path | str = DEFAULT_RAW_DIR) -> pd.DataFrame:
    """Load the MOROCO corpus into a DataFrame.

    Concatenates MOROCO's own train/validation/test splits into one frame
    (~33k rows). We re-split with our own stratified seed downstream so the
    LDA and BERTopic comparisons share the same evaluation set.
    """
    raw_dir = Path(raw_dir)
    # The repo clones into `MOROCO/MOROCO/...`; older manual extracts have
    # `MOROCO/preprocessed/...`. Try both.
    candidates = [
        raw_dir / "MOROCO" / "MOROCO" / "preprocessed",
        raw_dir / "MOROCO" / "preprocessed",
    ]
    base = next((c for c in candidates if c.exists()), None)
    if base is None:
        raise FileNotFoundError(
            f"MOROCO not found under any of {[str(c) for c in candidates]}.\n"
            "Run `python scripts/download_moroco.py` first."
        )

    frames = [_read_split(base / s) for s in _SPLITS]
    df = pd.concat(frames, ignore_index=True)

    # The MOROCO samples are pre-tokenized and use a special token
    # `$NE$` for named entities; turn it into a single space so it doesn't
    # poison the vocabulary.
    df["text"] = df["text"].str.replace("$NE$", " ", regex=False)

    return df
