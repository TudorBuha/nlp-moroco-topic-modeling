"""MOROCO loader. Owner: Mihai (Phase 1.3).

Expected output: a pandas DataFrame with columns
    id (str), text (str), dialect (str), topic (str)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

DEFAULT_RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"


def load_moroco(raw_dir: Path | str = DEFAULT_RAW_DIR) -> pd.DataFrame:
    """Load the MOROCO corpus from `raw_dir` into a DataFrame.

    TODO (Mihai, 1.3): implement. The exact reader depends on the MOROCO
    release format you download — adjust accordingly. Make sure UTF-8
    diacritics survive intact (ș, ț, ă, â, î).
    """
    raw_dir = Path(raw_dir)
    if not raw_dir.exists():
        raise FileNotFoundError(
            f"Raw data dir not found: {raw_dir}\n"
            "See data/raw/README.md for download instructions."
        )
    raise NotImplementedError("Mihai: implement MOROCO loader (Phase 1.3)")
