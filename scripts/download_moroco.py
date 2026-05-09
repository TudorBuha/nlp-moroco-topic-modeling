"""Download the MOROCO corpus into `data/raw/MOROCO/`.

The simplest reliable route is to shallow-clone the official repo, which
contains the preprocessed train/validation/test splits as plain text:

    butnaruandrei/MOROCO
      preprocessed/{train,validation,test}/samples.txt
      preprocessed/{train,validation,test}/dialect_labels.txt
      preprocessed/{train,validation,test}/category_labels.txt

Run:

    python scripts/download_moroco.py
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import _bootstrap  # noqa: F401

from src import paths

REPO_URL = "https://github.com/butnaruandrei/MOROCO.git"
TARGET = paths.DATA_RAW / "MOROCO"


def main() -> None:
    paths.ensure_dirs()

    if TARGET.exists():
        marker = TARGET / "preprocessed" / "train" / "samples.txt"
        if marker.exists():
            print(f"MOROCO already downloaded at {TARGET}")
            return
        print(f"{TARGET} exists but seems incomplete; re-cloning…")
        shutil.rmtree(TARGET)

    print(f"Cloning {REPO_URL} -> {TARGET} (shallow)…")
    try:
        subprocess.run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                REPO_URL,
                str(TARGET),
            ],
            check=True,
        )
    except FileNotFoundError:
        print(
            "git is not on PATH. Install Git for Windows from https://git-scm.com "
            "or download the MOROCO repo manually as a zip and extract it under "
            "`data/raw/MOROCO/`.",
            file=sys.stderr,
        )
        sys.exit(1)

    marker = TARGET / "preprocessed" / "train" / "samples.txt"
    if not marker.exists():
        print(
            f"Clone finished but the expected file {marker} is missing.\n"
            "The MOROCO repo layout may have changed.",
            file=sys.stderr,
        )
        sys.exit(2)

    print(f"Done. MOROCO is now under {TARGET}")


if __name__ == "__main__":
    main()
