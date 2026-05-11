"""Regenerate presentation SVG diagrams from app/static/*.mmd."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "app" / "static"
MMDC = ["npx", "--yes", "@mermaid-js/mermaid-cli@10.9.0"]


def main() -> int:
    mmd_files = sorted(STATIC.glob("*.mmd"))
    if not mmd_files:
        print(f"No .mmd files found in {STATIC}", file=sys.stderr)
        return 1

    for mmd in mmd_files:
        out = mmd.with_suffix(".svg")
        cmd = [*MMDC, "-i", str(mmd), "-o", str(out)]
        print(" ".join(cmd))
        subprocess.run(cmd, check=True, cwd=ROOT)
        print(f"wrote {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
