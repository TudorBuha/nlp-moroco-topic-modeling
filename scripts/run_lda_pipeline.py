"""Run the full LDA pipeline (M1 → M2 → M3 → M3.5 → M4) end-to-end.

This is the LDA equivalent of `run_bertopic_pipeline.py`. It does NOT run the
preprocessing step — call `python scripts/preprocess.py` first so that LDA and
BERTopic share the exact same train/test split.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable

STEPS = [
    ("Train LDA + sweep K + pyLDAvis", "scripts/train_lda.py"),
    ("Topic-keyword table", "scripts/inspect_lda.py"),
    ("Evaluate on test", "scripts/evaluate_lda.py"),
]


def main() -> int:
    for label, script in STEPS:
        print(f"\n=== {label} ({script}) ===")
        rc = subprocess.run([PY, str(ROOT / script)], cwd=ROOT).returncode
        if rc != 0:
            print(f"FAILED at {script} (exit {rc})", file=sys.stderr)
            return rc
    print("\nLDA pipeline complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
