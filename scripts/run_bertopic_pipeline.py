"""Run the full BERTopic pipeline (T1 -> T6) in order.

Useful once Mihai's hand-off lands so you can produce all artifacts in one go:

    python scripts/run_bertopic_pipeline.py

Each step is skipped automatically if its outputs already exist on disk.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"

PYTHON = sys.executable

STEPS = [
    ("T1: encode docs", "encode_docs.py"),
    ("T2: fit BERTopic", "fit_bertopic.py"),
    ("T3: inspect topics", "inspect_topics.py"),
    ("T4: hp sweep", "hp_sweep.py"),
    ("T5: embedding ablation", "embedding_ablation.py"),
    ("T6: evaluate on test", "evaluate_on_test.py"),
    ("T8: stability + bootstrap", "stability_analysis.py"),
]


def main() -> int:
    for label, script in STEPS:
        print(f"\n{'=' * 60}\n{label}\n{'=' * 60}")
        rc = subprocess.call([PYTHON, str(SCRIPTS / script)])
        if rc != 0:
            print(f"\n{label} failed (rc={rc}). Stopping pipeline.")
            return rc
    print("\nPipeline completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
