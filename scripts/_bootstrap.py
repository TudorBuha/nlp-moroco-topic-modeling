"""Tiny helper imported by every script under `scripts/` so that the
project root is on `sys.path` and `from src.bertopic import ...` works."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
