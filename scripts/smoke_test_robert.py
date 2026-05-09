"""Phase 1.C smoke test.

Encodes 10 Romanian sentences with `readerbench/robert-base` and verifies
that the output has shape [10, 768]. Run after `pip install -r requirements.txt`:

    python scripts/smoke_test_robert.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make `src` importable when running this script directly.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.bertopic.embeddings import RomanianEmbedder  # noqa: E402

SENTENCES = [
    "Bună ziua! Cum vă merge astăzi?",
    "Echipa națională a câștigat meciul de aseară cu 3-1.",
    "Banca Națională a anunțat o nouă rată a dobânzii.",
    "Cercetătorii au descoperit o nouă specie de pasăre în Carpați.",
    "Filmul a fost premiat la festivalul de la Cluj.",
    "Guvernul discută bugetul pentru anul viitor.",
    "Aplicația mobilă a fost lansată săptămâna trecută.",
    "Vremea va fi însorită în mare parte a țării.",
    "Profesorii cer salarii mai mari și condiții mai bune.",
    "Compania a raportat profituri record în ultimul trimestru.",
]


def main() -> int:
    print(f"Loading RomanianEmbedder (this may download ~500MB the first time)...")
    embedder = RomanianEmbedder()
    print(f"Device: {embedder.device}")

    vecs = embedder.encode(SENTENCES, batch_size=8, show_progress=True)

    print(f"\nOutput shape:        {vecs.shape}")
    print(f"Expected shape:      (10, 768)")
    print(f"Dtype:               {vecs.dtype}")
    print(f"L2 norms (first 3):  {[float(((v ** 2).sum()) ** 0.5) for v in vecs[:3]]}")

    ok = vecs.shape == (10, 768)
    print("\n" + ("PASS" if ok else "FAIL") + " smoke test")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
