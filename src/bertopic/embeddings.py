"""Sentence-level embeddings for Romanian text (Tudor, Phase 1 + T1).

Uses `readerbench/robert-base` by default, with mean-pooling over the last
hidden state (masking out PAD tokens).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

DEFAULT_MODEL = "readerbench/robert-base"
DEFAULT_MAX_LENGTH = 128


class RomanianEmbedder:
    """Wraps a HuggingFace encoder with mean-pooling.

    Example:
        >>> emb = RomanianEmbedder()
        >>> vecs = emb.encode(["Bună ziua!", "Cum vă merge?"])
        >>> vecs.shape
        (2, 768)
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str | None = None,
        max_length: int = DEFAULT_MAX_LENGTH,
    ) -> None:
        self.model_name = model_name
        self.max_length = max_length
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def encode(
        self,
        texts: Iterable[str],
        batch_size: int = 32,
        show_progress: bool = True,
    ) -> np.ndarray:
        texts = list(texts)
        loader = DataLoader(texts, batch_size=batch_size, shuffle=False)
        out: list[np.ndarray] = []
        iterator = tqdm(loader, desc="encoding", disable=not show_progress)
        for batch in iterator:
            enc = self.tokenizer(
                list(batch),
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)
            hidden = self.model(**enc).last_hidden_state  # [B, T, H]
            mask = enc["attention_mask"].unsqueeze(-1).float()  # [B, T, 1]
            summed = (hidden * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1e-9)
            mean = (summed / counts).cpu().numpy()
            out.append(mean)
        return np.vstack(out).astype(np.float32)


def encode_documents(
    docs: Iterable[str],
    out_path: Path | str,
    model_name: str = DEFAULT_MODEL,
    batch_size: int = 32,
    max_length: int = DEFAULT_MAX_LENGTH,
) -> np.ndarray:
    """Encode and save to disk so it isn't recomputed on every notebook run."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    embedder = RomanianEmbedder(
        model_name=model_name, max_length=max_length
    )
    vecs = embedder.encode(docs, batch_size=batch_size)
    np.save(out_path, vecs)
    print(f"Saved {vecs.shape} -> {out_path}")
    return vecs
