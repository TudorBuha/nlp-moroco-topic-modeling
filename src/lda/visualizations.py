"""LDA visualization helpers — coherence curve PNG + pyLDAvis HTML."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pandas as pd
from gensim.corpora import Dictionary
from gensim.models import LdaModel


def save_coherence_curve(sweep_df: pd.DataFrame, png_path: str | Path) -> Path:
    """Plot C_v vs K and persist as PNG. Headless-safe (Agg backend)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    png_path = Path(png_path)
    png_path.parent.mkdir(parents=True, exist_ok=True)

    df = sweep_df.sort_values("k")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(df["k"], df["c_v"], marker="o", linewidth=2)
    best_k = int(df.loc[df["c_v"].idxmax(), "k"])
    ax.axvline(best_k, linestyle="--", alpha=0.5, color="tab:orange",
               label=f"best K = {best_k}")
    ax.set_xlabel("number of topics (K)")
    ax.set_ylabel(r"C$_v$ topic coherence")
    ax.set_title("LDA coherence sweep")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    return png_path


def save_pyldavis_html(
    model: LdaModel,
    corpus: Sequence,
    dictionary: Dictionary,
    html_path: str | Path,
) -> Path:
    """Save the pyLDAvis interactive visualization as a standalone HTML."""
    import pyLDAvis
    import pyLDAvis.gensim_models as gensim_vis

    html_path = Path(html_path)
    html_path.parent.mkdir(parents=True, exist_ok=True)

    prepared = gensim_vis.prepare(model, list(corpus), dictionary)
    pyLDAvis.save_html(prepared, str(html_path))
    return html_path
