"""LDA training: fit, save/load, and sweep over K with C_v coherence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel, LdaModel

DEFAULT_K_VALUES: tuple[int, ...] = (5, 6, 8, 10, 12, 15, 20)


def fit_lda(
    corpus: Sequence,
    dictionary: Dictionary,
    *,
    num_topics: int,
    passes: int = 10,
    iterations: int = 200,
    random_state: int = 42,
    chunksize: int = 2000,
    alpha: str = "auto",
    eta: str = "auto",
) -> LdaModel:
    """Fit a Gensim LdaModel with sensible defaults for medium-sized corpora."""
    return LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        passes=passes,
        iterations=iterations,
        random_state=random_state,
        chunksize=chunksize,
        alpha=alpha,
        eta=eta,
        eval_every=None,
    )


def save_lda(model: LdaModel, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(path))


def load_lda(path: str | Path) -> LdaModel:
    return LdaModel.load(str(path))


def _coherence_cv(
    model: LdaModel,
    tokens: Iterable[Iterable[str]],
    dictionary: Dictionary,
) -> float:
    cm = CoherenceModel(
        model=model, texts=list(tokens), dictionary=dictionary, coherence="c_v"
    )
    return float(cm.get_coherence())


def sweep_k(
    corpus: Sequence,
    tokens: Iterable[Iterable[str]],
    dictionary: Dictionary,
    *,
    k_values: Iterable[int] = DEFAULT_K_VALUES,
    passes: int = 10,
    iterations: int = 200,
    random_state: int = 42,
) -> tuple[pd.DataFrame, dict[int, LdaModel]]:
    """Train one LDA per K, score with C_v, return (scores_df, models_by_k)."""
    rows: list[dict] = []
    models: dict[int, LdaModel] = {}
    tokens_list = list(tokens)

    for k in k_values:
        model = fit_lda(
            corpus,
            dictionary,
            num_topics=k,
            passes=passes,
            iterations=iterations,
            random_state=random_state,
        )
        cv = _coherence_cv(model, tokens_list, dictionary)
        rows.append({"k": k, "c_v": cv})
        models[k] = model
        print(f"    K={k:>3d} | C_v = {cv:.4f}")

    df = pd.DataFrame(rows).sort_values("c_v", ascending=False).reset_index(drop=True)
    return df, models


def write_sweep_csv(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.sort_values("k").to_csv(path, index=False)


def write_sweep_json(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(df.to_dict(orient="records"), f, ensure_ascii=False, indent=2)
