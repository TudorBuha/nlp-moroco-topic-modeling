"""Build a Gensim dictionary + bag-of-words corpus from cleaned MOROCO docs.

Tokenization for LDA is more aggressive than for BERTopic:
    1. apply `clean_text` (already done in preprocessing)
    2. drop stop words
    3. lemmatize with spaCy ro_core_news_sm
    4. drop tokens shorter than 3 chars
    5. filter the dictionary: `no_below=10, no_above=0.5, keep_n=20000`

The cached lemmatized tokens are saved alongside the corpus so we don't have
to re-run spaCy if we re-train.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Iterable

import pandas as pd
from gensim.corpora import Dictionary, MmCorpus

from src.preprocessing.tokenize import _RO_STOPWORDS, _get_spacy

MIN_TOKEN_LEN = 3


def tokenize_for_lda(
    cleaned_texts: Iterable[str],
    *,
    batch_size: int = 64,
    n_process: int = 1,
) -> list[list[str]]:
    """Tokenize + stop-word remove + lemmatize a sequence of cleaned texts.

    Uses `nlp.pipe` for batching; on CPU `n_process=1` is usually fastest
    because the per-process spaCy startup tax dominates for ~6k docs.
    """
    nlp = _get_spacy()
    out: list[list[str]] = []
    for doc in nlp.pipe(cleaned_texts, batch_size=batch_size, n_process=n_process):
        toks = [
            t.lemma_.lower()
            for t in doc
            if (
                t.lemma_
                and not t.is_space
                and not t.is_punct
                and len(t.lemma_) >= MIN_TOKEN_LEN
                and t.lemma_.lower() not in _RO_STOPWORDS
            )
        ]
        out.append(toks)
    return out


def build_corpus(
    train_texts: pd.Series | list[str],
    test_texts: pd.Series | list[str] | None = None,
    *,
    no_below: int = 10,
    no_above: float = 0.5,
    keep_n: int = 20_000,
) -> dict:
    """Tokenize, build a `Dictionary`, filter it, and produce BoW corpora.

    Returns a dict with:
        dictionary, train_tokens, train_corpus,
        test_tokens (or None), test_corpus (or None)
    """
    train_tokens = tokenize_for_lda(list(train_texts))
    dictionary = Dictionary(train_tokens)
    dictionary.filter_extremes(no_below=no_below, no_above=no_above, keep_n=keep_n)
    dictionary.compactify()

    train_corpus = [dictionary.doc2bow(t) for t in train_tokens]

    test_tokens: list[list[str]] | None = None
    test_corpus: list[list[tuple[int, int]]] | None = None
    if test_texts is not None:
        test_tokens = tokenize_for_lda(list(test_texts))
        test_corpus = [dictionary.doc2bow(t) for t in test_tokens]

    return {
        "dictionary": dictionary,
        "train_tokens": train_tokens,
        "train_corpus": train_corpus,
        "test_tokens": test_tokens,
        "test_corpus": test_corpus,
    }


def save_corpus(bundle: dict, out_dir: Path) -> dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "dictionary": out_dir / "dictionary.dict",
        "train_corpus": out_dir / "train_corpus.mm",
        "train_tokens": out_dir / "train_tokens.pkl",
    }
    bundle["dictionary"].save(str(paths["dictionary"]))
    MmCorpus.serialize(str(paths["train_corpus"]), bundle["train_corpus"])
    with paths["train_tokens"].open("wb") as f:
        pickle.dump(bundle["train_tokens"], f)

    if bundle.get("test_corpus") is not None:
        paths["test_corpus"] = out_dir / "test_corpus.mm"
        paths["test_tokens"] = out_dir / "test_tokens.pkl"
        MmCorpus.serialize(str(paths["test_corpus"]), bundle["test_corpus"])
        with paths["test_tokens"].open("wb") as f:
            pickle.dump(bundle["test_tokens"], f)
    return paths


def load_corpus(corpus_dir: Path) -> dict:
    corpus_dir = Path(corpus_dir)
    dictionary = Dictionary.load(str(corpus_dir / "dictionary.dict"))
    train_corpus = list(MmCorpus(str(corpus_dir / "train_corpus.mm")))
    with (corpus_dir / "train_tokens.pkl").open("rb") as f:
        train_tokens = pickle.load(f)

    test_corpus = None
    test_tokens = None
    if (corpus_dir / "test_corpus.mm").exists():
        test_corpus = list(MmCorpus(str(corpus_dir / "test_corpus.mm")))
        with (corpus_dir / "test_tokens.pkl").open("rb") as f:
            test_tokens = pickle.load(f)

    return {
        "dictionary": dictionary,
        "train_tokens": train_tokens,
        "train_corpus": train_corpus,
        "test_tokens": test_tokens,
        "test_corpus": test_corpus,
    }
