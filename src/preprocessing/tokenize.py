"""Tokenization + lemmatization (Phase 1.5).

Lemmatization is needed for **LDA only**; BERTopic consumes raw cleaned text.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Iterable

# A minimal Romanian stop-word list. TODO: extend / replace with a fuller
# list (e.g. from the `stop-words` package or NLTK) once one is chosen.
_RO_STOPWORDS: set[str] = {
    "a", "ai", "al", "ale", "am", "ar", "are", "as", "așa", "așadar", "atât",
    "au", "aveau", "avem", "aveți", "ca", "cam", "care", "ce", "cea", "cel",
    "cele", "cu", "da", "dar", "de", "din", "dintr", "doar", "după", "el",
    "ea", "ei", "ele", "este", "eu", "fi", "fie", "fost", "i", "iar", "în",
    "îl", "îi", "într", "într-o", "într-un", "la", "le", "lor", "lui", "mai",
    "mea", "meu", "mi", "mine", "ne", "nu", "o", "ori", "pe", "pentru",
    "peste", "prin", "sa", "să", "sau", "se", "sub", "sunt", "te", "ți",
    "tot", "tu", "un", "una", "unde", "unei", "unele", "uneori", "unii",
    "unor", "voi", "voastră", "voastre", "vor", "vouă", "și",
}


def tokenize(text: str) -> list[str]:
    """Whitespace tokenizer with Romanian stop-word removal."""
    return [tok for tok in text.split() if tok and tok not in _RO_STOPWORDS]


@lru_cache(maxsize=1)
def _get_spacy():
    """Load spaCy ro_core_news_sm once. Tokenization-only pipeline (faster)."""
    import spacy  # local import so non-LDA users don't pay for it
    try:
        return spacy.load("ro_core_news_sm", disable=["parser", "ner"])
    except OSError as exc:  # pragma: no cover
        raise RuntimeError(
            "spaCy Romanian model not installed. Run:\n"
            "    python -m spacy download ro_core_news_sm"
        ) from exc


def lemmatize_tokens(tokens: Iterable[str]) -> list[str]:
    """Lemmatize a token list using spaCy ro_core_news_sm. LDA only."""
    nlp = _get_spacy()
    doc = nlp(" ".join(tokens))
    return [t.lemma_ for t in doc if t.lemma_ and not t.is_space]
