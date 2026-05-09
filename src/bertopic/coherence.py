"""C_v topic coherence for BERTopic (Tudor) — also reusable by LDA via the
shared evaluation module.

Approach: extract the top-N keywords per BERTopic topic, then feed them
together with a tokenized version of the original documents into Gensim's
`CoherenceModel(coherence='c_v')`. Outlier topic (-1) is excluded.
"""

from __future__ import annotations

from collections.abc import Sequence


def _tokenize_for_coherence(docs: Sequence[str]) -> list[list[str]]:
    """Cheap whitespace tokenizer; matches what BERTopic's c-TF-IDF sees."""
    return [d.lower().split() for d in docs]


def topic_words_from_bertopic(topic_model, top_n: int = 10) -> list[list[str]]:
    """Return list of top-N word lists, one per non-outlier topic."""
    out: list[list[str]] = []
    for tid in topic_model.get_topic_info()["Topic"].tolist():
        if tid == -1:
            continue
        words = [w for w, _ in topic_model.get_topic(tid) or []][:top_n]
        if words:
            out.append(words)
    return out


def compute_cv_coherence(
    topic_model,
    docs: Sequence[str],
    top_n: int = 10,
) -> float:
    """Return mean C_v coherence over all non-outlier BERTopic topics.

    Returns NaN (as a float) if there aren't enough topics to score.
    """
    from gensim.corpora import Dictionary
    from gensim.models import CoherenceModel

    topic_words = topic_words_from_bertopic(topic_model, top_n=top_n)
    if len(topic_words) < 2:
        return float("nan")

    texts = _tokenize_for_coherence(docs)
    dictionary = Dictionary(texts)

    cm = CoherenceModel(
        topics=topic_words,
        texts=texts,
        dictionary=dictionary,
        coherence="c_v",
    )
    return float(cm.get_coherence())
