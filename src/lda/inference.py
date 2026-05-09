"""Single-doc inference helpers for the Streamlit Try-it-live tab."""

from __future__ import annotations

from typing import Sequence

from gensim.corpora import Dictionary
from gensim.models import LdaModel

from .corpus import tokenize_for_lda


def predict_topic_distribution(
    model: LdaModel,
    dictionary: Dictionary,
    cleaned_text: str,
    *,
    top_n: int = 3,
) -> list[dict]:
    """Return the top-N (topic_id, prob, top-10 keywords) for a cleaned text.

    The text is expected to have already passed through `clean_text`. We
    re-run the LDA tokenizer (lemmatization + stop-word removal) and BoW
    encode against the saved `dictionary` so the resulting BoW lives in the
    same vocabulary the model was trained on.
    """
    tokens = tokenize_for_lda([cleaned_text])[0]
    bow = dictionary.doc2bow(tokens)
    if not bow:
        return []

    distribution: Sequence[tuple[int, float]] = sorted(
        model.get_document_topics(bow, minimum_probability=0.0),
        key=lambda x: -x[1],
    )[:top_n]

    out: list[dict] = []
    for tid, prob in distribution:
        keywords = [w for w, _ in model.show_topic(int(tid), topn=10)]
        out.append(
            {"topic_id": int(tid), "prob": float(prob), "keywords": keywords}
        )
    return out
