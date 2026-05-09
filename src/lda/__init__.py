"""LDA pipeline (Method A) — gensim-based topic modeling for MOROCO.

Steps mirror BERTopic's so the comparison is symmetric:
    M1  build_corpus      -> tokenize + lemmatize + Dictionary + BoW
    M2  fit_lda / sweep_K -> train LdaModel, sweep over K, pick best by C_v
    M3  inspection        -> topic-keyword tables, manual labels
    M3.5 visualizations   -> coherence curve PNG, pyLDAvis HTML
    M4  evaluate_on_test  -> NMI, Purity, confusion matrix on held-out test

The same `compute_cv_coherence` helper from `src.bertopic.coherence` is used
by both methods to keep the two C_v numbers directly comparable.
"""

from .corpus import (
    build_corpus,
    load_corpus,
    save_corpus,
    tokenize_for_lda,
)
from .evaluate import evaluate_lda_on_test, save_evaluation
from .inference import predict_topic_distribution
from .inspection import (
    attach_manual_labels,
    save_topic_table,
    topic_keyword_table,
)
from .training import (
    fit_lda,
    load_lda,
    save_lda,
    sweep_k,
)
from .visualizations import save_coherence_curve, save_pyldavis_html

__all__ = [
    "attach_manual_labels",
    "build_corpus",
    "evaluate_lda_on_test",
    "fit_lda",
    "load_corpus",
    "load_lda",
    "predict_topic_distribution",
    "save_coherence_curve",
    "save_corpus",
    "save_evaluation",
    "save_lda",
    "save_pyldavis_html",
    "save_topic_table",
    "sweep_k",
    "tokenize_for_lda",
    "topic_keyword_table",
]
