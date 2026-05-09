"""BERTopic pipeline (Tudor)."""

from .ablation import run_ablation, save_ablation
from .coherence import compute_cv_coherence, topic_words_from_bertopic
from .embeddings import RomanianEmbedder, encode_documents
from .evaluate import evaluate_on_test, save_evaluation
from .hp_sweep import save_sweep, sweep
from .inspection import (
    attach_manual_labels,
    n_topics,
    outlier_proportion,
    save_topic_table,
    topic_keyword_table,
)
from .model import BERTopicConfig, build_bertopic_model
from .training import fit_bertopic, load_model, save_model
from .visualizations import save_default_visualizations

__all__ = [
    # config / model factory
    "BERTopicConfig",
    "build_bertopic_model",
    # embeddings
    "RomanianEmbedder",
    "encode_documents",
    # training
    "fit_bertopic",
    "save_model",
    "load_model",
    # inspection
    "topic_keyword_table",
    "save_topic_table",
    "attach_manual_labels",
    "outlier_proportion",
    "n_topics",
    # visualizations
    "save_default_visualizations",
    # coherence
    "compute_cv_coherence",
    "topic_words_from_bertopic",
    # hp sweep
    "sweep",
    "save_sweep",
    # ablation
    "run_ablation",
    "save_ablation",
    # evaluation
    "evaluate_on_test",
    "save_evaluation",
]
