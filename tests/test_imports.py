"""Smoke test: import everything to catch syntax / import errors early."""

from __future__ import annotations


def test_top_level_imports():
    from src import paths  # noqa: F401
    from src.preprocessing import (  # noqa: F401
        clean_text,
        load_moroco,
        train_test_split,
    )
    from src.evaluation.metrics import (  # noqa: F401
        build_confusion_matrix,
        nmi_score,
        purity_score,
    )


def test_bertopic_subpackage_imports():
    # This pulls in: ablation, coherence, embeddings, evaluate, hp_sweep,
    # inspection, model, training, visualizations.
    import src.bertopic as bt

    expected = {
        "BERTopicConfig",
        "RomanianEmbedder",
        "encode_documents",
        "fit_bertopic",
        "save_model",
        "load_model",
        "topic_keyword_table",
        "save_topic_table",
        "attach_manual_labels",
        "outlier_proportion",
        "n_topics",
        "save_default_visualizations",
        "compute_cv_coherence",
        "topic_words_from_bertopic",
        "sweep",
        "save_sweep",
        "run_ablation",
        "save_ablation",
        "evaluate_on_test",
        "save_evaluation",
    }
    missing = expected - set(dir(bt))
    assert not missing, f"missing exports from src.bertopic: {missing}"


def test_bertopic_config_defaults():
    from src.bertopic import BERTopicConfig

    cfg = BERTopicConfig()
    assert cfg.min_cluster_size == 30
    assert cfg.n_neighbors == 15
    assert cfg.n_components == 5
    assert cfg.random_state == 42
