"""BERTopic model factory (Phase 2 — Step T2)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BERTopicConfig:
    """Hyperparameters for the BERTopic pipeline (sweep these in T4)."""
    # UMAP
    n_neighbors: int = 15
    n_components: int = 5
    min_dist: float = 0.0
    # HDBSCAN
    min_cluster_size: int = 30
    # c-TF-IDF / CountVectorizer
    min_df: int = 5
    ngram_range: tuple[int, int] = (1, 2)
    # Misc
    language: str = "multilingual"
    calculate_probabilities: bool = True
    random_state: int = 42


def build_bertopic_model(cfg: BERTopicConfig | None = None):
    """Construct a BERTopic instance with UMAP + HDBSCAN as in the spec.

    A Romanian stop-word list is passed to the c-TF-IDF vectorizer so the
    per-topic keywords are not dominated by function words ("de", "în", "și",
    "la", …). The stop-word list is the same one used by the LDA pipeline so
    the two methods' topic-keyword tables are visually comparable.
    """
    cfg = cfg or BERTopicConfig()

    from bertopic import BERTopic
    from hdbscan import HDBSCAN
    from sklearn.feature_extraction.text import CountVectorizer
    from umap import UMAP

    from src.preprocessing.tokenize import _RO_STOPWORDS

    umap_model = UMAP(
        n_neighbors=cfg.n_neighbors,
        n_components=cfg.n_components,
        min_dist=cfg.min_dist,
        random_state=cfg.random_state,
    )
    hdbscan_model = HDBSCAN(
        min_cluster_size=cfg.min_cluster_size,
        metric="euclidean",
        prediction_data=True,
    )
    vectorizer_model = CountVectorizer(
        stop_words=sorted(_RO_STOPWORDS),
        min_df=cfg.min_df,
        ngram_range=cfg.ngram_range,
    )
    return BERTopic(
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        language=cfg.language,
        calculate_probabilities=cfg.calculate_probabilities,
        verbose=True,
    )
