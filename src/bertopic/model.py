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
    # Misc
    language: str = "multilingual"
    calculate_probabilities: bool = True
    random_state: int = 42


def build_bertopic_model(cfg: BERTopicConfig | None = None):
    """Construct a BERTopic instance with UMAP + HDBSCAN as in the spec."""
    cfg = cfg or BERTopicConfig()

    from bertopic import BERTopic
    from hdbscan import HDBSCAN
    from umap import UMAP

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
    return BERTopic(
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        language=cfg.language,
        calculate_probabilities=cfg.calculate_probabilities,
        verbose=True,
    )
