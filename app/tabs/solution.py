"""§2 Proposed solution — theory, dataset, application."""

from __future__ import annotations

import streamlit as st

from src import paths

from .. import utils
from ..mermaid import render_mermaid


def _render_theory() -> None:
    st.subheader("2.1 Theoretical aspects")
    st.markdown(
        """
**LDA (Method A).** Each document is a mixture of topics; each topic is a distribution over words.
Training estimates topic–word and document–topic distributions with approximate inference (Gensim).
The number of topics **K** is fixed up front; we sweep **K** and pick the value with best **C_v** coherence.

**BERTopic (Method B).** Documents are embedded with a Romanian RoBERT encoder, reduced with **UMAP**,
clustered with **HDBSCAN** (including an explicit outlier topic **-1**), then described with **c-TF-IDF**
keywords per cluster. The topic count is **discovered** from the data rather than chosen manually.

| Aspect | LDA | BERTopic |
| --- | --- | --- |
| Input | lemmatized bag-of-words | raw / lightly cleaned text |
| Topic count | user-chosen K | discovered by clustering |
| Outliers | every doc has a mixture | explicit outlier cluster |
| Strength | fast CPU baseline | semantic, finer-grained themes |
        """
    )


def _render_dataset() -> None:
    st.subheader("2.2 Dataset — MOROCO")
    st.markdown(
        """
**MOROCO** (Moldavian and Romanian Dialectal Corpus): ~33k news samples, six topical categories
(culture, finance, politics, science, sports, tech), two dialects (Romanian and Moldavian).

**Demo configuration.** Stratified subset with up to 1,000 documents per category, 80/20 train/test split,
`random_state=42`. Both methods consume the **same** parquet splits so metrics are comparable.
        """
    )

    stats = utils.load_csv_or_none(paths.DATA_PROCESSED / "dataset_stats.csv")
    if stats is not None:
        st.dataframe(stats, hide_index=True, use_container_width=True)
    else:
        st.info(
            "Run `python scripts/preprocess.py --max-per-class 1000` to populate "
            "`data/processed/dataset_stats.csv`."
        )

    st.markdown("**End-to-end pipeline (shared preprocessing, parallel model paths):**")
    render_mermaid("pipeline", height=520)


def _render_application() -> None:
    st.subheader("2.3 Application")
    st.markdown(
        """
The presentation is delivered **inside this Streamlit app**: read-only over saved artifacts in
`results/`, plus live inference on user text in the **Live demo** tab (no retraining in the browser).

**Application diagram** — how tabs load artifacts and how live prediction reuses cached models:
        """
    )
    render_mermaid("application", height=380)

    st.markdown("**Live demo sequence** — what happens when you click *Predict*:")
    render_mermaid("live_demo", height=680)


def render() -> None:
    utils.section_header(
        "2. Proposed solution",
        "Theory, dataset, and application architecture.",
    )

    tab_theory, tab_data, tab_app = st.tabs(
        ["2.1 Theory", "2.2 Dataset", "2.3 Application"]
    )
    with tab_theory:
        _render_theory()
    with tab_data:
        _render_dataset()
    with tab_app:
        _render_application()
