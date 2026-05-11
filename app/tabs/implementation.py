"""§3 Implementation — libraries and project modules."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from .. import utils


def render() -> None:
    utils.section_header(
        "3. Implementation",
        "Libraries and main Python modules used in the project.",
    )

    st.markdown(
        """
Code lives under `src/` (reusable logic) and `scripts/` (CLI entry points).
The Streamlit app in `app/` is a read-only frontend over artifacts in `results/`.
        """
    )

    st.subheader("Libraries")
    st.dataframe(
        pd.DataFrame(
            [
                ("pandas / numpy", "data frames, arrays, metrics tables"),
                ("scikit-learn", "CountVectorizer, train/test split"),
                ("gensim", "LDA, Dictionary, coherence C_v"),
                ("spaCy + ro_core_news_sm", "Romanian lemmatization for LDA"),
                ("transformers / torch", "RoBERT embeddings for BERTopic"),
                ("bertopic", "UMAP + HDBSCAN + c-TF-IDF pipeline"),
                ("umap-learn / hdbscan", "dimensionality reduction and clustering"),
                ("pyLDAvis / plotly", "LDA and BERTopic HTML visualizations"),
                ("streamlit", "interactive presentation and live demo"),
                ("pytest", "unit tests without MOROCO data"),
            ],
            columns=["Library", "Role"],
        ),
        hide_index=True,
        use_container_width=True,
    )

    st.subheader("Core modules")
    st.dataframe(
        pd.DataFrame(
            [
                ("src/preprocessing/", "load_moroco, clean_text, stratified split"),
                ("src/lda/", "corpus, training, inspection, evaluate, inference"),
                ("src/bertopic/", "embeddings, model, training, inspection, evaluate"),
                ("src/bertopic/stability.py", "multi-seed runs, bootstrap CIs"),
                ("src/evaluation/", "shared NMI, Purity, confusion matrix"),
                ("app/utils.py", "artifact loaders + human-readable topic labels"),
            ],
            columns=["Module", "Responsibility"],
        ),
        hide_index=True,
        use_container_width=True,
    )

    st.subheader("CLI scripts (reproducible runs)")
    st.markdown(
        """
- `scripts/preprocess.py` — shared train/test parquet
- `scripts/run_lda_pipeline.py` — LDA train, inspect, evaluate
- `scripts/run_bertopic_pipeline.py` — encode, fit, inspect, sweep, ablation, evaluate, stability
- `streamlit run app/streamlit_app.py` — this presentation + demo
        """
    )
