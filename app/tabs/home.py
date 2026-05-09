"""Home tab — landing page with project intro, status, and headline numbers."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src import paths

from .. import utils


def _headline_metrics() -> pd.DataFrame:
    """Pull the few numbers that summarize where we're at."""
    rows: list[dict] = []

    bert_eval = utils.load_json_or_none(paths.TEST_EVAL_JSON)
    if bert_eval is not None:
        rows.append(
            {
                "Method": "BERTopic",
                "NMI (test)": round(bert_eval.get("nmi_test", float("nan")), 4),
                "Purity (test)": round(bert_eval.get("purity_test", float("nan")), 4),
                "Outlier % (test)": round(
                    bert_eval.get("outlier_pct_test", float("nan")) * 100, 2
                ),
            }
        )
    else:
        rows.append(
            {
                "Method": "BERTopic",
                "NMI (test)": "—",
                "Purity (test)": "—",
                "Outlier % (test)": "—",
            }
        )

    rows.append(
        {
            "Method": "LDA",
            "NMI (test)": "—",
            "Purity (test)": "—",
            "Outlier % (test)": "n/a",
        }
    )

    return pd.DataFrame(rows)


def _artifact_status() -> pd.DataFrame:
    watch = [
        ("data/processed/train.parquet", paths.TRAIN_PARQUET),
        ("data/processed/test.parquet", paths.TEST_PARQUET),
        ("results/bertopic/embeddings_train.npy", paths.EMBEDDINGS_TRAIN),
        ("results/bertopic/embeddings_test.npy", paths.EMBEDDINGS_TEST),
        ("results/bertopic/model_main/", paths.MODEL_DIR_MAIN),
        ("results/bertopic/topic_keywords.csv", paths.TOPIC_TABLE_CSV),
        ("results/bertopic/test_evaluation.json", paths.TEST_EVAL_JSON),
        ("results/bertopic/stability_summary.csv", paths.STABILITY_SUMMARY_CSV),
        ("results/bertopic/bootstrap_nmi.json", paths.BOOTSTRAP_NMI_JSON),
    ]
    rows = [
        {
            "artifact": label,
            "status": "✅ ready" if path.exists() else "⏳ pending",
        }
        for label, path in watch
    ]
    return pd.DataFrame(rows)


def render() -> None:
    utils.section_header(
        "Topic Modeling on MOROCO — BERTopic vs. LDA",
        "Comparative study of two topic modeling approaches on the "
        "Moldavian and Romanian Dialectal Corpus (MOROCO).",
    )

    col_left, col_right = st.columns([0.55, 0.45])

    with col_left:
        st.markdown(
            "**The task.** Unsupervised topic discovery on Romanian news. "
            "Given articles labeled with one of six categories (culture, "
            "finance, politics, science, sports, tech), discover topics "
            "from the text alone and check how well they align with the "
            "gold labels.\n\n"
            "**Two methods compared:**\n"
            "- **LDA** — classical generative probabilistic model (`gensim`).\n"
            "- **BERTopic** — `RoBERT → UMAP → HDBSCAN → c-TF-IDF` pipeline.\n\n"
            "Use the tabs above to explore each method, compare them "
            "side-by-side, run live predictions on your own Romanian text, "
            "and inspect the stability / bootstrap-CI analysis."
        )

    with col_right:
        st.subheader("Headline metrics")
        st.dataframe(_headline_metrics(), hide_index=True, use_container_width=True)
        st.caption(
            "Numbers are filled in once `python scripts/run_bertopic_pipeline.py` "
            "(BERTopic) and the LDA pipeline have run."
        )

    st.divider()

    st.subheader("Pipeline status")
    st.caption(
        "Quick check of which pipeline outputs are already on disk. "
        "The other tabs use these files; missing ones show a friendly "
        "'run X first' card instead of crashing."
    )
    st.dataframe(_artifact_status(), hide_index=True, use_container_width=True)
