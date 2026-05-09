"""Comparison tab — side-by-side metrics + confusion matrices."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src import paths

from .. import utils

LDA_EVAL_JSON = paths.RESULTS_LDA / "test_evaluation.json"
LDA_CONFUSION_CSV = paths.RESULTS_LDA / "test_confusion_matrix.csv"


def _side_by_side_metrics() -> pd.DataFrame:
    bert_eval = utils.load_json_or_none(paths.TEST_EVAL_JSON) or {}
    lda_eval = utils.load_json_or_none(LDA_EVAL_JSON) or {}

    def _fmt(d: dict, key: str, suffix: str = "") -> str:
        if key not in d or d[key] is None:
            return "—"
        v = d[key]
        if suffix == "%":
            return f"{v * 100:.1f}%"
        return f"{v:.4f}"

    rows = [
        {
            "Metric": "NMI vs MOROCO (test)",
            "LDA": _fmt(lda_eval, "nmi_test"),
            "BERTopic": _fmt(bert_eval, "nmi_test"),
        },
        {
            "Metric": "Purity (test)",
            "LDA": _fmt(lda_eval, "purity_test"),
            "BERTopic": _fmt(bert_eval, "purity_test"),
        },
        {
            "Metric": "# topics found",
            "LDA": str(lda_eval.get("n_topics", "—")),
            "BERTopic": str(bert_eval.get("n_topics", "—")),
        },
        {
            "Metric": "Outlier proportion (test)",
            "LDA": "n/a",
            "BERTopic": _fmt(bert_eval, "outlier_pct_test", suffix="%"),
        },
    ]
    return pd.DataFrame(rows)


def _render_confusion(label: str, csv_path) -> None:
    cm = utils.load_csv_or_none(csv_path)
    st.markdown(f"**{label}**")
    if cm is None:
        utils.missing_artifact(
            csv_path,
            "python scripts/evaluate_on_test.py" if "bertopic" in str(csv_path)
            else "the LDA evaluation step (M3)",
        )
    else:
        st.dataframe(cm, use_container_width=True)


def render() -> None:
    utils.section_header(
        "Comparison",
        "How LDA and BERTopic compare on the same MOROCO test split.",
    )

    st.subheader("Metric table")
    st.dataframe(
        _side_by_side_metrics(), hide_index=True, use_container_width=True
    )

    st.divider()
    st.subheader("Confusion matrices")
    st.caption(
        "Rows = predicted topic id from each model. "
        "Columns = MOROCO ground-truth label."
    )
    col_a, col_b = st.columns(2)
    with col_a:
        _render_confusion("LDA", LDA_CONFUSION_CSV)
    with col_b:
        _render_confusion("BERTopic", paths.TEST_CONFUSION_CSV)

    st.divider()
    st.subheader("Embedding ablation (BERTopic only)")
    ablation = utils.load_csv_or_none(paths.ABLATION_CSV)
    if ablation is None:
        utils.missing_artifact(
            paths.ABLATION_CSV, "python scripts/embedding_ablation.py"
        )
    else:
        st.dataframe(ablation, hide_index=True, use_container_width=True)
        st.caption(
            "Same BERTopic config, two different sentence encoders. "
            "RoBERT is Romanian-specific; MPNet is multilingual."
        )
