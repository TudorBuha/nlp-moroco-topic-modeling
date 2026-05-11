"""§4 Experiments and results — metrics, explorers, stability."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src import paths

from .. import utils
from . import bertopic_tab, comparison, lda, stability


def _headline_metrics() -> pd.DataFrame:
    bert_eval = utils.load_json_or_none(paths.TEST_EVAL_JSON)
    lda_eval = utils.load_json_or_none(paths.LDA_TEST_EVAL_JSON)

    def _fmt(value, suffix: str = "") -> str:
        if value is None:
            return "—"
        try:
            v = float(value)
        except (TypeError, ValueError):
            return "—"
        if pd.isna(v):
            return "—"
        if suffix == "%":
            return f"{v * 100:.2f}%"
        return f"{v:.4f}"

    def _row(method: str, eval_json: dict | None, outliers: bool) -> dict:
        if eval_json is None:
            return {
                "Method": method,
                "NMI (test)": "—",
                "Purity (test)": "—",
                "# topics": "—",
                "Outlier % (test)": "—" if outliers else "n/a",
            }
        return {
            "Method": method,
            "NMI (test)": _fmt(eval_json.get("nmi_test")),
            "Purity (test)": _fmt(eval_json.get("purity_test")),
            "# topics": str(eval_json.get("n_topics", "—")),
            "Outlier % (test)": (
                _fmt(eval_json.get("outlier_pct_test"), "%") if outliers else "n/a"
            ),
        }

    return pd.DataFrame(
        [
            _row("LDA", lda_eval, outliers=False),
            _row("BERTopic", bert_eval, outliers=True),
        ]
    )


def render() -> None:
    utils.section_header(
        "4. Experiments and results",
        "Quantitative comparison, visual exploration, and robustness checks.",
    )

    tab_summary, tab_lda, tab_bertopic, tab_compare, tab_stability = st.tabs(
        [
            "Headline metrics",
            "LDA results",
            "BERTopic results",
            "Side-by-side comparison",
            "Stability & CIs",
        ]
    )

    with tab_summary:
        st.markdown(
            """
**Setup.** Same MOROCO demo subset and test split for both methods.
LDA: sweep **K**, pick best **C_v** (K = 15). BERTopic: RoBERT embeddings, default UMAP/HDBSCAN,
Romanian stop-words in c-TF-IDF. Extrinsic metrics: **NMI** and **Purity** vs. MOROCO topic labels.
            """
        )
        st.dataframe(_headline_metrics(), hide_index=True, use_container_width=True)
        st.caption(
            "Open **Side-by-side comparison** for confusion matrices and embedding ablation; "
            "**Stability & CIs** for multi-seed variance and bootstrap intervals."
        )

    with tab_lda:
        lda.render()

    with tab_bertopic:
        bertopic_tab.render()

    with tab_compare:
        comparison.render()

    with tab_stability:
        stability.render()
