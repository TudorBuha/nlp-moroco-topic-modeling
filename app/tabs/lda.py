"""LDA explorer tab — pyLDAvis embed + topic-keyword table.

Reads only LDA artifacts that Mihai's pipeline (Steps M1–M4) is expected
to write into `results/lda/`. Renders graceful fallbacks when files
aren't there yet.
"""

from __future__ import annotations

import streamlit as st
import streamlit.components.v1 as components

from src import paths

from .. import utils

LDA_TOPIC_TABLE_CSV = paths.RESULTS_LDA / "topic_keywords.csv"
LDA_LDAVIS_HTML = paths.RESULTS_LDA / "ldavis.html"
LDA_COHERENCE_PNG = paths.RESULTS_LDA / "coherence_curve.png"


def render() -> None:
    utils.section_header(
        "LDA explorer (Method A)",
        "Discovered topics from the Gensim LDA model.",
    )

    table = utils.load_csv_or_none(LDA_TOPIC_TABLE_CSV)
    if table is None:
        utils.missing_artifact(
            LDA_TOPIC_TABLE_CSV,
            "the LDA pipeline (Steps M1–M3) — Mihai's side",
        )
    else:
        st.subheader("Topic-keyword table")
        st.dataframe(table, hide_index=True, use_container_width=True)

    st.divider()
    st.subheader("Coherence sweep over K")
    if LDA_COHERENCE_PNG.exists():
        st.image(str(LDA_COHERENCE_PNG))
    else:
        utils.missing_artifact(LDA_COHERENCE_PNG, "Step M2.5 (coherence curve)")

    st.divider()
    st.subheader("pyLDAvis interactive visualization")
    html = utils.load_html_or_none(LDA_LDAVIS_HTML)
    if html is None:
        utils.missing_artifact(LDA_LDAVIS_HTML, "Step M3.3 (`pyLDAvis.save_html`)")
    else:
        components.html(html, height=900, scrolling=True)
