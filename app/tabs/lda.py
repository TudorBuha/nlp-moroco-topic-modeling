"""LDA explorer tab — pyLDAvis embed + topic-keyword table + coherence curve."""

from __future__ import annotations

import streamlit as st
import streamlit.components.v1 as components

from src import paths

from .. import utils


def render() -> None:
    utils.section_header(
        "LDA explorer (Method A)",
        "Discovered topics from the Gensim LDA model.",
    )

    table = utils.load_csv_or_none(paths.LDA_TOPIC_TABLE_CSV)
    if table is None:
        utils.missing_artifact(
            paths.LDA_TOPIC_TABLE_CSV, "python scripts/inspect_lda.py"
        )
    else:
        st.subheader("Topic-keyword table")
        labels = utils.lda_topic_label_map()
        table = table.copy()
        table.insert(
            0,
            "topic_name",
            table["topic_id"].map(lambda tid: labels.get(int(tid), f"Topic {int(tid)}")),
        )
        st.caption(
            "Each row is one LDA topic: **topic_name** is a manual GUI label; "
            "**topic_id** is the gensim index (edit labels in "
            "`results/lda/topic_keywords_labeled.csv` or `src/lda/gui_labels.py`)."
        )
        st.dataframe(table, hide_index=True, use_container_width=True)

    st.divider()
    st.subheader("Coherence sweep over K")
    sweep = utils.load_csv_or_none(paths.LDA_SWEEP_CSV)
    if sweep is not None:
        sweep = sweep.sort_values("k")
        st.line_chart(sweep.set_index("k")["c_v"], use_container_width=True)
        best_idx = sweep["c_v"].idxmax()
        st.caption(
            f"Best K = **{int(sweep.loc[best_idx, 'k'])}** "
            f"with C_v = {sweep.loc[best_idx, 'c_v']:.4f}"
        )
    if paths.LDA_COHERENCE_PNG.exists():
        st.image(str(paths.LDA_COHERENCE_PNG))
    elif sweep is None:
        utils.missing_artifact(
            paths.LDA_SWEEP_CSV, "python scripts/train_lda.py"
        )

    st.divider()
    st.subheader("pyLDAvis interactive visualization")
    html = utils.load_html_or_none(paths.LDA_LDAVIS_HTML)
    if html is None:
        utils.missing_artifact(
            paths.LDA_LDAVIS_HTML, "python scripts/train_lda.py"
        )
    else:
        components.html(html, height=900, scrolling=True)
