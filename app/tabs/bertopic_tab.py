"""BERTopic explorer tab — topic table + 3 interactive HTML visualizations
+ keyword search."""

from __future__ import annotations

import streamlit as st
import streamlit.components.v1 as components

from src import paths

from .. import utils

VIZ_FILES = {
    "topics_2d": paths.RESULTS_BERTOPIC / "topics_2d.html",
    "barchart": paths.RESULTS_BERTOPIC / "barchart.html",
    "heatmap": paths.RESULTS_BERTOPIC / "heatmap.html",
}


def _render_table_with_search() -> None:
    table = utils.load_csv_or_none(
        paths.RESULTS_BERTOPIC / "topic_keywords_labeled.csv"
    )
    if table is None:
        table = utils.load_csv_or_none(paths.TOPIC_TABLE_CSV)

    if table is None:
        utils.missing_artifact(
            paths.TOPIC_TABLE_CSV, "python scripts/inspect_topics.py"
        )
        return

    query = st.text_input(
        "Filter by keyword (Romanian)",
        placeholder="e.g. politic, fotbal, banca, calculator…",
    ).strip().lower()
    filtered = table
    if query:
        filtered = table[
            table["top_keywords"].fillna("").str.lower().str.contains(query, na=False)
        ]
        if filtered.empty:
            st.warning(f"No topic contains the keyword `{query}`.")
            return

    st.dataframe(filtered, hide_index=True, use_container_width=True)
    st.caption(f"Showing {len(filtered)} of {len(table)} topics.")


def _render_viz(name: str, height: int = 600) -> None:
    path = VIZ_FILES[name]
    html = utils.load_html_or_none(path)
    if html is None:
        utils.missing_artifact(path, "python scripts/inspect_topics.py")
        return
    components.html(html, height=height, scrolling=True)


def render() -> None:
    utils.section_header(
        "BERTopic explorer (Method B)",
        "Topics discovered by the `RoBERT → UMAP → HDBSCAN → c-TF-IDF` pipeline.",
    )

    st.subheader("Topic-keyword table")
    _render_table_with_search()

    st.divider()
    viz_tabs = st.tabs(["Topics 2-D map", "Per-topic barchart", "Topic similarity heatmap"])
    with viz_tabs[0]:
        _render_viz("topics_2d", height=650)
    with viz_tabs[1]:
        _render_viz("barchart", height=650)
    with viz_tabs[2]:
        _render_viz("heatmap", height=650)
