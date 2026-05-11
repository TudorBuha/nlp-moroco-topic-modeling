"""Streamlit entry point.

Run with:

    streamlit run app/streamlit_app.py

The app is a read-only frontend over the artifacts produced by the CLI
pipeline in `scripts/`. Each tab gracefully handles missing artifacts.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make `src` importable when running via `streamlit run`.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st

from app.tabs import bertopic_tab, comparison, home, lda, stability, try_it


def main() -> None:
    st.set_page_config(
        page_title="MOROCO topic modeling — LDA vs BERTopic",
        page_icon="📚",
        layout="wide",
    )

    with st.sidebar:
        st.markdown("### MOROCO topic modeling")
        st.caption("BERTopic vs. LDA — interactive explorer.")
        st.markdown(
            "- [Repository](https://github.com/TudorBuha/nlp-moroco-topic-modeling)\n"
            "- [Architecture](https://github.com/TudorBuha/nlp-moroco-topic-modeling/blob/main/docs/architecture.md)\n"
            "- [Report](https://github.com/TudorBuha/nlp-moroco-topic-modeling/blob/main/docs/report.md)\n"
        )
        st.divider()
        st.caption(
            "If a tab shows a 'run X first' card, run the matching script under "
            "`scripts/` to generate the missing artifact."
        )

    tab_home, tab_try, tab_lda, tab_bertopic, tab_compare, tab_stability = st.tabs(
        ["Home", "Try it live", "LDA explorer", "BERTopic explorer", "Comparison", "Stability"]
    )

    with tab_home:
        home.render()
    with tab_try:
        try_it.render()
    with tab_lda:
        lda.render()
    with tab_bertopic:
        bertopic_tab.render()
    with tab_compare:
        comparison.render()
    with tab_stability:
        stability.render()


if __name__ == "__main__":
    main()
