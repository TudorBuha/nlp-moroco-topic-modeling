"""Streamlit entry point.

Run with:

    streamlit run app/streamlit_app.py

The app doubles as the live presentation: assignment sections 1–4 plus a live demo tab.
Each tab gracefully handles missing artifacts so the demo never crashes mid-presentation.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make `src` importable when running via `streamlit run`.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st

from app.tabs import experiments, implementation, problem, solution, try_it


def main() -> None:
    st.set_page_config(
        page_title="MOROCO topic modeling — LDA vs BERTopic",
        page_icon="📚",
        layout="wide",
    )

    with st.sidebar:
        st.markdown("### Presentation mode")
        st.caption("Walk the numbered tabs top-to-bottom (~12–15 min).")
        st.markdown(
            "1. **Problem** — task definition\n"
            "2. **Solution** — theory, MOROCO, app diagram\n"
            "3. **Implementation** — libraries and modules\n"
            "4. **Experiments** — metrics, plots, stability\n"
            "5. **Live demo** — Romanian examples + Predict"
        )
        st.divider()
        st.markdown("### Links")
        st.markdown(
            "- [Repository](https://github.com/TudorBuha/nlp-moroco-topic-modeling)\n"
            "- [Architecture notes](https://github.com/TudorBuha/nlp-moroco-topic-modeling/blob/main/docs/architecture.md)\n"
            "- [Written report](https://github.com/TudorBuha/nlp-moroco-topic-modeling/blob/main/docs/report.md)\n"
        )
        st.divider()
        st.caption(
            "Missing artifacts show a friendly *run X first* card instead of crashing. "
            "The GUI does not retrain models."
        )
        st.caption(
            "Diagrams in **2. Solution** are pre-rendered SVG files under `app/static/` "
            "(no live Mermaid CDN). There is no long-running training job in the background."
        )

    (
        tab_problem,
        tab_solution,
        tab_impl,
        tab_experiments,
        tab_demo,
    ) = st.tabs(
        [
            "1. Problem",
            "2. Solution",
            "3. Implementation",
            "4. Experiments",
            "5. Live demo",
        ]
    )

    with tab_problem:
        problem.render()
    with tab_solution:
        solution.render()
    with tab_impl:
        implementation.render()
    with tab_experiments:
        experiments.render()
    with tab_demo:
        try_it.render()


if __name__ == "__main__":
    main()
