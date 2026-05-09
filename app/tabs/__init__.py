"""Per-tab render functions for the Streamlit app.

Each module exposes a single `render()` function that the main entry point
in `app.streamlit_app` wires into a `st.tabs(...)` block.
"""

from . import bertopic_tab, comparison, home, lda, stability, try_it

__all__ = ["bertopic_tab", "comparison", "home", "lda", "stability", "try_it"]
