"""Human-readable labels for LDA topics (GUI + CSV).

These defaults match the K = 15 sweep trained with seed 42 on the demo subset.
If you re-run LDA with a different K, update `DEFAULT_LDA_GUI_LABELS` or fill the
`label` column in `results/lda/topic_keywords_labeled.csv` — non-empty CSV labels
override these defaults in the Streamlit app.
"""

from __future__ import annotations

# Keys are gensim topic ids (0 … num_topics − 1).
DEFAULT_LDA_GUI_LABELS: dict[int, str] = {
    0: "Weather & environment",
    1: "Finance & currency",
    2: "Politics & government",
    3: "Elections & voting",
    4: "Science & research",
    5: "General discourse",
    6: "Football & matches",
    7: "Astronomy & discovery",
    8: "General / vague wording",
    9: "Automotive & tech products",
    10: "General discourse (variant)",
    11: "Culture & entertainment mix",
    12: "Culture, film & events",
    13: "Sports & tournaments",
    14: "Economy & business",
}


def default_label(topic_id: int) -> str | None:
    """Return the built-in GUI label for `topic_id`, or None if unknown."""
    return DEFAULT_LDA_GUI_LABELS.get(int(topic_id))
