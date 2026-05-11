"""Human-readable labels for BERTopic topics (GUI + CSV).

These defaults match the model trained on the demo subset (1000 / class,
seed 42) — currently 19 topics + the outlier (`-1`).

If you re-run BERTopic and topic ids shift, edit either:
- `DEFAULT_BERTOPIC_GUI_LABELS` here, or
- the `gui_label` column of `results/bertopic/topic_keywords_labeled.csv`
  (overrides take precedence inside the Streamlit app).
"""

from __future__ import annotations

# Keys are BERTopic topic ids. `-1` is the catch-all outlier topic.
DEFAULT_BERTOPIC_GUI_LABELS: dict[int, str] = {
    -1: "Outlier (no clear topic)",
    0: "General news prose",
    1: "Sports — matches & scores",
    2: "General prose (ASCII spelling)",
    3: "Science / opinion",
    4: "Generic / filler",
    5: "Science — research & studies",
    6: "Politics — government statements",
    7: "Science (Moldavian spelling)",
    8: "Politics — official quotes",
    9: "Tech — smartphones",
    10: "Tech — data & privacy",
    11: "Science (Moldavian spelling, variant)",
    12: "Finance — currency amounts",
    13: "Accidents & police incidents",
    14: "Crime / hospital news",
    15: "Economy — wages & income",
    16: "Automotive — cars & production",
    17: "Weather — temperatures & forecast",
    18: "Currency & exchange rate",
}


def default_label(topic_id: int) -> str | None:
    """Return the built-in GUI label for `topic_id`, or None if unknown."""
    return DEFAULT_BERTOPIC_GUI_LABELS.get(int(topic_id))
