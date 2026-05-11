"""§1 Problem statement — presentation tab."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from .. import utils


def render() -> None:
    utils.section_header(
        "1. Problem statement",
        "Describe the NLP task solved in this project.",
    )

    st.markdown(
        """
**Task.** Unsupervised **topic discovery** on Romanian news from the MOROCO corpus.

Given a collection of documents, the goal is to:

1. Partition (or softly assign) documents into coherent topics **without using gold labels at training time**.
2. For each topic, produce a ranked list of characteristic words.
3. Evaluate topics intrinsically (do the words cohere?) and extrinsically (do topics align with MOROCO categories?).

**Why it matters.** News archives are large and heterogeneous. Topic models help summarize what a corpus is about, compare editorial focus, and support downstream search or monitoring — without hand-labeling every article.

**What we compare.** Two standard but very different families:
        """
    )

    st.dataframe(
        pd.DataFrame(
            [
                (
                    "Method A — LDA — classical generative probabilistic topic model "
                    "on bag-of-words inputs.",
                    "Buha Tudor",
                ),
                (
                    "Method B — BERTopic — transformer embeddings, density clustering, "
                    "and class-based TF-IDF for topic words.",
                    "Ciorascu Mihai",
                ),
            ],
            columns=["Method", "Author"],
        ),
        hide_index=True,
        use_container_width=True,
    )
