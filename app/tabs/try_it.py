"""Try-it-live tab — user types Romanian text → both models predict.

The "live" piece of the application. Loads the saved BERTopic model
and (when available) the saved LDA model + dictionary, and runs them
on whatever the user pastes in.
"""

from __future__ import annotations

import numpy as np
import streamlit as st

from src import paths
from src.preprocessing.clean import clean_text

from .. import utils

EXAMPLES = {
    "Politics": (
        "Guvernul a aprobat astăzi un nou proiect de lege privind "
        "reforma sistemului public de pensii, care urmează să fie "
        "dezbătut în Parlament săptămâna viitoare."
    ),
    "Sports": (
        "Echipa națională a României a câștigat meciul de calificare "
        "la Campionatul European cu scorul de 3-1, după trei goluri "
        "marcate în repriza a doua."
    ),
    "Tech": (
        "O nouă aplicație mobilă dezvoltată în România folosește "
        "inteligența artificială pentru a traduce automat conversațiile "
        "în limba română în peste 30 de limbi."
    ),
    "Finance": (
        "Banca Națională a anunțat o nouă creștere a ratei dobânzii "
        "de referință, după ce inflația anuală a depășit nivelul prognozat "
        "pentru luna trecută."
    ),
    "Science": (
        "Cercetătorii de la o universitate din București au publicat un studiu "
        "despre efectele schimbărilor climatice asupra agriculturii din sudul "
        "României."
    ),
    "Culture": (
        "Festivalul de film de la Cluj a anunțat selecția oficială pentru "
        "ediția din acest an, cu premiere românești și proiecții în aer liber."
    ),
}


def _predict_bertopic(text: str) -> dict | None:
    """Return {topic_id, keywords, probability} or None if model missing."""
    model = utils.get_bertopic_model(paths.MODEL_DIR_MAIN)
    if model is None:
        return None
    embedder = utils.get_romanian_embedder()
    vec = embedder.encode([text], batch_size=1, show_progress=False)
    topics, probs = model.transform([text], vec)
    topic_id = int(topics[0])
    keywords = (
        [w for w, _ in (model.get_topic(topic_id) or [])][:10]
        if topic_id != -1
        else []
    )
    prob = float(np.max(probs[0])) if probs is not None and len(probs) else float("nan")
    return {"topic_id": topic_id, "keywords": keywords, "probability": prob}


def _predict_lda(text: str) -> dict | None:
    """Return {top_topics: [(topic_id, prob, keywords)]} or None if model missing.

    Loads the saved LDA model + dictionary and runs the same lemmatization +
    BoW pipeline used at training time, then returns the top-3 topics with
    keyword previews.
    """
    bundle = utils.get_lda_artifacts(paths.LDA_MODEL_MAIN, paths.LDA_DICTIONARY)
    if bundle is None:
        return None

    from src.lda.inference import predict_topic_distribution

    cleaned = clean_text(text)
    top_topics = predict_topic_distribution(
        bundle["model"], bundle["dictionary"], cleaned, top_n=3
    )
    return {"top_topics": top_topics}


def _render_card(title: str, body) -> None:
    with st.container(border=True):
        st.markdown(f"#### {title}")
        body()


def render() -> None:
    utils.section_header(
        "5. Live demo",
        "Paste Romanian news text and compare LDA vs. BERTopic predictions in real time.",
    )

    with st.expander("Presentation walkthrough (~4 min)", expanded=False):
        st.markdown(
            """
1. Click **Example: Politics**, then **Predict** — note human-readable topic names and keywords.
2. Repeat for **Sports**, **Tech**, and **Finance** (one sentence each).
3. Use **Science** and **Culture** to show coverage beyond the first four buttons.
4. Mention that LDA returns a **mixture** (top-3) while BERTopic returns one cluster (+ outliers).
5. Return to **4. Experiments** for metrics, confusion matrices, and stability if time allows.
            """
        )

    labels = list(EXAMPLES.items())
    row_a = st.columns(3)
    row_b = st.columns(3)
    if "try_it_text" not in st.session_state:
        st.session_state["try_it_text"] = labels[0][1]
    for col, (label, text) in zip(row_a + row_b, labels):
        if col.button(f"Example: {label}", use_container_width=True):
            st.session_state["try_it_text"] = text

    text = st.text_area(
        "Romanian text",
        value=st.session_state["try_it_text"],
        height=160,
        key="try_it_text_area",
    )

    if not st.button("Predict", type="primary"):
        st.caption("Click *Predict* to run both models on the text above.")
        return

    if not text.strip():
        st.warning("Please enter some text first.")
        return

    col_a, col_b = st.columns(2)

    with col_a:

        def _bert_body():
            with st.spinner("BERTopic encoding + transform..."):
                pred = _predict_bertopic(text)
            if pred is None:
                utils.missing_artifact(
                    paths.MODEL_DIR_MAIN,
                    "python scripts/fit_bertopic.py",
                )
                return
            tid = pred["topic_id"]
            name = utils.bertopic_topic_display_name(tid)
            st.markdown(f"**{name}** · `topic_id={tid}`")
            if pred["keywords"]:
                st.markdown("**Top keywords:** " + ", ".join(pred["keywords"]))
            if not np.isnan(pred["probability"]):
                st.caption(f"max probability over topics: {pred['probability']:.3f}")

        _render_card("BERTopic prediction", _bert_body)

    with col_b:

        def _lda_body():
            with st.spinner("LDA inference..."):
                pred = _predict_lda(text)
            if pred is None:
                st.info(
                    "LDA model not available yet. Run "
                    "`python scripts/run_lda_pipeline.py` to populate "
                    "`results/lda/model_main.gensim` + "
                    "`results/lda/dictionary.dict`."
                )
                return
            if not pred["top_topics"]:
                st.warning(
                    "No words matched the LDA dictionary "
                    "(text may be too short or all out-of-vocabulary)."
                )
                return
            labels = utils.lda_topic_label_map()
            for entry in pred["top_topics"]:
                tid = int(entry["topic_id"])
                name = labels.get(tid, f"Topic {tid}")
                st.markdown(
                    f"- **{name}** · `topic_id={tid}` · prob **{entry['prob']:.3f}**"
                    f"\n\n  Keywords: " + ", ".join(entry["keywords"])
                )

        _render_card("LDA prediction", _lda_body)
