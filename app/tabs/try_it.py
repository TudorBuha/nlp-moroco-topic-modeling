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

    Reads the LDA model + dictionary from `results/lda/`. The exact filenames
    depend on Mihai's M2.3 (`model_K{N}.gensim`); we try a couple of common ones.
    """
    model_path = paths.RESULTS_LDA / "model_main.gensim"
    if not model_path.exists():
        candidates = sorted(paths.RESULTS_LDA.glob("model_K*.gensim"))
        if not candidates:
            return None
        model_path = candidates[0]
    dictionary_path = paths.RESULTS_LDA / "dictionary.dict"

    bundle = utils.get_lda_artifacts(model_path, dictionary_path)
    if bundle is None:
        return None
    nlp = utils.get_spacy_ro()
    if nlp is None:
        return None

    cleaned = clean_text(text)
    doc = nlp(cleaned)
    tokens = [t.lemma_ for t in doc if not t.is_space and t.lemma_]

    bow = bundle["dictionary"].doc2bow(tokens)
    if not bow:
        return {"top_topics": []}

    distribution = sorted(
        bundle["model"].get_document_topics(bow), key=lambda x: -x[1]
    )[:3]
    top_topics = []
    for tid, p in distribution:
        keywords = [w for w, _ in bundle["model"].show_topic(tid, topn=10)]
        top_topics.append({"topic_id": int(tid), "prob": float(p), "keywords": keywords})
    return {"top_topics": top_topics}


def _render_card(title: str, body) -> None:
    with st.container(border=True):
        st.markdown(f"#### {title}")
        body()


def render() -> None:
    utils.section_header(
        "Try it live",
        "Paste any Romanian news article and see what topic each model assigns.",
    )

    col_examples = st.columns(len(EXAMPLES))
    if "try_it_text" not in st.session_state:
        st.session_state["try_it_text"] = list(EXAMPLES.values())[0]
    for col, (label, text) in zip(col_examples, EXAMPLES.items()):
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
            label = "🚫 outlier topic" if tid == -1 else f"Topic **#{tid}**"
            st.markdown(label)
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
                    "LDA model not available yet.\n\n"
                    "Once the LDA pipeline (Steps M1–M3) saves "
                    "`results/lda/model_K{N}.gensim` + "
                    "`results/lda/dictionary.dict`, this card lights up."
                )
                return
            if not pred["top_topics"]:
                st.warning(
                    "No words matched the LDA dictionary "
                    "(text may be too short or all out-of-vocabulary)."
                )
                return
            for entry in pred["top_topics"]:
                st.markdown(
                    f"- Topic **#{entry['topic_id']}** "
                    f"(prob {entry['prob']:.3f}): "
                    + ", ".join(entry["keywords"])
                )

        _render_card("LDA prediction", _lda_body)
