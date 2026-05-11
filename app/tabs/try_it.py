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
        "Președintele a convocat o ședință de urgență a Guvernului pentru a "
        "discuta măsurile de securitate energetică propuse de ministerele de resort."
    ),
    "Sports": (
        "FCSB a învins cu 2-0 echipa din deplasare, după goluri marcate de "
        "atacantul titular în minutele 55 și 78."
    ),
    "Tech": (
        "Compania de tehnologie a lansat un smartphone cu procesor nou, ecran "
        "OLED și cameră de 200 de megapixeli, disponibil din luna viitoare."
    ),
    "Finance": (
        "Banca română anunță faptul că euro crește la 5,22 lei, iar analiștii "
        "estimează presiuni suplimentare pe piața valutară."
    ),
    "Science": (
        "Astronomii români au observat o cometă rară cu telescopul de la "
        "observatorul din Cluj, publicând date spectrale în revista științifică."
    ),
    "Culture": (
        "Muzeul Național de Artă din București inaugurează o expoziție dedicată "
        "pictorilor români din secolul al XX-lea, cu lucrări restaurate recent."
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
        "Try it live",
        "Paste Romanian news text and compare LDA vs. BERTopic predictions in real time.",
    )

    labels = list(EXAMPLES.items())
    row_a = st.columns(3)
    row_b = st.columns(3)
    if "try_it_text" not in st.session_state:
        st.session_state["try_it_text"] = labels[0][1]
    for col, (label, text) in zip(row_a + row_b, labels):
        if col.button(
            f"Example: {label}",
            use_container_width=True,
            key=f"try_it_example_{label.lower()}",
        ):
            st.session_state["try_it_text"] = text

    text = st.text_area(
        "Romanian text",
        height=160,
        key="try_it_text",
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
