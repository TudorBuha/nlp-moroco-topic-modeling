"""Unit tests for the LDA package.

Most of these stay below the spaCy boundary by feeding pre-tokenized lists
directly into Gensim, so the suite remains fast and doesn't depend on the
Romanian spaCy model.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from gensim.corpora import Dictionary

from src.lda.evaluate import predict_dominant_topics
from src.lda.inspection import (
    attach_manual_labels,
    save_topic_table,
    topic_keyword_table,
)
from src.lda.training import fit_lda, load_lda, save_lda, sweep_k


@pytest.fixture
def synthetic_corpus():
    """Three-topic toy corpus that LDA can pull apart trivially."""
    topic_a = ["football", "match", "goal", "stadium", "score", "team"]
    topic_b = ["election", "vote", "president", "parliament", "government", "law"]
    topic_c = ["computer", "software", "code", "algorithm", "data", "machine"]

    docs = []
    truth = []
    rng = np.random.default_rng(0)
    for label, words in [("sports", topic_a), ("politics", topic_b), ("tech", topic_c)]:
        for _ in range(30):
            n = int(rng.integers(8, 16))
            doc = list(rng.choice(words, size=n, replace=True))
            docs.append(doc)
            truth.append(label)

    dictionary = Dictionary(docs)
    corpus = [dictionary.doc2bow(d) for d in docs]
    return {"docs": docs, "corpus": corpus, "dictionary": dictionary, "truth": truth}


def test_fit_lda_produces_correct_num_topics(synthetic_corpus):
    model = fit_lda(
        synthetic_corpus["corpus"],
        synthetic_corpus["dictionary"],
        num_topics=3,
        passes=3,
        iterations=50,
        random_state=0,
    )
    assert model.num_topics == 3


def test_save_and_load_roundtrip(tmp_path: Path, synthetic_corpus):
    model = fit_lda(
        synthetic_corpus["corpus"],
        synthetic_corpus["dictionary"],
        num_topics=3,
        passes=3,
        iterations=50,
        random_state=0,
    )
    path = tmp_path / "lda.gensim"
    save_lda(model, path)
    assert path.exists()

    reloaded = load_lda(path)
    assert reloaded.num_topics == 3


def test_sweep_k_returns_one_row_per_k(synthetic_corpus):
    sweep_df, models = sweep_k(
        synthetic_corpus["corpus"],
        synthetic_corpus["docs"],
        synthetic_corpus["dictionary"],
        k_values=[2, 3, 4],
        passes=2,
        iterations=30,
        random_state=0,
    )
    assert set(sweep_df["k"]) == {2, 3, 4}
    assert set(models) == {2, 3, 4}
    assert all(isinstance(c, float) for c in sweep_df["c_v"])


def test_topic_keyword_table_shape_and_keys(synthetic_corpus):
    model = fit_lda(
        synthetic_corpus["corpus"],
        synthetic_corpus["dictionary"],
        num_topics=3,
        passes=3,
        iterations=50,
        random_state=0,
    )
    table = topic_keyword_table(model, top_n=5)
    assert list(table.columns) == ["topic_id", "top_keywords", "top_weights"]
    assert len(table) == 3
    assert table["top_keywords"].iloc[0].count(",") == 4  # 5 keywords -> 4 commas


def test_attach_manual_labels_fills_missing_with_empty_string(synthetic_corpus):
    model = fit_lda(
        synthetic_corpus["corpus"],
        synthetic_corpus["dictionary"],
        num_topics=3,
        passes=3,
        iterations=50,
        random_state=0,
    )
    table = topic_keyword_table(model, top_n=5)
    labeled = attach_manual_labels(table, labels={0: "alpha"})
    assert "label" in labeled.columns
    assert labeled.loc[labeled["topic_id"] == 0, "label"].iloc[0] == "alpha"
    other_labels = labeled.loc[labeled["topic_id"] != 0, "label"].tolist()
    assert all(lbl == "" for lbl in other_labels)


def test_save_topic_table_writes_csv(tmp_path: Path, synthetic_corpus):
    model = fit_lda(
        synthetic_corpus["corpus"],
        synthetic_corpus["dictionary"],
        num_topics=3,
        passes=3,
        iterations=30,
        random_state=0,
    )
    table = topic_keyword_table(model, top_n=3)
    out = tmp_path / "out" / "topics.csv"
    save_topic_table(table, out)
    assert out.exists()
    reloaded = pd.read_csv(out)
    assert len(reloaded) == 3


def test_predict_dominant_topics_aligns_with_truth(synthetic_corpus):
    """With this trivially-separable corpus LDA's dominant topic should give
    a near-perfect majority alignment with the gold labels."""
    model = fit_lda(
        synthetic_corpus["corpus"],
        synthetic_corpus["dictionary"],
        num_topics=3,
        passes=10,
        iterations=200,
        random_state=0,
    )
    preds = predict_dominant_topics(model, synthetic_corpus["corpus"])
    assert preds.shape == (90,)
    assert set(preds.tolist()) <= {0, 1, 2}
    # Each gold class should map to a single dominant cluster (purity=1 ideally,
    # but allow some looseness in the tiny synthetic case).
    df = pd.DataFrame({"true": synthetic_corpus["truth"], "pred": preds})
    purity = (
        df.groupby("pred")["true"].apply(lambda s: s.value_counts().iloc[0]).sum()
        / len(df)
    )
    assert purity >= 0.9


def test_predict_dominant_topics_handles_empty_bow(synthetic_corpus):
    model = fit_lda(
        synthetic_corpus["corpus"],
        synthetic_corpus["dictionary"],
        num_topics=3,
        passes=3,
        iterations=30,
        random_state=0,
    )
    preds = predict_dominant_topics(model, synthetic_corpus["corpus"] + [[]])
    assert preds[-1] == -1
