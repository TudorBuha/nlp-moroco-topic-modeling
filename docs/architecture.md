# Application architecture

This document holds the **two diagrams** referenced from the joint report:

- **Pipeline diagram** — how MOROCO becomes saved artifacts (referenced from §2.2 / §3).
- **Application diagram** — how the Streamlit GUI consumes those artifacts (referenced from §2.3).

Both are written in [Mermaid](https://mermaid.js.org/), which renders natively on GitHub and in most LaTeX/Pandoc tooling.

---

## 1. Pipeline diagram

End-to-end flow from raw MOROCO files to the saved artifacts both methods produce. The two method pipelines run independently on the **same** train/test split; they only meet again at the evaluation layer.

```mermaid
flowchart TD
    raw["MOROCO raw files<br/>(samples + labels)"]:::data

    subgraph prep["src/preprocessing/  &nbsp;(Phase 1)"]
        load["load_moroco()"]
        clean["clean_text()"]
        tok["tokenize() / lemmatize_tokens()"]
        split["train_test_split() · stratified 80/20"]
    end

    raw --> load --> clean --> tok --> split

    train["train.parquet"]:::data
    test["test.parquet"]:::data
    split --> train
    split --> test

    subgraph lda["src/lda/  &nbsp;(Method A)"]
        bow["BoW corpus + Gensim Dictionary"]
        ldaTrain["LdaModel · K ∈ {5,6,8,10,12,15,20}"]
        ldaPick["pick best K by C_v coherence"]
    end

    subgraph bert["src/bertopic/  &nbsp;(Method B)"]
        embed["RomanianEmbedder<br/>readerbench/robert-base"]
        umap["UMAP n_neighbors=15, n_components=5"]
        hdb["HDBSCAN min_cluster_size=30"]
        ctfidf["c-TF-IDF topic representation"]
    end

    train --> bow --> ldaTrain --> ldaPick
    train --> embed --> umap --> hdb --> ctfidf

    subgraph eval["src/evaluation/  &nbsp;(shared)"]
        nmi["nmi_score · purity_score"]
        cv["compute_cv_coherence"]
        boot["bootstrap_metric (95% CI)"]
        stab["stability.run_multi_seed"]
    end

    ldaPick --> eval
    ctfidf --> eval
    test --> eval

    artifacts["results/<br/>"]:::artifacts
    eval --> artifacts
    ldaPick --> artifacts
    ctfidf --> artifacts

    classDef data fill:#e8f4f8,stroke:#2c7da0,color:#000;
    classDef artifacts fill:#fff5e1,stroke:#e09f3e,color:#000;
```

### Saved artifacts

| Path | Produced by | Consumed by |
|---|---|---|
| `data/processed/{train,test}.parquet` | preprocessing | both pipelines + the app |
| `results/lda/model_K{N}.gensim` | LDA training (M2.3) | LDA tab in app |
| `results/lda/dictionary.dict`, `corpus.mm` | LDA training (M1.6) | LDA tab in app |
| `results/lda/ldavis.html` | LDA inspection (M3.3) | LDA tab in app |
| `results/lda/topic_keywords.csv` | LDA inspection (M3.1) | LDA tab + Comparison tab |
| `results/bertopic/embeddings_{train,test}.npy` | T1 | T2, T6, app |
| `results/bertopic/model_main/` | T2 | T3, T6, T8, **app live inference** |
| `results/bertopic/topic_keywords.csv`, `topic_keywords_labeled.csv` | T3 | BERTopic tab |
| `results/bertopic/{topics_2d,barchart,heatmap}.html` | T3 | BERTopic tab (embedded) |
| `results/bertopic/hp_sweep.csv` | T4 | Stability tab |
| `results/bertopic/ablation.csv` | T5 | Comparison tab |
| `results/bertopic/test_evaluation.json`, `test_confusion_matrix.csv` | T6 | Comparison tab |
| `results/bertopic/stability_runs.csv`, `stability_summary.csv` | T8 | Stability tab |
| `results/bertopic/bootstrap_{nmi,purity}.json` | T8 | Stability tab |

---

## 2. Application diagram (§2.3)

The Streamlit app is a **read-only frontend**: it never trains a model, it only loads what the pipeline already saved. The single live operation is **encoding a user-supplied text** in the *Try it live* tab, which reuses the trained models.

```mermaid
flowchart LR
    user(["User<br/>(browser)"]):::user

    subgraph app["app/streamlit_app.py"]
        direction TB
        home["Home tab"]
        try["Try it live tab"]
        ldaTab["LDA explorer tab"]
        bertTab["BERTopic explorer tab"]
        cmp["Comparison tab"]
        stab["Stability tab"]
        utils["app/utils.py<br/>load_*() / artifact checks"]
    end

    subgraph cache["@st.cache_resource"]
        ldaModel["LDA model<br/>+ dictionary"]
        bertModel["BERTopic model"]
        spacyNlp["spaCy ro_core_news_sm"]
        robert["RoBERT encoder"]
    end

    subgraph results["results/  (saved artifacts)"]
        ldaCsv["lda/topic_keywords.csv"]
        ldaHtml["lda/ldavis.html"]
        bertCsv["bertopic/topic_keywords*.csv"]
        bertHtml["bertopic/{topics_2d, barchart, heatmap}.html"]
        evalJson["bertopic/test_evaluation.json"]
        confCsv["bertopic/test_confusion_matrix.csv"]
        stabCsv["bertopic/stability_summary.csv"]
        bootJson["bertopic/bootstrap_*.json"]
    end

    user -->|HTTP| app

    home    --> utils
    ldaTab  --> utils
    bertTab --> utils
    cmp     --> utils
    stab    --> utils

    try --> ldaModel
    try --> bertModel
    try --> spacyNlp
    try --> robert

    utils --> ldaCsv
    utils --> ldaHtml
    utils --> bertCsv
    utils --> bertHtml
    utils --> evalJson
    utils --> confCsv
    utils --> stabCsv
    utils --> bootJson

    classDef user fill:#dde7c7,stroke:#6b8e23,color:#000;
```

### How a "Try it live" prediction flows

```mermaid
sequenceDiagram
    actor U as User
    participant A as Streamlit app
    participant P as preprocessing
    participant LDA as LDA (gensim)
    participant E as RoBERT encoder
    participant B as BERTopic

    U->>A: pastes a Romanian news text
    A->>P: clean_text + tokenize + lemmatize
    P-->>A: cleaned tokens
    A->>LDA: dictionary.doc2bow(tokens)
    LDA-->>A: topic distribution (top-3)
    A->>E: encode(raw_text)
    E-->>A: 768-d embedding
    A->>B: transform([text], embedding)
    B-->>A: topic id + top-10 c-TF-IDF keywords
    A-->>U: side-by-side prediction card
```

---

## 3. Graceful degradation

Each tab follows this pattern:

```python
def render() -> None:
    artifact = load_or_none(paths.SOMETHING)
    if artifact is None:
        st.info(
            "This tab needs `results/.../something.csv`. "
            "Run `python scripts/<step>.py` first."
        )
        return
    render_real_content(artifact)
```

Concretely: if Mihai hasn't pushed `data/processed/*` yet, the LDA + Comparison + Try-it-live (LDA half) tabs show a friendly *"data hand-off pending"* card instead of crashing — but the Stability and BERTopic tabs already work end-to-end as soon as the BERTopic pipeline runs.

---

## 4. Re-rendering the diagrams

GitHub renders Mermaid blocks automatically — no action needed in the browser. For the PDF report, the easiest options are:

- **Pandoc + mermaid-filter** (`pandoc report.md --filter mermaid-filter -o report.pdf`).
- **Online renderer**: paste the Mermaid block into [mermaid.live](https://mermaid.live), export PNG/SVG, embed in your final PDF.
- **VS Code Markdown Preview Mermaid Support** extension to confirm the diagrams look right while editing.
