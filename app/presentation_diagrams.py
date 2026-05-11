"""Mermaid sources for in-app presentation (§2.3)."""

PIPELINE_DIAGRAM = """
flowchart TD
    raw["MOROCO raw files"]:::data
    subgraph prep["Preprocessing"]
        load["load_moroco"]
        clean["clean_text"]
        split["stratified train test split"]
    end
    raw --> load --> clean --> split
    train["train.parquet"]:::data
    test["test.parquet"]:::data
    split --> train
    split --> test
    subgraph lda["LDA pipeline"]
        bow["BoW and Dictionary"]
        ldaTrain["LdaModel K sweep"]
    end
    subgraph bert["BERTopic pipeline"]
        embed["RoBERT embeddings"]
        umap["UMAP"]
        hdb["HDBSCAN"]
        ctfidf["c-TF-IDF"]
    end
    train --> bow --> ldaTrain
    train --> embed --> umap --> hdb --> ctfidf
    subgraph eval["Shared evaluation"]
        metrics["NMI and Purity and C_v"]
    end
    ldaTrain --> metrics
    ctfidf --> metrics
    test --> metrics
    artifacts["results artifacts"]:::artifacts
    metrics --> artifacts
    classDef data fill:#e8f4f8,stroke:#2c7da0,color:#000;
    classDef artifacts fill:#fff5e1,stroke:#e09f3e,color:#000;
"""

APP_DIAGRAM = """
flowchart LR
    user(["User browser"]):::user
    subgraph streamlit["Streamlit app"]
        direction TB
        problem["1 Problem"]
        solution["2 Solution"]
        impl["3 Implementation"]
        experiments["4 Experiments"]
        demo["5 Live demo"]
        utils["artifact loaders"]
    end
    subgraph cache["Cached models"]
        ldaModel["LDA and dictionary"]
        bertModel["BERTopic"]
        robert["RoBERT encoder"]
    end
    subgraph saved["Saved results"]
        csv["topic tables and metrics"]
        htmlviz["pyLDAvis and Plotly HTML"]
    end
    user --> problem
    problem --> utils
    solution --> utils
    experiments --> utils
    demo --> ldaModel
    demo --> bertModel
    demo --> robert
    utils --> csv
    utils --> htmlviz
    classDef user fill:#dde7c7,stroke:#6b8e23,color:#000;
"""

LIVE_DEMO_SEQUENCE = """
sequenceDiagram
    actor U as User
    participant A as Streamlit
    participant P as preprocessing
    participant LDA as LDA
    participant E as RoBERT
    participant B as BERTopic
    U->>A: paste Romanian news text
    A->>P: clean_text
    P-->>A: cleaned tokens
    A->>LDA: BoW inference
    LDA-->>A: top topic mixture
    A->>E: encode text
    E-->>A: embedding vector
    A->>B: transform
    B-->>A: topic id and keywords
    A-->>U: side by side cards
"""
