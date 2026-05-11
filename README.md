# Topic Modeling on MOROCO — BERTopic vs. LDA

Comparative study of two topic modeling approaches on the [MOROCO](https://github.com/butnaruandrei/MOROCO) corpus (Moldavian and Romanian Dialectal Corpus):

- **Method A — LDA** (classical probabilistic topic model)
- **Method B — BERTopic** (transformer-based clustering pipeline)

The goal is to train both methods on the same train/test split, evaluate them with comparable metrics (C_v coherence, NMI, Purity), and produce a fair side-by-side comparison.

The deliverables follow the assignment spec:

1. Problem statement
2. Proposed solution — 2.1 Theoretical aspects · 2.2 Dataset · 2.3 Application + diagram
3. Implementation — libraries / functions
4. Experiments and results

See [`docs/report.md`](docs/report.md) for the joint report and [`docs/architecture.md`](docs/architecture.md) for the pipeline + application diagrams.

---

## Team

| Person | GitHub | Method |
|---|---|---|
| Buha Tudor | [@TudorBuha](https://github.com/TudorBuha) | **BERTopic** |
| Ciorăscu Mihai | [@MihaiCiorascu](https://github.com/MihaiCiorascu) | **LDA** |

See [`docs/Team_Task_Breakdown.md`](docs/Team_Task_Breakdown.md) for the full step-by-step plan.

---

## Repository structure

```
.
├── data/
│   ├── raw/             # MOROCO source files (gitignored)
│   └── processed/       # train.parquet / test.parquet (gitignored)
├── notebooks/
│   ├── 01_data_inspection.ipynb     # MOROCO inspection
│   ├── 02_lda_full.ipynb            # LDA pipeline
│   └── 03_bertopic_full.ipynb       # BERTopic pipeline (T1–T7)
├── app/                 # Streamlit GUI (assignment §2.3)
│   ├── streamlit_app.py # entry point
│   ├── utils.py         # artifact loading helpers
│   └── tabs/            # one render() per tab
├── src/
│   ├── paths.py         # shared filesystem paths
│   ├── preprocessing/   # shared text loading, cleaning, tokenization
│   ├── lda/             # LDA training + evaluation
│   ├── bertopic/        # BERTopic pipeline + ablations
│   │   ├── embeddings.py    # RoBERT mean-pool encoder
│   │   ├── model.py         # BERTopicConfig + factory
│   │   ├── training.py      # fit / save / load
│   │   ├── inspection.py    # topic-keyword tables, outlier %
│   │   ├── visualizations.py# topics_2d / barchart / heatmap HTMLs
│   │   ├── coherence.py     # C_v via gensim
│   │   ├── hp_sweep.py      # T4 grid search
│   │   ├── ablation.py      # T5 RoBERT vs MPNet
│   │   ├── evaluate.py      # T6 NMI / Purity on test
│   │   └── stability.py     # T8 multi-seed stability runner
│   └── evaluation/      # shared metrics: NMI, Purity, confusion matrix
├── scripts/             # CLI wrappers around src/ modules (one per T-step)
├── tests/               # pytest unit tests (no data needed)
├── results/
│   ├── lda/             # trained models, dictionary, pyLDAvis HTML, plots
│   └── bertopic/        # embeddings, fitted model, visualizations, CSVs
├── docs/                # final report, planning, Method B draft
├── requirements.txt     # runtime deps
├── requirements-dev.txt # + pytest, ruff, nbformat
├── pyproject.toml       # pytest config
└── README.md
```

## End-to-end CLI workflow

```bash
# 0. Get data + spaCy Romanian model (one-time, ~50 MB + ~30 MB)
python scripts/download_moroco.py
python -m spacy download ro_core_news_sm

# 1. Preprocess. Use --max-per-class for a fast demo subset (~30 min total CPU run);
#    omit it (or pass -1) to run on the full ~33k corpus.
python scripts/preprocess.py --max-per-class 1000   # → data/processed/{train,test}.parquet

# 2a. Method A — LDA pipeline (gensim)
python scripts/run_lda_pipeline.py
#    or step by step:
#    python scripts/train_lda.py        # build dict + corpus, sweep K, pyLDAvis
#    python scripts/inspect_lda.py      # topic-keyword table
#    python scripts/evaluate_lda.py     # NMI / Purity / confusion on test

# 2b. Method B — BERTopic pipeline
python scripts/run_bertopic_pipeline.py
#    or step by step:
#    python scripts/encode_docs.py            # T1 — encode train + test with RoBERT
#    python scripts/fit_bertopic.py           # T2 — fit BERTopic and save model
#    python scripts/inspect_topics.py         # T3 — topic table + 3 HTML viz
#    python scripts/hp_sweep.py               # T4 — hyperparameter sweep
#    python scripts/embedding_ablation.py     # T5 — RoBERT vs multilingual MPNet
#    python scripts/evaluate_on_test.py       # T6 — NMI / Purity / confusion matrix
#    python scripts/stability_analysis.py     # T8 — multi-seed runs + bootstrap 95% CIs
```

The notebooks `notebooks/03_bertopic_full.ipynb` and `notebooks/02_lda_full.ipynb` are equivalent to the above scripts and run cell-by-cell for interactive exploration.

## Application (GUI — assignment §2.3)

The interactive Streamlit app wraps both trained models:

```bash
pip install -r requirements.txt   # streamlit is already in there
streamlit run app/streamlit_app.py
```

Then open the URL Streamlit prints (default `http://localhost:8501`). The app has ten tabs:

- **Home** — project intro, dataset summary, headline metrics, pipeline status.
- **Problem** — task definition and method ownership.
- **Solution** — theory, MOROCO overview, and architecture diagrams.
- **Implementation** — libraries, modules, and how to reproduce the pipelines.
- **Experiments** — headline metrics, plots, and stability summaries.
- **Try it live** — paste any Romanian news text and see the topic each model assigns.
- **LDA explorer** — coherence curve, topic-keyword table, embedded `pyLDAvis`.
- **BERTopic explorer** — topic-keyword table with search + the three interactive HTML viz.
- **Comparison** — side-by-side metrics, confusion matrices, embedding ablation.
- **Stability** — multi-seed runs, bootstrap 95 % CIs, hyperparameter sweep.

The app is a **read-only frontend** over the saved artifacts in `results/`. If a step hasn't been run yet, the corresponding tab shows a friendly *"run X first"* card instead of crashing.

See [`docs/architecture.md`](docs/architecture.md) for the full architecture diagram (Mermaid).

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

Unit tests cover preprocessing helpers, evaluation metrics, BERTopic inspection logic, and import smoke checks. They don't need MOROCO data, so they run in seconds.

---

## Setup

### 1. Clone

```bash
git clone https://github.com/TudorBuha/nlp-moroco-topic-modeling.git
cd nlp-moroco-topic-modeling
```

### 2. Create a virtual environment (Python 3.10+)

```bash
# Windows PowerShell
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Per-method extras

**For LDA work:**
```bash
python -m spacy download ro_core_news_sm
```

**For BERTopic work:**
The first time `src/bertopic/` is run, the `readerbench/robert-base` model (~500MB) is downloaded from HuggingFace into the local cache. Pre-fetch with:
```bash
python -c "from transformers import AutoModel, AutoTokenizer; AutoTokenizer.from_pretrained('readerbench/robert-base'); AutoModel.from_pretrained('readerbench/robert-base')"
```

---

## Project phases

| Phase | Description | Status |
|---|---|---|
| 0 | Joint setup (repo, env, deps)         | done |
| 1 | Data preparation (download + cleaning + split) | done |
| 2 | Independent modeling (LDA + BERTopic) | done — both pipelines run end-to-end |
| 3 | Joint comparison                      | done — see `docs/report.md` §4.4 |
| 4 | Final report                          | done — `docs/report.md` |
| 5 | Application (Streamlit)               | done — `streamlit run app/streamlit_app.py` |

### Headline numbers (1,000 docs / class subset, 80 / 20 split, seed 42)

| Method   | NMI   | Purity | # topics | Outlier % |
|---|---:|---:|---:|---:|
| LDA      | 0.346 | 0.591  | 15       | n/a       |
| BERTopic | 0.334 | 0.598  | 19       | 3.7 %     |

See `docs/report.md` §4 for the full results, including the K-sweep, hyperparameter grid, embedding ablation, and bootstrap / multi-seed stability analysis.

---

## Dataset

**MOROCO** — Moldavian and Romanian Dialectal Corpus. ~33k news samples labeled with one of 6 topic categories (culture, finance, politics, science, sports, tech) and 2 dialects (Romanian / Moldavian). Source: [butnaruandrei/MOROCO](https://github.com/butnaruandrei/MOROCO).

> ⚠️ Raw data files are **not** committed. Download them into `data/raw/` after cloning. See [`data/raw/README.md`](data/raw/README.md) for instructions.

---

## License

MIT — see [`LICENSE`](LICENSE).
