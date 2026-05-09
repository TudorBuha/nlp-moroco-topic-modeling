# Topic Modeling on MOROCO — BERTopic vs. LDA

Comparative study of two topic modeling approaches on the [MOROCO](https://github.com/butnaruandrei/MOROCO) corpus (Moldavian and Romanian Dialectal Corpus):

- **Method A — LDA** (classical probabilistic topic model)
- **Method B — BERTopic** (transformer-based clustering pipeline)

The goal is to train both methods on the same train/test split, evaluate them with comparable metrics (C_v coherence, NMI, Purity), and produce a fair side-by-side comparison.

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
│   │   └── evaluate.py      # T6 NMI / Purity on test
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

## BERTopic CLI workflow

Once `data/processed/{train,test}.parquet` is available, run the whole Phase 2 pipeline:

```bash
python scripts/encode_docs.py          # T1 — encode train + test with RoBERT
python scripts/fit_bertopic.py         # T2 — fit BERTopic and save model
python scripts/inspect_topics.py       # T3 — topic table + 3 HTML viz
python scripts/hp_sweep.py             # T4 — hyperparameter sweep
python scripts/embedding_ablation.py   # T5 — RoBERT vs multilingual MPNet
python scripts/evaluate_on_test.py     # T6 — NMI / Purity / confusion matrix
```

Or run all of them in order:

```bash
python scripts/run_bertopic_pipeline.py
```

Equivalently, open `notebooks/03_bertopic_full.ipynb` and run cells top-to-bottom.

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
| 0 | Joint setup (repo, env, deps) | in progress |
| 1 | Data preparation | not started |
| 2 | Independent modeling (LDA + BERTopic) | not started |
| 3 | Joint comparison | not started |
| 4 | Final report | not started |
| 5 | Presentation | not started |

---

## Dataset

**MOROCO** — Moldavian and Romanian Dialectal Corpus. ~33k news samples labeled with one of 6 topic categories (culture, finance, politics, science, sports, tech) and 2 dialects (Romanian / Moldavian). Source: [butnaruandrei/MOROCO](https://github.com/butnaruandrei/MOROCO).

> ⚠️ Raw data files are **not** committed. Download them into `data/raw/` after cloning. See [`data/raw/README.md`](data/raw/README.md) for instructions.

---

## License

MIT — see [`LICENSE`](LICENSE).
