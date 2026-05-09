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
| Ciorăscu Mihai | _TBD_ | **LDA** |

See [`docs/Team_Task_Breakdown.md`](docs/Team_Task_Breakdown.md) for the full step-by-step plan.

---

## Repository structure

```
.
├── data/
│   ├── raw/             # MOROCO source files (gitignored)
│   └── processed/       # train.parquet / test.parquet (gitignored)
├── notebooks/
│   ├── 01_data_inspection.ipynb     # Mihai
│   ├── 02_lda_full.ipynb            # Mihai
│   └── 03_bertopic_full.ipynb       # Tudor
├── src/
│   ├── preprocessing/   # shared text loading, cleaning, tokenization (Mihai)
│   ├── lda/             # LDA training + evaluation (Mihai)
│   ├── bertopic/        # BERTopic pipeline + ablations (Tudor)
│   └── evaluation/      # shared metrics: NMI, Purity, coherence helpers
├── results/
│   ├── lda/             # trained models, dictionary, pyLDAvis HTML, plots
│   └── bertopic/        # embeddings, fitted model, visualizations
├── docs/                # final report, presentation, planning docs
├── requirements.txt
└── README.md
```

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

### 4. Per-person extras

**Mihai (LDA):**
```bash
python -m spacy download ro_core_news_sm
```

**Tudor (BERTopic):**
The first time you run `src/bertopic/`, the `readerbench/robert-base` model (~500MB) will be downloaded from HuggingFace into your local cache. You can pre-fetch it with:
```bash
python -c "from transformers import AutoModel, AutoTokenizer; AutoTokenizer.from_pretrained('readerbench/robert-base'); AutoModel.from_pretrained('readerbench/robert-base')"
```

---

## Project phases

| Phase | Description | Owner | Status |
|---|---|---|---|
| 0 | Joint setup (repo, env, deps) | Both | in progress |
| 1 | Data preparation | Mihai writes, both use | not started |
| 2 | Independent modeling | Mihai (LDA) / Tudor (BERTopic) | not started |
| 3 | Joint comparison | Both | not started |
| 4 | Final report | Both | not started |
| 5 | Presentation | Both | not started |

---

## Dataset

**MOROCO** — Moldavian and Romanian Dialectal Corpus. ~33k news samples labeled with one of 6 topic categories (culture, finance, politics, science, sports, tech) and 2 dialects (Romanian / Moldavian). Source: [butnaruandrei/MOROCO](https://github.com/butnaruandrei/MOROCO).

> ⚠️ Raw data files are **not** committed. Download them into `data/raw/` after cloning. See [`data/raw/README.md`](data/raw/README.md) for instructions.

---

## License

MIT — see [`LICENSE`](LICENSE).
