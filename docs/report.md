# Topic Modeling on MOROCO — BERTopic vs. LDA

**Status:** end-to-end results filled in from a stratified 6,000-document
demo subset of MOROCO (1,000 per topic class, 80 / 20 train-test split,
seed 42). The same code runs on the full ~33k corpus by passing
`--max-per-class -1` to `scripts/preprocess.py`.

This is the joint report. Section numbering follows the assignment spec:

> 1. Problem statement
> 2. Proposed solution — 2.1 Theoretical aspects · 2.2 Dataset · 2.3 Application + diagram
> 3. Implementation — libraries / functions
> 4. Experiments and results

---

## 1. Problem statement

The NLP task is **unsupervised topic discovery on Romanian news**.

Given a corpus of news articles labeled with broad topical categories (politics, sports, finance, science, culture, tech), we want a set of topics — each represented as a coherent group of words — that the model **discovers from the text alone**, without using the gold labels at training time.

Concretely, given a collection of documents \( D = \{d_1, \dots, d_N\} \), we seek:

1. A partition (or soft assignment) of \( D \) into \( K \) topics.
2. For each topic \( k \), a ranked list of words that characterize it.
3. A way to *evaluate* the discovered topics, both intrinsically (do the words go together?) and extrinsically (do the topics align with the gold MOROCO categories?).

We compare two methodologically very different approaches to this problem:

- **Method A — Latent Dirichlet Allocation (LDA)** — the classical probabilistic generative model.
- **Method B — BERTopic** — a modern transformer-based clustering pipeline.

The deliverable includes the trained models, the comparison results, and an interactive web application (see §2.3) that lets the user explore both models side-by-side.

---

## 2. Proposed solution

### 2.1 Theoretical aspects

#### 2.1.1 Latent Dirichlet Allocation (LDA) — Method A

LDA (Blei, Ng & Jordan, 2003) is a *generative probabilistic* model that posits the following process for each document \( d \) in a corpus:

1. Draw a topic distribution \( \theta_d \sim \text{Dirichlet}(\alpha) \) over \( K \) topics.
2. For each word position \( n \) in the document:
   - Draw a topic \( z_{d,n} \sim \text{Multinomial}(\theta_d) \).
   - Draw a word \( w_{d,n} \sim \text{Multinomial}(\beta_{z_{d,n}}) \), where \( \beta_k \) is the word distribution of topic \( k \).

The latent variables \( \theta_d \), \( z_{d,n} \), and \( \beta_k \) are estimated by approximate inference (variational EM in our implementation, via Gensim). The user must choose \( K \) up-front; we sweep over \( K \in \{5, 6, 8, 10, 12, 15, 20\} \) and pick the value that maximizes the C_v topic coherence.

Inputs to LDA must be **bag-of-words (BoW)** vectors over a curated vocabulary; this is why LDA needs aggressive preprocessing (lowercasing, punctuation/digit removal, stop-word removal, lemmatization) — none of those steps are free in the model.

#### 2.1.2 BERTopic — Method B

BERTopic (Grootendorst, 2022) is a *clustering-based* topic modeling pipeline composed of four interchangeable stages:

1. **Embed** — each document \( d \) is mapped to a dense vector \( \mathbf{e}_d \in \mathbb{R}^{768} \) using a pre-trained transformer encoder. We use `readerbench/robert-base` (a Romanian BERT) and mean-pool the last hidden state with attention masking.
2. **Reduce** — the embeddings are projected to a lower-dimensional space with **UMAP** (\( \mathbb{R}^{768} \to \mathbb{R}^{5} \), `n_neighbors=15`, `min_dist=0`), preserving local neighborhood structure so density-based clustering becomes tractable.
3. **Cluster** — **HDBSCAN** is run on the reduced embeddings with `min_cluster_size=30`. Documents that don't fit any density cluster are flagged as **outliers** (topic id `-1`); this is one of BERTopic's main differences from LDA.
4. **Topic representation** — for each cluster we compute a **class-based TF-IDF (c-TF-IDF)**:
   \[
   c\text{-}TF\text{-}IDF_{w,c} = \text{tf}_{w,c} \cdot \log\!\left(1 + \frac{A}{\text{tf}_w}\right)
   \]
   where \( \text{tf}_{w,c} \) is the frequency of word \( w \) in cluster \( c \), \( \text{tf}_w \) the global frequency, and \( A \) the average document length. The top-N words per topic are the most distinctive words for that cluster relative to the rest.

Unlike LDA, the number of topics is **discovered** from the data (no \( K \) to set), and BERTopic operates on raw text — preprocessing matters far less because the encoder handles morphology and word order.

#### Why compare these two

| Aspect | LDA | BERTopic |
|---|---|---|
| Generative? | yes | no (discriminative clustering) |
| Choose K up-front? | yes | no (HDBSCAN finds it) |
| Order-aware? | no (BoW) | yes (transformer attention) |
| Outlier handling? | every doc has a topic mix | explicit outlier topic |
| Preprocessing sensitivity | high | low |
| Compute | CPU, fast | GPU-friendly, slow on CPU |

### 2.2 Dataset

**MOROCO** — *Moldavian and Romanian Dialectal Corpus* (Butnaru & Ionescu, 2019).

| Attribute | Value |
|---|---|
| Documents | ~33,000 news samples |
| Topic categories | 6 — culture, finance, politics, science, sports, tech |
| Dialects | 2 — Romanian, Moldavian |
| Encoding | UTF-8 (with diacritics: ș, ț, ă, â, î) |
| Source | [butnaruandrei/MOROCO](https://github.com/butnaruandrei/MOROCO) |

**Per-topic distribution** in the demo subset (saved to `data/processed/dataset_stats.csv` by `scripts/preprocess.py`):

| Topic | Train count | Test count | Total |
|---|---:|---:|---:|
| culture  | 796 | 200 | 996 |
| finance  | 800 | 200 | 1,000 |
| politics | 800 | 200 | 1,000 |
| science  | 800 | 200 | 1,000 |
| sports   | 789 | 197 | 986 |
| tech     | 800 | 200 | 1,000 |
| **Total**| **4,785** | **1,197** | **5,982** |

(18 documents in the original 6,000-row sample were dropped by the preprocessor for having fewer than 5 tokens after cleaning.)

**Train/test split.** We use a **stratified 80/20 split** on the topic column with `random_state=42`, persisted as `data/processed/train.parquet` and `data/processed/test.parquet`. Both methods consume the *same* split so their results are directly comparable. The full corpus has ~33,564 samples (politics 9,134, finance 8,534, sports 6,026, tech 4,658, science 2,920, culture 2,292) — the demo subset down-samples each class to 1,000 to keep CPU compute under ~30 minutes; the same pipeline scales to the full corpus given more time / a GPU.

### 2.3 Application

The deliverable includes an interactive web application built with **Streamlit** that wraps both trained models and lets a user explore them side-by-side without writing any code.

**Architecture diagram** — see [`docs/architecture.md`](architecture.md) for the full Mermaid source. High-level overview:

```
MOROCO raw → preprocessing → train/test parquet
                                  │
                ┌─────────────────┴─────────────────┐
                ▼                                   ▼
         LDA pipeline                       BERTopic pipeline
       (gensim BoW + EM)                (RoBERT → UMAP → HDBSCAN
                                          → c-TF-IDF)
                │                                   │
                └─────────────────┬─────────────────┘
                                  ▼
                       Saved artifacts in results/
                                  │
                                  ▼
                    app/streamlit_app.py
                    ├── Home
                    ├── Try it live (typed Romanian text → both models)
                    ├── LDA explorer (pyLDAvis + topic table)
                    ├── BERTopic explorer (3 HTML viz + search)
                    ├── Comparison (side-by-side metrics + confusion)
                    └── Stability (multi-seed + bootstrap CIs)
                                  │
                                  ▼
                                User
```

#### User-facing tabs

| Tab | What the user sees / does |
|---|---|
| **Home** | Project intro, dataset stats, headline comparison numbers, link to the report. |
| **Try it live** | Text area for a Romanian news article; both models predict and the app shows the assigned topic + top-10 keywords for each. |
| **LDA explorer** | Coherence curve, topic-keyword table with manual labels, embedded `pyLDAvis` HTML. |
| **BERTopic explorer** | Topic-keyword table, embedded `topics_2d` / `barchart` / `heatmap` HTMLs, keyword search. |
| **Comparison** | Side-by-side metrics (NMI, Purity, C_v, # topics, training time), confusion matrices, agreement on test set. |
| **Stability** | Multi-seed mean ± std bars, bootstrap-CI histograms, HP-sensitivity heatmap. |

#### How the app loads data

The app is a **read-only frontend** over the pipeline's saved artifacts. Every tab calls `app/utils.py::load_*` helpers that:

1. Resolve paths through `src.paths` (so the app and the CLI agree on locations).
2. If the artifact exists → load and render.
3. If missing → render a friendly card explaining which `python scripts/...` to run first.

This means the app *never crashes*, even when only some pipeline steps have been run. It also means the app is automatically up-to-date the moment you re-run a script.

---

## 3. Implementation — libraries and functions

The implementation is organized as a small Python package under `src/` with thin CLI wrappers under `scripts/` and a Streamlit frontend under `app/`.

### Libraries

| Concern | Library | Version (pinned in `requirements.txt`) |
|---|---|---|
| Numerical / data | `numpy`, `pandas`, `pyarrow` | numpy ≥1.26 <2.0, pandas ≥2.1 |
| Splits / metrics | `scikit-learn` | ≥1.4 — `train_test_split`, `normalized_mutual_info_score` |
| Tokenization & lemmatization (Romanian) | `spacy` + `ro_core_news_sm` | ≥3.7 |
| LDA training & coherence | `gensim` | ≥4.3 — `Dictionary`, `LdaModel`, `CoherenceModel` |
| LDA visualization | `pyLDAvis` | ≥3.4 — `gensim_models.prepare` |
| Sentence embeddings | `transformers`, `torch`, `sentence-transformers` | transformers ≥4.40, torch ≥2.2 |
| BERTopic | `bertopic` | ≥0.16 — `BERTopic.fit_transform`, `.transform`, `.visualize_topics`, `.visualize_barchart`, `.visualize_heatmap` |
| BERTopic vectorizer | `scikit-learn` | `CountVectorizer(stop_words=RO, ngram_range=(1,2), min_df=5)` for c-TF-IDF |
| Dimensionality reduction | `umap-learn` | ≥0.5 |
| Clustering | `hdbscan` | ≥0.8 |
| Plots | `matplotlib`, `seaborn`, `plotly` | matplotlib ≥3.8 |
| GUI | `streamlit` | ≥1.30 — `st.tabs`, `st.text_area`, `st.dataframe`, `st.cache_resource`, `streamlit.components.v1.html` |
| Testing | `pytest` | ≥8.0 |

### Module layout

| Module | Public functions | Purpose |
|---|---|---|
| `src.preprocessing.load` | `load_moroco` | read raw MOROCO into a DataFrame |
| `src.preprocessing.clean` | `clean_text`, `normalize_diacritics`, … | UTF-8 normalization, lowercasing, punctuation/digit stripping |
| `src.preprocessing.tokenize` | `tokenize`, `lemmatize_tokens` | RO stop-word removal, spaCy lemmatization |
| `src.preprocessing.split` | `train_test_split`, `save_splits` | stratified 80/20 split + parquet I/O |
| `src.bertopic.embeddings` | `RomanianEmbedder`, `encode_documents` | mean-pooled RoBERT embeddings, cached as `.npy` |
| `src.bertopic.model` | `BERTopicConfig`, `build_bertopic_model` | UMAP + HDBSCAN factory |
| `src.bertopic.training` | `fit_bertopic`, `save_model`, `load_model` | end-to-end fit/serialize |
| `src.bertopic.inspection` | `topic_keyword_table`, `outlier_proportion`, `n_topics`, `attach_manual_labels`, `save_topic_table` | topic introspection |
| `src.bertopic.visualizations` | `save_default_visualizations` | the three required HTMLs |
| `src.bertopic.coherence` | `compute_cv_coherence` | C_v via Gensim |
| `src.bertopic.hp_sweep` | `sweep`, `save_sweep` | hyperparameter grid |
| `src.bertopic.ablation` | `run_ablation`, `encode_with_sbert` | RoBERT vs MPNet |
| `src.bertopic.stability` | `run_multi_seed`, `summarize` | model-variance analysis |
| `src.bertopic.evaluate` | `evaluate_on_test`, `save_evaluation` | NMI / Purity / confusion on test |
| `src.evaluation.metrics` | `purity_score`, `nmi_score`, `build_confusion_matrix`, **`bootstrap_metric`** | metrics shared between LDA and BERTopic, including percentile bootstrap CIs |
| `src.lda.corpus`         | `tokenize_for_lda`, `build_corpus`, `save_corpus`, `load_corpus` | spaCy lemmatization → `Dictionary` → BoW |
| `src.lda.training`       | `fit_lda`, `sweep_k`, `save_lda`, `load_lda` | `LdaModel` factory + K sweep with C_v |
| `src.lda.inspection`     | `topic_keyword_table`, `attach_manual_labels`, `save_topic_table` | per-topic top-N keywords |
| `src.lda.visualizations` | `save_coherence_curve`, `save_pyldavis_html` | the PNG + interactive HTML |
| `src.lda.evaluate`       | `predict_dominant_topics`, `evaluate_lda_on_test`, `save_evaluation` | NMI / Purity / confusion on test |
| `src.lda.inference`      | `predict_topic_distribution` | top-N topic distribution for a single new document (used by the Streamlit app's *Try it live* tab) |

### CLI entry points

| Step | Command | Output |
|---|---|---|
| 0a   | `python scripts/download_moroco.py`           | `data/raw/MOROCO/...` |
| 0b   | `python scripts/preprocess.py [--max-per-class N]` | `data/processed/{train,test}.parquet`, `dataset_stats.csv` |
| **Method A — LDA** | | |
| M1+M2+M3.5 | `python scripts/train_lda.py` | `dictionary.dict`, `train/test_corpus.mm`, `coherence_sweep.csv`, `coherence_curve.png`, `model_main.gensim`, `ldavis.html` |
| M3   | `python scripts/inspect_lda.py`               | `lda/topic_keywords.csv`, `topic_keywords_labeled.csv` |
| M4   | `python scripts/evaluate_lda.py`              | `lda/test_evaluation.json`, `lda/test_confusion_matrix.csv` |
| LDA all | `python scripts/run_lda_pipeline.py`       | everything above in order |
| **Method B — BERTopic** | | |
| T1   | `python scripts/encode_docs.py`               | `results/bertopic/embeddings_{train,test}.npy` |
| T2   | `python scripts/fit_bertopic.py`              | `results/bertopic/model_main/` |
| T3   | `python scripts/inspect_topics.py`            | `topic_keywords.csv`, three `*.html` |
| T4   | `python scripts/hp_sweep.py`                  | `hp_sweep.csv` |
| T5   | `python scripts/embedding_ablation.py`        | `ablation.csv` |
| T6   | `python scripts/evaluate_on_test.py`          | `test_evaluation.json`, `test_confusion_matrix.csv` |
| T8   | `python scripts/stability_analysis.py`        | `stability_runs.csv`, `stability_summary.csv`, `bootstrap_*.json` |
| BERTopic all | `python scripts/run_bertopic_pipeline.py` | everything above in order |
| GUI  | `streamlit run app/streamlit_app.py`          | runs the application on http://localhost:8501 |

### Tests

`pytest` covers `clean_text`, `purity_score`, `nmi_score`, `build_confusion_matrix`, `bootstrap_metric`, `topic_keyword_table`, `outlier_proportion`, `n_topics`, all module imports, and a smoke test on the Streamlit app module — none of which need MOROCO data, so they run in seconds.

---

## 4. Experiments and results

All numbers below come from the saved CSVs / JSONs in `results/`. The exact file each table is sourced from is given inline.

### 4.1 LDA results

**Coherence sweep over K.** Source: `results/lda/coherence_sweep.csv` (visualization: `coherence_curve.png`).

| K | C_v |
|---:|---:|
| 5  | 0.4222 |
| 6  | 0.4231 |
| 8  | 0.4286 |
| 10 | 0.4538 |
| 12 | 0.4908 |
| **15** | **0.4913** |
| 20 | 0.4686 |

Chosen K = **15** (highest C_v; the curve plateaus and starts to drop after 15).

**Topic table (top-10 keywords per topic).** Source: `results/lda/topic_keywords.csv`.

| id | Top-10 keywords | Manual label |
|---:|---|---|
| 0  | grad, temperatură, zonă, apă, până, oră, mare, gaz, kilometru, lună | weather / environment |
| 1  | leu, ban, milion, euro, lună, dolar, miliard, preț, mare, mult | finance |
| 2  | președinte, stat, ministru, lege, partid, declara, funcție, proiect, face, privind | politics |
| 3  | vot, oră, vota, alegere, primar, candidat, loc, oraș, electoral, persoană | elections |
| 4  | putea, studiu, știință, mult, cercetător, acesta, caz, dintre, acela, timp | science |
| 5  | putea, face, spune, acesta, acela, trebui, dată, dacă, mult, cum | (generic) |
| 6  | meci, echipă, doi, scor, fotbal, gol, minut, prim, partidă, juca | football |
| 7  | descoperi, planetă, afla, obiect, urmă, descoperire, zonă, mare, vin, kilometru | science (astronomy) |
| 8  | său, putea, acesta, mult, acela, nou, doi, timp, mare, sistem | (generic) |
| 9  | mașină, nou, model, companie, producător, producție, auto, piață, vehicul, produce | automotive / tech |
| 10 | mult, face, spune, putea, acesta, foarte, acela, când, său, cum | (generic) |
| 11 | doi, femeie, său, echipă, aur, francez, timp, scrie, roman, lună | culture / sports mix |
| 12 | eveniment, film, loc, acela, țară, spectacol, său, dintre, cadru, mult | culture (events / film) |
| 13 | loc, prim, doi, finală, turneu, mondial, câștiga, sportiv, trei, punct | sports (competition) |
| 14 | domeniu, cadru, dezvoltare, țară, proiect, economic, stat, afacere, serviciu, precum | economy / business |

About 10 of the 15 topics are clearly aligned with one of the 6 gold MOROCO categories; topics 5, 8, 10 capture generic news prose (fillers / discourse markers); topics 11 and 14 sit on category boundaries.

**Test-set alignment** (source: `results/lda/test_evaluation.json`):

| Metric | Value |
|---|---:|
| NMI vs MOROCO topics | **0.3459** |
| Purity                | **0.5906** |
| Documents with empty BoW (skipped) | 0 / 1,197 |
| Outlier handling      | n/a — every document has a topic mix |

### 4.2 BERTopic results

**Discovered topics** (source: `results/bertopic/topic_keywords.csv`). The c-TF-IDF vectorizer uses the same Romanian stop-word list as the LDA tokenizer and 1–2-gram features, so the keywords are directly comparable.

| id | Count | Top-10 keywords | Manual label |
|---:|---:|---|---|
| -1 | 718  | şi, că, va, când, dintre, până, ani, dacă, acest, două | (outlier) |
| 0  | 1079 | că, va, cadrul, şi, acest, despre, astăzi, către, transmite, loc | misc news prose |
| 1  | 666  | şi, va, că, meci, locul, primul, ani, două, scorul, meciul | sports |
| 2  | 536  | in, si, dupa, daca, pana, ani, va, doua, aceasta, fata | mixed (no diacritics articles) |
| 3  | 348  | că, va, putea, poate, acest, ani, dacă, știință, pot, până | science / opinion |
| 4  | 308  | şi, că, va, ani, foarte, era, acest, când, cei, dintre | culture / generic |
| 5  | 249  | şi, că, putea, ştiinţă, cercetătorii, va, dintre, studiu, ani, unui | science |
| 6  | 158  | şi, că, preşedintele, dacă, spus, va, ministru, trebuie, declarat, această | politics |
| 7  | 102  | sînt, că, cînd, va, știință, oamenii, decît, poate, cercetătorii, pînă | science (Moldavian dialect) |
| 8  | 91   | că, spus, dacă, despre, trebuie, vă, președintele, şi, privind, va | politics |
| 9  | 83   | şi, va, smartphone, putea, go4it, uri, scrie go4it, go4it ro, însă, telefonul | tech (smartphones) |
| 10 | 67   | şi, că, datelor, utilizatori, informaţii, datele, utilizatorilor, milioane, conturi, personale | tech (privacy / accounts) |
| 11 | 62   | şi, că, sînt, cînd, cercetătorii, decît, poate, acest, atunci cînd, studiu | science (Moldavian) |
| 12 | 59   | şi, lei, euro, 000, că, milioane, miliarde, anul, 2017, va | finance |
| 13 | 53   | ani, şi, faţa locului, locului, faţa, că, poliţiştii, autoturism, ani şi, accident | accident / news |
| 14 | 52   | ani, că, spital, doi, bărbatul, bărbat, mama, fața, vârstă, cei doi | crime / news |
| 15 | 49   | că, lei, va, dacă, mediu, putea, venituri, minim, ani, către | economy (wages, income) |
| 16 | 35   | şi, maşini, auto, că, magazine, producţie, va, 000, compania, milioane | automotive |
| 17 | 35   | grade, va, şi, temperaturile, vremea, vremea va, medie, până, munte, sud | weather |
| 18 | 35   | bani, lei, lei şi, bani şi, curs, moneda, valoarea, unică, patru, românesc | currency / finance |

20 unique cluster ids in total (one of which is the catch-all outlier topic `-1`). Note how BERTopic naturally separates the **science** topic into two distinct clusters along the **dialectal axis** (topics 7 + 11 collect Moldavian-spelling forms `sînt / cînd / decît / pînă`) — something LDA cannot do because the BoW collapses both spellings.

**Visualizations** are saved as standalone HTML and embedded live in the app's BERTopic tab:

| File | Renders |
|---|---|
| `results/bertopic/topics_2d.html` | UMAP projection of topic centroids on a 2-D plane. |
| `results/bertopic/barchart.html`  | Top-N c-TF-IDF keywords per topic, side by side. |
| `results/bertopic/heatmap.html`   | Topic-similarity matrix (cosine of c-TF-IDF representations). |

**Hyperparameter sensitivity (T4).** Sweep over `min_cluster_size ∈ {10, 20, 30, 50}` × `n_neighbors ∈ {5, 15, 30}`. Source: `results/bertopic/hp_sweep.csv`. Selection criterion: maximize `c_v × (1 − outlier_pct)`.

| min_cluster_size | n_neighbors | n_topics | outlier % | C_v   | score |
|---:|---:|---:|---:|---:|---:|
| 30 | 30 | 18 | 11.4% | 0.5527 | **0.4899** |
| 20 |  5 | 32 | 11.2% | 0.5437 | 0.4828 |
| 10 | 30 | 42 | 19.0% | 0.5821 | 0.4718 |
| 20 | 15 | 23 | 13.0% | 0.5249 | 0.4568 |
| 30 |  5 | 20 | 11.8% | 0.5040 | 0.4444 |
| 20 | 30 | 27 | 18.2% | 0.5411 | 0.4425 |
| 10 |  5 | 96 | 27.0% | 0.6024 | 0.4398 |
| 30 | 15 | 19 | 15.0% | 0.5152 | 0.4379 |
| 50 |  5 | 13 | 11.2% | 0.4676 | 0.4151 |
| 10 | 15 | 78 | 33.7% | 0.6086 | 0.4037 |
| 50 | 15 | 13 | 17.3% | 0.4588 | 0.3795 |
| 50 | 30 | —  | —     | —     | _failed (vocabulary collapsed under min_df=5)_ |

The default config used in §4.2 (`min_cluster_size=30, n_neighbors=15`, score 0.4379) is close to the optimum (`30, 30`, score 0.4899). Smaller `min_cluster_size` (10) maximizes raw C_v but at the cost of many tiny topics with high outlier proportions — bad for the side-by-side readability.

**Embedding ablation (T5).** RoBERT vs `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`. Source: `results/bertopic/ablation.csv`. The ablation relaxes `min_df` to 2 (from the default 5) so neither encoder fails the c-TF-IDF stage on small clusters; otherwise UMAP / HDBSCAN settings are identical.

| Embedding | n_topics | outlier % (test) | C_v (train) | NMI | Purity |
|---|---:|---:|---:|---:|---:|
| `readerbench/robert-base` (Romanian)               | **19** | 17.7 % | **0.5213** | **0.3759** | **0.5789** |
| `paraphrase-multilingual-mpnet-base-v2` (104 langs) | 3      | 0.0 %  | 0.4311     | 0.3388     | 0.3317     |

The Romanian-specific encoder produces ~6× more granular topics with substantially higher Purity. The multilingual MPNet treats Romanian as a generic Romance language and collapses everything into 3 mega-clusters that capture only the most prominent semantic axis — bad for topic discovery, fine for cross-lingual retrieval. This validates the choice of RoBERT for the main run.

**Test-set alignment** (source: `results/bertopic/test_evaluation.json`):

| Metric | Value |
|---|---:|
| NMI vs MOROCO topics    | **0.3343** |
| Purity                  | **0.5982** |
| Outlier proportion      | 3.7 % (44 / 1,197 documents) |
| Number of topics        | 19 (excl. outlier) |

### 4.3 Stability and confidence intervals (Step T8)

Two complementary techniques addressing *"are these numbers reliable?"*.

**Multi-seed runs** (N = 10 seeds varying only the UMAP `random_state`). Source: `stability_runs.csv` / `stability_summary.csv`.

| Metric | mean | std | min | max |
|---|---:|---:|---:|---:|
| `n_topics`         | 17.30 | 5.64  | 7    | 23   |
| `c_v_train`        | 0.5030 | 0.0316 | 0.4467 | 0.5352 |
| `outlier_pct_train`| 12.3 % | 6.9 % | 0.13 % | 20.2 % |
| `outlier_pct_test` | 15.1 % | 8.5 % | 0.08 % | 24.6 % |
| `nmi_test`         | **0.3722** | **0.0181** | 0.3425 | 0.3956 |
| `purity_test`      | **0.5510** | **0.0787** | 0.4027 | 0.6057 |

The headline finding here is that **the number of discovered topics is highly sensitive to the random seed** (7 → 23). NMI is much more stable (CV ≈ 5 %), Purity moderately so (CV ≈ 14 %). This is the classic UMAP / HDBSCAN warning: the 2-D layout is non-convex, so clustering on it is essentially random-walk-dependent. A practical mitigation (not implemented in the demo) is to ensemble several seeds and merge near-duplicate topics by c-TF-IDF cosine.

**Bootstrap percentile CIs** (B = 1000 resamples on the test predictions). Source: `bootstrap_nmi.json`, `bootstrap_purity.json`.

| Metric | Test point estimate | Bootstrap mean | 95 % CI |
|---|---:|---:|---|
| NMI    | 0.3343 | 0.3471 | [0.3245, 0.3703] |
| Purity | 0.5982 | 0.5998 | [0.5747, 0.6249] |

The test point estimate falls inside the bootstrap 95 % CI in both cases, as expected. Width of the CI is around ±0.013 (NMI) / ±0.025 (Purity) — narrower than the multi-seed std, which is consistent with sample variance < model variance for this corpus size.

### 4.4 Side-by-side comparison

| Metric                          | LDA (Method A) | BERTopic (Method B) |
|---|---:|---:|
| C_v coherence (best config)     | 0.4913 (K = 15) | 0.5527 (HP-best) / 0.5152 (default) |
| NMI vs MOROCO (test)            | **0.3459**      | 0.3343 |
| Purity (test)                   | 0.5906          | **0.5982** |
| # topics found                  | 15 (forced)     | 19 (+ outlier) |
| Outlier handling                | every doc gets a mix | explicit `-1` topic (3.7 % of test) |
| Seed-stability of metrics       | _not measured_  | NMI ± 0.018, Purity ± 0.079 |
| Training preprocessing required | aggressive (lemmatization, stop-words, BoW) | minimal (raw text → encoder) |
| Compute (4,785 train docs, CPU) | ~10 min         | ~10 min encode + ~1 min fit |

The two methods land in **the same NMI / Purity ballpark** on this corpus and subset size, with LDA marginally better on NMI (0.346 vs 0.334) and BERTopic marginally better on Purity (0.598 vs 0.591). The interesting differences are qualitative: BERTopic discovers a **dialect axis** (Moldavian-spelling clusters) that LDA's BoW cannot represent, while LDA's topic-keyword tables are more readable per topic at this corpus size because every topic is forced to have meaningful structure (no outlier escape hatch).

### 4.5 Qualitative analysis

- **BERTopic specialty topics:** tech 9 (`smartphone, go4it`), 10 (`utilizatori, informaţii, conturi, personale` — privacy), 17 (`grade, temperaturile, vremea` — weather forecasts), 18 (`bani, lei, leul, curs, moneda` — currency). LDA also recovers these (1 = leu/ban/euro, 9 = mașină/auto, 0 = grad/temperatură) but in a slightly less granular way.
- **Where BERTopic struggles:** clusters 0, 4, 8 absorb a long tail of generic news prose (`că, va, dacă, despre`); HDBSCAN merges anything not dense enough into a "cluster of leftovers" rather than the explicit outlier topic.
- **Where LDA struggles:** topics 5, 8, 10 are dominated by discourse markers (`putea, face, spune, acesta`) — the symmetric problem to BERTopic's "leftovers cluster", but distributed across every document via the soft assignment.
- **Cross-dialect sensitivity:** BERTopic separates Romanian and Moldavian science articles into distinct clusters (topic 5 vs topic 7 / 11). This is a *side effect of the embedding*, not a labeled goal. For a production deployment this might be a feature (dialect-aware clustering) or a bug (over-fragmentation by orthography).

### 4.6 Conclusion

For unsupervised topic discovery on Romanian news at this scale (≈6k documents) the two methods are **roughly tied on alignment with the gold MOROCO categories** (NMI ≈ 0.34, Purity ≈ 0.59). The choice between them is therefore driven by qualitative properties:

- Use **LDA** when you need a **fixed, interpretable taxonomy** with every document mixed across a small set of topics; preprocessing cost is high but inference is essentially free.
- Use **BERTopic** when you want **automatic K**, an **explicit "I don't know" topic** (outliers), and the ability to capture semantic / dialectal cues that BoW can't see; pay for it with a one-time GPU-friendly encoding and a brittle UMAP-seed dependency.

A practical recommendation: run both, then surface the agreement / disagreement breakdown as a diagnostic — exactly what the **Comparison** tab of the included Streamlit app does.

---

### References

- Blei, D. M., Ng, A. Y., & Jordan, M. I. (2003). *Latent Dirichlet Allocation*. JMLR 3.
- Grootendorst, M. (2022). *BERTopic: Neural topic modeling with a class-based TF-IDF procedure*. arXiv:2203.05794.
- Butnaru, A. M., & Ionescu, R. T. (2019). *MOROCO: The Moldavian and Romanian Dialectal Corpus*. ACL.
- McInnes, L., Healy, J., & Melville, J. (2018). *UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction*. arXiv:1802.03426.
- Campello, R., Moulavi, D., & Sander, J. (2013). *Density-Based Clustering Based on Hierarchical Density Estimates*. PAKDD.
- Röder, M., Both, A., & Hinneburg, A. (2015). *Exploring the Space of Topic Coherence Measures*. WSDM.
- Efron, B., & Tibshirani, R. (1993). *An Introduction to the Bootstrap*. Chapman & Hall.
