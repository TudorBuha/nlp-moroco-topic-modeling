# Topic Modeling on MOROCO — BERTopic vs. LDA

**Status:** draft skeleton — placeholders marked `_TBD_` are filled in by the
saved artifacts under `results/` after the pipeline runs end-to-end.

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

**Per-topic distribution** (filled by `notebooks/01_data_inspection.ipynb`):

| Topic | Train count | Test count |
|---|---|---|
| culture | _TBD_ | _TBD_ |
| finance | _TBD_ | _TBD_ |
| politics | _TBD_ | _TBD_ |
| science | _TBD_ | _TBD_ |
| sports | _TBD_ | _TBD_ |
| tech | _TBD_ | _TBD_ |

**Train/test split.** We use a **stratified 80/20 split** on the topic column with `random_state=42`, persisted as `data/processed/train.parquet` and `data/processed/test.parquet`. Both methods consume the *same* split so their results are directly comparable.

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
| `src.lda.*` | _TBD_ | Method A pipeline (Step M1–M4) |

### CLI entry points

| Step | Command | Output |
|---|---|---|
| T1 | `python scripts/encode_docs.py` | `results/bertopic/embeddings_{train,test}.npy` |
| T2 | `python scripts/fit_bertopic.py` | `results/bertopic/model_main/` |
| T3 | `python scripts/inspect_topics.py` | `topic_keywords.csv`, three `*.html` |
| T4 | `python scripts/hp_sweep.py` | `hp_sweep.csv` |
| T5 | `python scripts/embedding_ablation.py` | `ablation.csv` |
| T6 | `python scripts/evaluate_on_test.py` | `test_evaluation.json`, `test_confusion_matrix.csv` |
| T8 | `python scripts/stability_analysis.py` | `stability_runs.csv`, `stability_summary.csv`, `bootstrap_*.json` |
| all | `python scripts/run_bertopic_pipeline.py` | everything above |
| GUI | `streamlit run app/streamlit_app.py` | runs the application on http://localhost:8501 |

### Tests

`pytest` covers `clean_text`, `purity_score`, `nmi_score`, `build_confusion_matrix`, `bootstrap_metric`, `topic_keyword_table`, `outlier_proportion`, `n_topics`, all module imports, and a smoke test on the Streamlit app module — none of which need MOROCO data, so they run in seconds.

---

## 4. Experiments and results

> Numbers come from the saved CSVs / JSONs in `results/` after running the pipeline end-to-end. Until then everything is `_TBD_`.

### 4.1 LDA results

**Coherence sweep over K.** Source: `results/lda/coherence_curve.{csv,png}`.

| K | C_v |
|---|---|
| 5 | _TBD_ |
| 6 | _TBD_ |
| 8 | _TBD_ |
| 10 | _TBD_ |
| 12 | _TBD_ |
| 15 | _TBD_ |
| 20 | _TBD_ |

Chosen \( K \): **_TBD_** (highest coherence / clearest elbow).

**Topic table (top-10 keywords per topic).** Source: `results/lda/topic_keywords.csv`.

| Topic id | Top-10 keywords | Manual label |
|---|---|---|
| 0 | _TBD_ | _TBD_ |
| ... | | |

**Test-set alignment.** Source: `results/lda/test_evaluation.json`.

| Metric | Value |
|---|---|
| NMI vs MOROCO topics | _TBD_ |
| Purity | _TBD_ |
| Outlier proportion | n/a (LDA assigns every doc) |

### 4.2 BERTopic results

**Discovered topics.** Source: `results/bertopic/topic_keywords.csv` and `topic_keywords_labeled.csv`.

| Topic id | Count | Top-10 keywords | Manual label |
|---|---|---|---|
| -1 (outlier) | _TBD_ | _TBD_ | — |
| 0 | _TBD_ | _TBD_ | _TBD_ |
| ... | | | |

**Visualizations.** `results/bertopic/topics_2d.html`, `barchart.html`, `heatmap.html` — embedded live in the app's BERTopic tab; screenshots in this PDF.

**Hyperparameter sensitivity (T4).** Sweep over `min_cluster_size ∈ {10,20,30,50}` × `n_neighbors ∈ {5,15,30}`. Source: `results/bertopic/hp_sweep.csv`. Selection criterion: maximize \( c\_v \times (1 - \text{outlier\_pct}) \).

**Embedding ablation (T5).** RoBERT vs `paraphrase-multilingual-mpnet-base-v2`. Source: `results/bertopic/ablation.csv`.

| Embedding | n_topics | outlier % | C_v | NMI | Purity |
|---|---|---|---|---|---|
| `readerbench/robert-base` | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| `mpnet-multilingual` | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |

**Test-set alignment.** Source: `results/bertopic/test_evaluation.json`.

| Metric | Value |
|---|---|
| NMI vs MOROCO topics | _TBD_ |
| Purity | _TBD_ |
| Outlier proportion (test) | _TBD_ |

### 4.3 Stability and confidence intervals (Step T8)

Two complementary techniques addressing *"are these numbers reliable?"*.

**Multi-seed runs.** N = 10 seeds varying only the UMAP `random_state`. Source: `stability_runs.csv` / `stability_summary.csv`.

| Metric | mean | std |
|---|---|---|
| `n_topics` | _TBD_ | _TBD_ |
| `c_v_train` | _TBD_ | _TBD_ |
| `outlier_pct_test` | _TBD_ | _TBD_ |
| `nmi_test` | _TBD_ | _TBD_ |
| `purity_test` | _TBD_ | _TBD_ |

**Bootstrap percentile CIs.** B = 1000 resamples on the test predictions. Source: `bootstrap_nmi.json`, `bootstrap_purity.json`.

| Metric | Test point estimate | Bootstrap mean | 95 % CI |
|---|---|---|---|
| NMI | _TBD_ | _TBD_ | _[TBD, TBD]_ |
| Purity | _TBD_ | _TBD_ | _[TBD, TBD]_ |

(Histogram with the 95 % CI markers: `results/bertopic/bootstrap_distributions.png`.)

### 4.4 Side-by-side comparison

| Metric | LDA | BERTopic |
|---|---|---|
| C_v coherence (best config) | _TBD_ | _TBD_ |
| NMI vs MOROCO | _TBD_ | _TBD_ |
| Purity | _TBD_ | _TBD_ |
| # topics found | _TBD_ | _TBD_ |
| Training time | _TBD_ | _TBD_ |
| Outlier handling | every doc has a topic mix | explicit outlier topic |

### 4.5 Qualitative analysis

Look at 20 sampled test documents:

- For each: LDA's dominant topic, BERTopic's topic id, true MOROCO label.
- Categorize disagreements: cross-dialect (RO↔MD), short-text errors, topic-overlap errors (e.g. politics ↔ finance).

Discussion paragraph: which method is better for which kind of document, and why.

### 4.6 Conclusion

(Short paragraph summarizing the headline finding once the numbers are in: which method wins on which metric, and the practical recommendation for someone who wants to do topic modeling on Romanian news.)

---

### References

- Blei, D. M., Ng, A. Y., & Jordan, M. I. (2003). *Latent Dirichlet Allocation*. JMLR 3.
- Grootendorst, M. (2022). *BERTopic: Neural topic modeling with a class-based TF-IDF procedure*. arXiv:2203.05794.
- Butnaru, A. M., & Ionescu, R. T. (2019). *MOROCO: The Moldavian and Romanian Dialectal Corpus*. ACL.
- McInnes, L., Healy, J., & Melville, J. (2018). *UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction*. arXiv:1802.03426.
- Campello, R., Moulavi, D., & Sander, J. (2013). *Density-Based Clustering Based on Hierarchical Density Estimates*. PAKDD.
- Röder, M., Both, A., & Hinneburg, A. (2015). *Exploring the Space of Topic Coherence Measures*. WSDM.
- Efron, B., & Tibshirani, R. (1993). *An Introduction to the Bootstrap*. Chapman & Hall.
