# Team Task Breakdown — Step by Step

**Project:** Topic Modeling on MOROCO (BERTopic vs. LDA)
**Team:** Ciorascu Mihai (LDA) & Buha Tudor (BERTopic)

---

## Project Timeline — High-Level View

| Phase | Description | Who |
|---|---|---|
| **Phase 0** | Joint setup | Both |
| **Phase 1** | Data preparation | One writes, both use |
| **Phase 2** | Independent modeling | Mihai (LDA) \| Tudor (BERTopic) |
| **Phase 3** | Joint comparison | Both |
| **Phase 4** | Documentation | Both |
| **Phase 5** | Presentation prep | Both |

---

## Phase 0 — Joint Setup

> **Who:** Both together
> **Estimated time:** ~1 day
> **Goal:** Establish a shared workspace and confirm the project plan.

### Joint Tasks

- [ ] **0.1** Create a shared GitHub repo (e.g., `nlp-moroco-topic-modeling`)
- [ ] **0.2** Add a clean folder structure:
  ```
  /data            ← raw + processed MOROCO files
  /notebooks       ← exploratory Jupyter notebooks
  /src             ← reusable Python modules
      /preprocessing
      /lda         (Mihai's code)
      /bertopic    (Tudor's code)
      /evaluation  (shared metrics)
  /results         ← output tables, plots, models
  /docs            ← final report + presentation
  /requirements.txt
  /README.md
  ```
- [ ] **0.3** Agree on Python version (3.10+) and create a virtual env
- [ ] **0.4** Add core dependencies to `requirements.txt`:
  - `numpy`, `pandas`, `scikit-learn`, `gensim`, `spacy`
  - `bertopic`, `sentence-transformers`, `umap-learn`, `hdbscan`
  - `pyLDAvis`, `matplotlib`, `seaborn`
- [ ] **0.5** Decide on a final due date and sub-deadlines for each phase
- [ ] **0.6** Decide who installs what:
  - **Mihai** → spaCy Romanian model (`ro_core_news_sm`)
  - **Tudor** → `readerbench/robert-base` from HuggingFace
- [ ] **0.7** Read the assignment requirements together and confirm that both methods + comparison satisfy them

---

## Phase 1 — Data Preparation

> **Who:** ONE person writes the preprocessing module, BOTH use it.
> **Recommendation:** Mihai handles preprocessing (LDA needs more of it: lemmatization, stop-word removal, stricter cleaning).

### Mihai's Tasks (Phase 1, ~1–2 days)

- [ ] **1.1** Download MOROCO from GitHub (or use the local zip)
- [ ] **1.2** Inspect dataset structure:
  - how many files
  - what columns (text, label, dialect)
  - language: Romanian or Moldavian?
  - topic distribution
- [ ] **1.3** Write `src/preprocessing/load.py`:
  - function `load_moroco()` → pandas DataFrame
  - columns: `id`, `text`, `dialect`, `topic`
  - sanity-check encoding (UTF-8, diacritics intact)
- [ ] **1.4** Write `src/preprocessing/clean.py`:
  - `normalize_diacritics(text)` — fix ș/ț/ă/â/î
  - `lowercase(text)`
  - `remove_punctuation(text)`
  - `remove_digits(text)`
  - `collapse_whitespace(text)`
- [ ] **1.5** Write `src/preprocessing/tokenize.py`:
  - `tokenize(text)` → `list[str]`
  - remove Romanian stop words (use a known Romanian stop-word list)
  - `lemmatize_tokens(tokens)` using spaCy `ro_core_news_sm` — needed for **LDA only**; BERTopic uses raw text
- [ ] **1.6** Write `src/preprocessing/split.py`:
  - `train_test_split(df, test_size=0.2, stratify='topic', random_state=42)`
  - Save splits to disk:
    - `data/processed/train.parquet`
    - `data/processed/test.parquet`
  - Print stats: total docs, per-topic counts in each split
- [ ] **1.7** Write a small smoke-test notebook: `notebooks/01_data_inspection.ipynb`
  - Load MOROCO
  - Show 5 examples per topic
  - Plot topic distribution
  - Plot text length distribution
- [ ] **1.8** **HAND OFF to Tudor:**
  - A clean `train.parquet` + `test.parquet` on disk
  - A working preprocessing module he can import:
    ```python
    from src.preprocessing import load_moroco, clean_text
    ```

### Tudor's Tasks During Phase 1 (in parallel)

While Mihai handles preprocessing, Tudor sets up the BERTopic side:

- [ ] **1.A** Install and verify HuggingFace + sentence-transformers
- [ ] **1.B** Download `readerbench/robert-base` + tokenizer locally (avoid surprise downloads later)
- [ ] **1.C** Write a tiny test: encode 10 sentences with RoBERT, verify shape `[10 × 768]` and no errors on Romanian text
- [ ] **1.D** Confirm GPU availability if using a local machine, OR set up Google Colab / Kaggle with a notebook
- [ ] **1.E** Read the BERTopic documentation ([maartengr.github.io/BERTopic](https://maartengr.github.io/BERTopic/)) and the original paper (Grootendorst, 2022)
- [ ] **1.F** Bookmark UMAP and HDBSCAN docs for later parameter tuning

---

## Phase 2 — Independent Modeling

> **Who:** Each works in parallel on their own method.
> **Estimated time:** ~1–2 weeks each.

### Mihai's Tasks — Method A: LDA (~1.5 weeks)

> **Goal:** Train and evaluate an LDA model on the MOROCO training split.

#### Step M1 — Build the vocabulary and corpus

- [ ] **M1.1** Load `train.parquet`
- [ ] **M1.2** For each document: apply `tokenize()` + `lemmatize()`
- [ ] **M1.3** Build a Gensim `Dictionary` from training tokens
- [ ] **M1.4** Filter the dictionary:
  ```python
  dictionary.filter_extremes(no_below=5, no_above=0.8)
  ```
- [ ] **M1.5** Convert all docs to BoW format
- [ ] **M1.6** Save the dictionary and corpus to disk:
  - `results/lda/dictionary.dict`
  - `results/lda/corpus.mm`

#### Step M2 — Train LDA models with varying K

- [ ] **M2.1** Define K values to try: `K ∈ {5, 6, 8, 10, 12, 15, 20}`
- [ ] **M2.2** For each K, train an `LdaModel`:
  ```python
  LdaModel(corpus, id2word=dictionary, num_topics=K,
           passes=50, iterations=400, random_state=42)
  ```
- [ ] **M2.3** Save each trained model to `results/lda/model_K{N}.gensim`
- [ ] **M2.4** Compute C_v coherence for each K using `gensim.CoherenceModel`
- [ ] **M2.5** Plot coherence curve (K on x-axis, C_v on y-axis)
- [ ] **M2.6** Pick the BEST K (highest coherence or clear elbow)

#### Step M3 — Inspect the chosen LDA model

- [ ] **M3.1** Print the top 10 keywords for each topic
- [ ] **M3.2** Manually label each topic (e.g., `"Topic 3 → Politics"`)
- [ ] **M3.3** Run pyLDAvis and save the interactive HTML visualization → `results/lda/ldavis.html`
- [ ] **M3.4** For each test document: get its dominant topic
- [ ] **M3.5** Build a confusion matrix:
  - rows = LDA topic id
  - columns = true MOROCO category
- [ ] **M3.6** Compute alignment metrics: **Purity**, **NMI** between LDA topics and MOROCO labels

#### Step M4 — Document Method A

- [ ] **M4.1** Write `notebooks/02_lda_full.ipynb` (clean, end-to-end)
- [ ] **M4.2** Save all numbers/plots into `results/lda/`
- [ ] **M4.3** Write a short Method A report (~2 pages):
  - theory recap
  - implementation choices + hyperparameters
  - coherence curve plot
  - final topic table (topic id, top 10 words, my label)
  - alignment to MOROCO categories (NMI, purity)
  - discussion of strengths/weaknesses observed

#### Mihai's deliverables at end of Phase 2

- ✅ Trained LDA model + dictionary
- ✅ Coherence curve plot
- ✅ pyLDAvis HTML
- ✅ Topic-keyword table with manual labels
- ✅ NMI / Purity numbers vs. MOROCO ground truth
- ✅ Method A report draft (2 pages)

---

### Tudor's Tasks — Method B: BERTopic (~1.5 weeks)

> **Goal:** Train and evaluate a BERTopic model on the MOROCO training split.

#### Step T1 — Generate sentence embeddings

- [ ] **T1.1** Load `train.parquet` (use Mihai's preprocessing for text)
- [ ] **T1.2** Choose primary embedding model: `readerbench/robert-base` (Romanian-specific)
- [ ] **T1.3** Encode all training docs in batches (`batch_size=32`) — be careful with `max_length=128` truncation
- [ ] **T1.4** Save embeddings to disk: `results/bertopic/embeddings_train.npy` (so you don't recompute every time)
- [ ] **T1.5** Same for test set → `embeddings_test.npy`

#### Step T2 — Configure and fit BERTopic

- [ ] **T2.1** Build the BERTopic pipeline:
  ```python
  from bertopic import BERTopic
  from umap import UMAP
  from hdbscan import HDBSCAN

  umap_model = UMAP(n_neighbors=15, n_components=5,
                    min_dist=0.0, random_state=42)
  hdbscan_model = HDBSCAN(min_cluster_size=30,
                          metric='euclidean',
                          prediction_data=True)

  topic_model = BERTopic(
      umap_model=umap_model,
      hdbscan_model=hdbscan_model,
      language='multilingual',
      calculate_probabilities=True,
      verbose=True
  )
  ```
- [ ] **T2.2** Fit on training docs:
  ```python
  topics, probs = topic_model.fit_transform(docs_train, embeddings_train)
  ```
- [ ] **T2.3** Save the fitted model: `topic_model.save('results/bertopic/model_main')`

#### Step T3 — Inspect discovered topics

- [ ] **T3.1** Get topic info: `topic_model.get_topic_info()` → number of topics, doc count per topic
- [ ] **T3.2** For each topic: print top 10 keywords (c-TF-IDF based)
- [ ] **T3.3** Manually label each topic
- [ ] **T3.4** Note the size of the outlier topic (`-1`)
- [ ] **T3.5** Save visualizations:
  - `topic_model.visualize_topics()` → `topics_2d.html`
  - `topic_model.visualize_barchart()` → `barchart.html`
  - `topic_model.visualize_heatmap()` → `heatmap.html`

#### Step T4 — Hyperparameter sensitivity

- [ ] **T4.1** Vary `min_cluster_size ∈ {10, 20, 30, 50}`
- [ ] **T4.2** Vary `n_neighbors ∈ {5, 15, 30}`
- [ ] **T4.3** For each config, record:
  - number of topics discovered
  - outlier proportion
  - C_v coherence
- [ ] **T4.4** Pick the BEST config (highest coherence, low outlier rate)

#### Step T5 — Embedding ablation

- [ ] **T5.1** Re-run BERTopic with a different embedding model: `paraphrase-multilingual-mpnet-base-v2`
- [ ] **T5.2** Compare against RoBERT version on:
  - coherence
  - NMI vs. MOROCO labels
  - visual quality of clusters
- [ ] **T5.3** Report which embedding is better for Romanian news

#### Step T6 — Test-set evaluation

- [ ] **T6.1** Use `topic_model.transform(docs_test, embeddings_test)` → get topic assignment for each test doc
- [ ] **T6.2** Build confusion matrix vs. MOROCO categories
- [ ] **T6.3** Compute NMI, Purity on the test split

#### Step T7 — Document Method B

- [ ] **T7.1** Write `notebooks/03_bertopic_full.ipynb`
- [ ] **T7.2** Save all artifacts into `results/bertopic/`
- [ ] **T7.3** Write a short Method B report (~2 pages):
  - theory recap (4-stage pipeline)
  - embedding model choice + ablation result
  - final topic table with labels
  - visualizations
  - alignment to MOROCO + discussion

#### Tudor's deliverables at end of Phase 2

- ✅ Trained BERTopic model + saved embeddings
- ✅ Topic-keyword table with manual labels
- ✅ Interactive HTML visualizations (3 of them)
- ✅ Hyperparameter sensitivity table
- ✅ Embedding ablation result (RoBERT vs. multilingual)
- ✅ NMI / Purity on the test set
- ✅ Method B report draft (2 pages)

---

## Phase 3 — Joint Comparison

> **Who:** Both together
> **Estimated time:** ~3 days
> **Goal:** Combine the two independent results into a fair comparison.

### Joint Tasks

- [ ] **3.1** Pool the final numbers from both methods into a single table:

  | Metric | LDA (Mihai) | BERTopic (Tudor) |
  |---|---|---|
  | C_v coherence | ? | ? |
  | Topic diversity | ? | ? |
  | NMI vs MOROCO | ? | ? |
  | Purity | ? | ? |
  | # topics found | ? | ? |
  | Training time | ? | ? |

- [ ] **3.2** Side-by-side topic table: For each MOROCO category, show which LDA topic AND which BERTopic topic best matches it. Compare the top-10 keywords from each.
- [ ] **3.3** Qualitative analysis (do this together):
  - Look at 20 test documents
  - For each: what topic did LDA assign? BERTopic? True label?
  - Where do they disagree, and why?
- [ ] **3.4** Identify common error patterns:
  - Cross-dialect confusion (Romanian vs. Moldavian splits)
  - Short-text errors
  - Topic-overlap errors (politics ↔ finance)
- [ ] **3.5** Write the **Comparison section** of the report (~2 pages):
  - Quantitative table
  - Qualitative comparison
  - Strengths/weaknesses of each method
  - Recommendation: when to use which

---

## Phase 4 — Documentation

> **Who:** Both
> **Estimated time:** ~1 week
> **Goal:** Produce the final deliverable document.

### Final Report Structure

| Section | Title | Author |
|---|---|---|
| 1 | Introduction | Mihai + Tudor |
| 2 | Theoretical Background (LDA theory + theoretical-report link) | Mihai |
| 3 | BERTopic Method (4-stage pipeline + theory) | Tudor |
| 4 | Dataset & Preprocessing (MOROCO, train/test split, diagram) | Mihai |
| 5.1 | LDA Implementation | Mihai |
| 5.2 | BERTopic Implementation | Tudor |
| 6.1 | LDA Results | Mihai |
| 6.2 | BERTopic Results | Tudor |
| 6.3 | Comparison | Both |
| 7 | Discussion & Error Analysis | Both |
| 8 | Conclusion | Both |
| — | References, Appendix | Both |

### Workflow Tasks

- [ ] **4.1** Set up a shared LaTeX/Word/Google Docs document
- [ ] **4.2** Each writes their own sections first
- [ ] **4.3** Cross-review each other's sections (catch errors, typos)
- [ ] **4.4** Add diagrams (pipeline, comparison flowcharts)
- [ ] **4.5** Add all required figures (coherence curve, pyLDAvis screenshot, BERTopic 2D map, confusion matrices)
- [ ] **4.6** Final formatting pass + reference check
- [ ] **4.7** Export PDF

---

## Phase 5 — Presentation Prep

> **Who:** Both
> **Estimated time:** ~3 days
> **Goal:** Build slides for the project defense.

- [ ] **5.1** Decide structure (10–12 slides total):
  - 1 × title
  - 1 × problem statement
  - 1 × dataset
  - 2 × LDA (theory + results) — *Mihai presents*
  - 2 × BERTopic (theory + results) — *Tudor presents*
  - 1 × hyperparameter / ablation
  - 1 × comparison table
  - 1 × conclusion + future work
  - 1 × Q&A
- [ ] **5.2** Each builds their own method's slides
- [ ] **5.3** Joint slides: title, intro, comparison, conclusion
- [ ] **5.4** Add speaker notes (~1 minute per slide)
- [ ] **5.5** Practice 2–3 full run-throughs together
- [ ] **5.6** Time the talk (target: 8–10 minutes if Q&A separate)

---

## Quick Reference — Who Does What

| Person | Phase 0<br/>Setup | Phase 1<br/>Data | Phase 2<br/>Modeling | Phase 3<br/>Compare | Phase 4<br/>Report | Phase 5<br/>Slides |
|---|---|---|---|---|---|---|
| **Mihai** | repo + env | preprocessing + data load | LDA full pipeline | pool + discuss | LDA sections | LDA slides + intro |
| **Tudor** | repo + env | embedding setup | BERTopic full + ablation | pool + discuss | BERTopic sections | BERTopic slides + conclusion |

---

## Accountability Checkpoints

### Suggested Weekly Sync (15 min on Discord/Zoom)

- What did each person finish this week?
- What's blocking progress?
- Any code/data hand-offs needed?

### Hard Deadlines (counting backwards from due date)

| Days before due | Milestone |
|---|---|
| **Due − 14 days** | Phase 2 complete (both methods working) |
| **Due − 10 days** | Phase 3 comparison done |
| **Due − 4 days** | Final report draft done |
| **Due − 1 day** | Slides + practice run done |
