# Method B — BERTopic on MOROCO

**Companion of:** _Method A — LDA report_
**Status:** draft skeleton — fill in `_TBD_` after the pipeline runs end-to-end.

This document is the source for **Section 6.2 (BERTopic Results)** of the joint report.
Numbers come straight from `results/bertopic/`:

| Artifact | Source file |
|---|---|
| Topic-keyword table | `topic_keywords.csv` / `topic_keywords_labeled.csv` |
| Test metrics | `test_evaluation.json` |
| Confusion matrix | `test_confusion_matrix.csv` |
| HP sweep | `hp_sweep.csv` |
| Ablation | `ablation.csv` |
| Stability runs | `stability_runs.csv` / `stability_summary.csv` |
| Bootstrap CIs | `bootstrap_nmi.json` / `bootstrap_purity.json` |
| Visualizations | `topics_2d.html`, `barchart.html`, `heatmap.html`, `bootstrap_distributions.png` |

---

## 1. Theory recap (4-stage pipeline)

BERTopic (Grootendorst, 2022) decomposes topic modeling into four interchangeable stages:

1. **Embed** — convert each document into a dense vector with a transformer encoder.
2. **Reduce** — project the embeddings into a low-dimensional space with **UMAP**, preserving local neighborhoods so density-based clustering is meaningful.
3. **Cluster** — run **HDBSCAN** on the reduced space; documents that don't fit any density cluster are flagged as outliers (topic `-1`).
4. **Topic representation** — for each cluster, compute a **class-based TF-IDF (c-TF-IDF)** to extract the most distinctive words.

Unlike LDA, the number of topics is **discovered** from data (no `K` to set up front), and the representation step works on already-clustered groups instead of as part of the generative model.

## 2. Implementation choices

- **Encoder:** `readerbench/robert-base` — Romanian-specific BERT, 12 layers, hidden size 768.
  - Pooling: mean of last hidden state with attention-mask weighting.
  - `max_length = 128`, `batch_size = 32`. Embeddings cached to `embeddings_train.npy` / `embeddings_test.npy`.
- **UMAP:** `n_neighbors=15`, `n_components=5`, `min_dist=0.0`, `random_state=42`.
- **HDBSCAN:** `min_cluster_size=30`, `metric='euclidean'`, `prediction_data=True`.
- **BERTopic:** `language='multilingual'`, `calculate_probabilities=True`.
- **Train/test split:** Mihai's stratified 80/20 (random_state 42) on the `topic` column.

Code: `src/bertopic/{embeddings,model,training}.py`. CLI: `scripts/encode_docs.py`, `scripts/fit_bertopic.py`.

## 3. Discovered topics

After fitting on the training split, BERTopic discovered **_TBD_ topics** (excluding the outlier cluster, which contained **_TBD_ %** of the documents).

### Topic table (top-10 c-TF-IDF keywords + manual labels)

> Replace this once `topic_keywords_labeled.csv` is generated.

| Topic id | Count | Top-10 keywords | Manual label |
|---|---|---|---|
| 0 | _TBD_ | _TBD_ | _TBD_ |
| 1 | _TBD_ | _TBD_ | _TBD_ |
| ... | | | |

### Visualizations

- **`topics_2d.html`** — 2-D projection of cluster centroids (UMAP-of-UMAP).
- **`barchart.html`** — top keywords per topic.
- **`heatmap.html`** — pairwise topic similarity.

(Embed screenshots from these HTMLs into the final PDF report.)

## 4. Hyperparameter sensitivity (T4)

Sweep over the cartesian product of:

- `min_cluster_size ∈ {10, 20, 30, 50}`
- `n_neighbors ∈ {5, 15, 30}`

For each config we record `n_topics`, `outlier_pct`, and `c_v` coherence. Selection criterion: **maximize `c_v × (1 − outlier_pct)`** so we don't pick a config that achieves "high coherence" by sending most docs to the outlier topic.

> Replace with the table from `hp_sweep.csv` once available.

| min_cluster_size | n_neighbors | n_topics | outlier_pct | c_v | score |
|---|---|---|---|---|---|

**Chosen config:** _TBD_

## 5. Embedding ablation (T5)

We re-ran the same BERTopic config with `paraphrase-multilingual-mpnet-base-v2` (a general multilingual sentence encoder) instead of RoBERT.

> Replace with the table from `ablation.csv`.

| Embedding | n_topics | outlier_pct | c_v | NMI vs MOROCO | Purity |
|---|---|---|---|---|---|
| `readerbench/robert-base` | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| `mpnet-multilingual`     | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |

**Discussion (1–2 paragraphs):** which embedding wins on which metric and why we'd expect that for Romanian news (Romanian-specific pre-training vs. multilingual breadth).

## 6. Test-set evaluation (T6)

Inference on the held-out test split → BERTopic's `transform()` returns a topic id per document. Metrics computed against the MOROCO `topic` ground-truth:

| Metric | Value |
|---|---|
| # test docs | _TBD_ |
| Outlier proportion | _TBD_ |
| **NMI** | _TBD_ |
| **Purity** | _TBD_ |

### Confusion matrix

Rows: BERTopic cluster id. Columns: MOROCO category. Source: `test_confusion_matrix.csv`.

(Insert pretty version in the final PDF.)

## 6.5 Stability analysis

A single train/fit/test number doesn't say whether the result is reproducible. We address that from two angles.

### Multi-seed runs

We re-fit BERTopic **N = 10** times, varying only the UMAP `random_state`. Source: `stability_runs.csv` / `stability_summary.csv` (filled in by `scripts/stability_analysis.py`).

| Metric | mean | std | min | max |
|---|---|---|---|---|
| `n_topics` | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| `outlier_pct_train` | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| `c_v_train` | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| `outlier_pct_test` | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| `nmi_test` | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| `purity_test` | _TBD_ | _TBD_ | _TBD_ | _TBD_ |

> _Discussion (1 paragraph):_ how variable are the metrics across seeds, and what does that imply for our headline numbers in §6?

### Bootstrap confidence intervals on test metrics

For the chosen ("main") model, we resample the test predictions with replacement **B = 1000** times and report a **95 % percentile CI** on each metric. Source: `bootstrap_nmi.json` / `bootstrap_purity.json`.

| Metric | Test point estimate | Bootstrap mean | 95 % CI |
|---|---|---|---|
| NMI | _TBD_ | _TBD_ | _[TBD, TBD]_ |
| Purity | _TBD_ | _TBD_ | _[TBD, TBD]_ |

> _Reading:_ a tight CI means the test-set NMI/Purity number is statistically reliable; a wide CI means the held-out test set may be too small or too imbalanced to support strong claims.

(See `bootstrap_distributions.png` for the bootstrap histograms with the CIs marked.)

## 7. Strengths and weaknesses observed

- **Strengths**
  - No `K` to set up front — model discovers topic count from density.
  - Embeddings capture semantic similarity beyond word overlap (synonyms, paraphrases).
  - Outlier topic (`-1`) is honest about noise instead of forcing a label.

- **Weaknesses**
  - Results are sensitive to the UMAP / HDBSCAN parameters (see §4).
  - Outlier proportion can be high on short / atypical docs.
  - Less interpretable than LDA's word distributions because topics are clusters, not multinomials.

## 8. Reproducibility notes

- `random_state=42` everywhere a seed is exposed (UMAP, train/test split, BERTopic).
- HDBSCAN is deterministic given a fixed UMAP seed but can produce different clusterings on different sklearn / numba versions; pinned via `requirements.txt`.
- Embeddings are cached → fitting BERTopic again is `O(seconds)` on the same machine.
