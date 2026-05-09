"""One-shot helper to (re)generate `notebooks/03_bertopic_full.ipynb`.

This is checked in so the notebook can always be rebuilt from a single source
of truth (helpful for diff-friendly review). Run with:

    python scripts/_build_notebook.py
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "03_bertopic_full.ipynb"


def md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(text)


def code(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(text)


CELLS = [
    md(
        "# BERTopic — Full pipeline on MOROCO\n"
        "\n"
        "**Method B (BERTopic).** Companion to `02_lda_full.ipynb` (Method A — LDA).\n"
        "\n"
        "Walks through Steps **T1 → T7** of the project plan:\n"
        "\n"
        "| Step | Description |\n"
        "|---|---|\n"
        "| T1 | Sentence embeddings with `readerbench/robert-base` |\n"
        "| T2 | Configure and fit BERTopic (UMAP + HDBSCAN) |\n"
        "| T3 | Inspect topics + save visualizations |\n"
        "| T4 | Hyperparameter sensitivity sweep |\n"
        "| T5 | Embedding ablation (RoBERT vs. multilingual MPNet) |\n"
        "| T6 | Test-set evaluation: NMI, Purity, confusion matrix |\n"
        "| T7 | Document Method B (manual labels + report) |\n"
        "\n"
        "> **Data dependency:** expects `data/processed/{train,test}.parquet` "
        "(Mihai's Phase 1 hand-off). The data cell below raises a clear "
        "error if the files aren't there yet."
    ),
    md("## Setup"),
    code(
        "from __future__ import annotations\n"
        "\n"
        "import sys\n"
        "from pathlib import Path\n"
        "\n"
        "ROOT = Path.cwd().resolve()\n"
        "if (ROOT / 'src').exists() is False:\n"
        "    ROOT = ROOT.parent\n"
        "sys.path.insert(0, str(ROOT))\n"
        "\n"
        "import numpy as np\n"
        "import pandas as pd\n"
        "\n"
        "from src import paths\n"
        "from src.bertopic import (\n"
        "    BERTopicConfig,\n"
        "    RomanianEmbedder,\n"
        "    encode_documents,\n"
        "    fit_bertopic,\n"
        "    save_model, load_model,\n"
        "    topic_keyword_table, save_topic_table, attach_manual_labels,\n"
        "    n_topics, outlier_proportion,\n"
        "    save_default_visualizations,\n"
        "    compute_cv_coherence,\n"
        "    sweep, save_sweep,\n"
        "    run_ablation, save_ablation,\n"
        "    evaluate_on_test, save_evaluation,\n"
        ")\n"
        "from src.evaluation.metrics import build_confusion_matrix\n"
        "\n"
        "paths.ensure_dirs()\n"
        "TEXT_COL = 'text'\n"
        "LABEL_COL = 'topic'"
    ),
    md(
        "## Load Mihai's train / test split\n"
        "\n"
        "If this cell fails, Mihai's Phase 1 hand-off isn't ready yet. The "
        "rest of the notebook depends on it."
    ),
    code(
        "if not paths.TRAIN_PARQUET.exists() or not paths.TEST_PARQUET.exists():\n"
        "    raise FileNotFoundError(\n"
        "        f'Missing {paths.TRAIN_PARQUET} or {paths.TEST_PARQUET}. '\n"
        "        \"Wait for Mihai to finish Phase 1 (preprocessing + train_test_split).\"\n"
        "    )\n"
        "\n"
        "train_df = pd.read_parquet(paths.TRAIN_PARQUET)\n"
        "test_df = pd.read_parquet(paths.TEST_PARQUET)\n"
        "print(f'train: {len(train_df):,} rows  |  test: {len(test_df):,} rows')\n"
        "print('columns:', list(train_df.columns))\n"
        "train_df.head()"
    ),
    md(
        "## T1 — Generate sentence embeddings\n"
        "\n"
        "We use `readerbench/robert-base` with mean-pooling over the last "
        "hidden state (masked by attention). Embeddings are cached to disk so "
        "we don't recompute them every time we re-run BERTopic."
    ),
    code(
        "if paths.EMBEDDINGS_TRAIN.exists():\n"
        "    embeddings_train = np.load(paths.EMBEDDINGS_TRAIN)\n"
        "    print(f'loaded cached train embeddings: {embeddings_train.shape}')\n"
        "else:\n"
        "    embeddings_train = encode_documents(\n"
        "        docs=train_df[TEXT_COL].tolist(),\n"
        "        out_path=paths.EMBEDDINGS_TRAIN,\n"
        "        batch_size=32,\n"
        "    )\n"
        "\n"
        "if paths.EMBEDDINGS_TEST.exists():\n"
        "    embeddings_test = np.load(paths.EMBEDDINGS_TEST)\n"
        "    print(f'loaded cached test embeddings: {embeddings_test.shape}')\n"
        "else:\n"
        "    embeddings_test = encode_documents(\n"
        "        docs=test_df[TEXT_COL].tolist(),\n"
        "        out_path=paths.EMBEDDINGS_TEST,\n"
        "        batch_size=32,\n"
        "    )\n"
        "\n"
        "assert embeddings_train.shape[1] == 768, 'expected RoBERT 768-d output'"
    ),
    md(
        "## T2 — Configure and fit BERTopic\n"
        "\n"
        "Default config matches the spec: UMAP `n_neighbors=15, n_components=5, "
        "min_dist=0`, HDBSCAN `min_cluster_size=30`, multilingual c-TF-IDF."
    ),
    code(
        "cfg = BERTopicConfig(min_cluster_size=30, n_neighbors=15, n_components=5)\n"
        "docs_train = train_df[TEXT_COL].tolist()\n"
        "topic_model, topics_train, probs_train = fit_bertopic(docs_train, embeddings_train, cfg)\n"
        "\n"
        "save_model(topic_model, paths.MODEL_DIR_MAIN)\n"
        "print(f'saved model -> {paths.MODEL_DIR_MAIN}')\n"
        "print(f'discovered {n_topics(topic_model, exclude_outlier=True)} topics '\n"
        "      f'(+ outlier), outlier proportion = {outlier_proportion(topics_train):.2%}')"
    ),
    md(
        "## T3 — Inspect topics + visualize\n"
        "\n"
        "Top-10 keywords per topic and the three required interactive HTMLs."
    ),
    code(
        "table = topic_keyword_table(topic_model, top_k=10)\n"
        "table"
    ),
    code(
        "save_topic_table(topic_model, paths.TOPIC_TABLE_CSV, top_k=10)\n"
        "saved = save_default_visualizations(topic_model, paths.RESULTS_BERTOPIC)\n"
        "list(saved.keys())"
    ),
    md(
        "### Manual labels (T7)\n"
        "\n"
        "Fill in `MANUAL_LABELS` below after you've inspected the keyword "
        "table above. Each topic should map to one of the MOROCO categories "
        "(culture, finance, politics, science, sports, tech) or a free-form "
        "label like 'sub-topic of politics'."
    ),
    code(
        "MANUAL_LABELS: dict[int, str] = {\n"
        "    # 0: 'politics',\n"
        "    # 1: 'sports',\n"
        "    # ...\n"
        "}\n"
        "labeled = attach_manual_labels(table, MANUAL_LABELS)\n"
        "labeled.to_csv(paths.RESULTS_BERTOPIC / 'topic_keywords_labeled.csv', index=False)\n"
        "labeled"
    ),
    md(
        "## T4 — Hyperparameter sensitivity\n"
        "\n"
        "Sweep `min_cluster_size ∈ {10, 20, 30, 50}` x `n_neighbors ∈ {5, 15, 30}`. "
        "This trains 12 BERTopic models — slow on CPU. If it's too slow locally, "
        "run `python scripts/hp_sweep.py` on Colab/Kaggle with a GPU."
    ),
    code(
        "sweep_df = sweep(\n"
        "    docs=docs_train,\n"
        "    embeddings=embeddings_train,\n"
        "    min_cluster_sizes=(10, 20, 30, 50),\n"
        "    n_neighbors_list=(5, 15, 30),\n"
        ")\n"
        "save_sweep(sweep_df, paths.HP_SWEEP_CSV)\n"
        "sweep_df"
    ),
    md(
        "## T5 — Embedding ablation (RoBERT vs. multilingual MPNet)"
    ),
    code(
        "from src.bertopic.ablation import encode_with_sbert, MPNET_NAME\n"
        "\n"
        "if paths.EMBEDDINGS_TRAIN_MPNET.exists():\n"
        "    e_mpnet_train = np.load(paths.EMBEDDINGS_TRAIN_MPNET)\n"
        "else:\n"
        "    e_mpnet_train = encode_with_sbert(docs_train, MPNET_NAME)\n"
        "    np.save(paths.EMBEDDINGS_TRAIN_MPNET, e_mpnet_train)\n"
        "\n"
        "if paths.EMBEDDINGS_TEST_MPNET.exists():\n"
        "    e_mpnet_test = np.load(paths.EMBEDDINGS_TEST_MPNET)\n"
        "else:\n"
        "    e_mpnet_test = encode_with_sbert(test_df[TEXT_COL].tolist(), MPNET_NAME)\n"
        "    np.save(paths.EMBEDDINGS_TEST_MPNET, e_mpnet_test)\n"
        "\n"
        "ablation_df = run_ablation(\n"
        "    docs_train=docs_train,\n"
        "    docs_test=test_df[TEXT_COL].tolist(),\n"
        "    y_test=test_df[LABEL_COL].tolist(),\n"
        "    embeddings_robert_train=embeddings_train,\n"
        "    embeddings_robert_test=embeddings_test,\n"
        "    embeddings_mpnet_train=e_mpnet_train,\n"
        "    embeddings_mpnet_test=e_mpnet_test,\n"
        "    cfg=cfg,\n"
        ")\n"
        "save_ablation(ablation_df, paths.ABLATION_CSV)\n"
        "ablation_df"
    ),
    md(
        "## T6 — Test-set evaluation\n"
        "\n"
        "Inference on the held-out test split, then NMI / Purity / confusion matrix."
    ),
    code(
        "metrics = evaluate_on_test(\n"
        "    topic_model=topic_model,\n"
        "    docs_test=test_df[TEXT_COL].tolist(),\n"
        "    embeddings_test=embeddings_test,\n"
        "    y_test=test_df[LABEL_COL].tolist(),\n"
        ")\n"
        "save_evaluation(\n"
        "    metrics,\n"
        "    test_df[LABEL_COL].tolist(),\n"
        "    out_json=paths.TEST_EVAL_JSON,\n"
        "    out_confusion_csv=paths.TEST_CONFUSION_CSV,\n"
        ")\n"
        "{k: v for k, v in metrics.items() if k != 'topics_pred'}"
    ),
    code(
        "cm = build_confusion_matrix(\n"
        "    test_df[LABEL_COL].tolist(),\n"
        "    metrics['topics_pred'],\n"
        ")\n"
        "cm"
    ),
    md(
        "## T7 — Method B summary\n"
        "\n"
        "Use `docs/method_b_bertopic_report.md` as the writing template. "
        "Numbers from this notebook (saved into `results/bertopic/`) feed "
        "directly into Section 6.2 of the final report."
    ),
]


def main() -> None:
    nb = nbf.v4.new_notebook()
    nb["cells"] = CELLS
    nb["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3 (.venv)",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.11"},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"wrote {OUT} ({len(CELLS)} cells)")


if __name__ == "__main__":
    main()
