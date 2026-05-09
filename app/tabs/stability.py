"""Stability tab — multi-seed runs + bootstrap CIs + HP sweep heatmap."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src import paths

from .. import utils


def _render_multi_seed() -> None:
    runs = utils.load_csv_or_none(paths.STABILITY_RUNS_CSV)
    summary = utils.load_csv_or_none(paths.STABILITY_SUMMARY_CSV)

    if runs is None or summary is None:
        utils.missing_artifact(
            paths.STABILITY_SUMMARY_CSV, "python scripts/stability_analysis.py"
        )
        return

    st.markdown("**Per-run metrics** (one row per seed)")
    st.dataframe(runs, hide_index=True, use_container_width=True)

    st.markdown("**Summary across seeds**")
    st.dataframe(summary, hide_index=True, use_container_width=True)

    plottable = ["c_v_train", "nmi_test", "purity_test", "outlier_pct_test"]
    available = [c for c in plottable if c in runs.columns and runs[c].notna().any()]
    if available:
        st.markdown("**Spread across seeds**")
        for col in available:
            st.markdown(f"- `{col}`")
            st.bar_chart(runs.set_index("seed")[col], use_container_width=True)


def _render_bootstrap(metric: str, json_path) -> None:
    boot = utils.load_json_or_none(json_path)
    if boot is None:
        utils.missing_artifact(
            json_path, "python scripts/stability_analysis.py"
        )
        return

    cols = st.columns(4)
    cols[0].metric(f"{metric} mean", f"{boot['mean']:.4f}")
    cols[1].metric(f"{metric} std", f"{boot['std']:.4f}")
    cols[2].metric(
        f"{int(boot['ci'] * 100)}% CI low", f"{boot['ci_low']:.4f}"
    )
    cols[3].metric(
        f"{int(boot['ci'] * 100)}% CI high", f"{boot['ci_high']:.4f}"
    )

    samples = boot.get("samples", [])
    if samples:
        df = pd.DataFrame({metric: samples})
        st.markdown(f"**Bootstrap distribution** (B = {boot['n_bootstrap']:,})")
        st.bar_chart(
            df[metric].value_counts(bins=40).sort_index(), use_container_width=True
        )


def _render_hp_sweep() -> None:
    sweep = utils.load_csv_or_none(paths.HP_SWEEP_CSV)
    if sweep is None:
        utils.missing_artifact(paths.HP_SWEEP_CSV, "python scripts/hp_sweep.py")
        return

    st.dataframe(sweep, hide_index=True, use_container_width=True)
    st.caption(
        "Selection criterion: maximize `c_v × (1 − outlier_pct)` — "
        "discourages configs that achieve high coherence by sending most "
        "documents to the outlier topic."
    )


def render() -> None:
    utils.section_header(
        "Stability and confidence intervals",
        "Two complementary techniques for assessing how reliable the BERTopic "
        "headline numbers are.",
    )

    inner = st.tabs(
        [
            "Multi-seed (model variance)",
            "Bootstrap CIs (sample variance)",
            "Hyperparameter sweep",
        ]
    )

    with inner[0]:
        st.markdown(
            "Re-fit BERTopic N times varying only the UMAP `random_state`. "
            "Variance comes from the model's stochasticity."
        )
        _render_multi_seed()

    with inner[1]:
        st.markdown(
            "Resample the test predictions with replacement B = 1000 times "
            "and report a 95% percentile CI for each metric. Variance comes "
            "from the test sample, not the model."
        )
        st.subheader("NMI")
        _render_bootstrap("NMI", paths.BOOTSTRAP_NMI_JSON)
        st.subheader("Purity")
        _render_bootstrap("Purity", paths.BOOTSTRAP_PURITY_JSON)

    with inner[2]:
        st.markdown(
            "Cartesian product of `min_cluster_size ∈ {10, 20, 30, 50}` × "
            "`n_neighbors ∈ {5, 15, 30}`."
        )
        _render_hp_sweep()
