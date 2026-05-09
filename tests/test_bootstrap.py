"""Unit tests for `bootstrap_metric` in `src.evaluation.metrics`."""

from __future__ import annotations

import math

import pytest

from src.evaluation.metrics import bootstrap_metric, nmi_score, purity_score


class TestBootstrapMetric:
    def test_returns_expected_keys(self):
        y_true = ["A", "A", "B", "B"]
        y_pred = [0, 0, 1, 1]
        out = bootstrap_metric(
            y_true, y_pred, metric_fn=purity_score, n_bootstrap=50
        )
        assert set(out.keys()) == {
            "mean",
            "std",
            "ci_low",
            "ci_high",
            "n_bootstrap",
            "ci",
            "samples",
        }

    def test_perfect_alignment_gives_purity_one(self):
        y_true = ["A", "A", "B", "B"]
        y_pred = [0, 0, 1, 1]
        out = bootstrap_metric(
            y_true, y_pred, metric_fn=purity_score, n_bootstrap=200
        )
        assert out["mean"] == pytest.approx(1.0)
        assert out["ci_low"] == pytest.approx(1.0)
        assert out["ci_high"] == pytest.approx(1.0)
        assert math.isclose(out["std"], 0.0, abs_tol=1e-9)

    def test_ci_brackets_mean(self):
        # Use a moderately balanced configuration so resamples produce
        # variation but NMI rarely degenerates.
        y_true = ["A"] * 30 + ["B"] * 30 + ["C"] * 30
        y_pred = [0] * 30 + [1] * 30 + [2] * 30
        out = bootstrap_metric(
            y_true, y_pred, metric_fn=nmi_score, n_bootstrap=300, random_state=7
        )
        assert out["ci_low"] <= out["mean"] <= out["ci_high"]

    def test_reproducible_with_seed(self):
        y_true = ["A"] * 10 + ["B"] * 10
        y_pred = [0] * 9 + [1] + [1] * 9 + [0]  # mostly aligned
        a = bootstrap_metric(
            y_true, y_pred, metric_fn=purity_score, n_bootstrap=100, random_state=123
        )
        b = bootstrap_metric(
            y_true, y_pred, metric_fn=purity_score, n_bootstrap=100, random_state=123
        )
        assert a["mean"] == b["mean"]
        assert a["ci_low"] == b["ci_low"]
        assert a["ci_high"] == b["ci_high"]

    def test_different_ci_levels(self):
        y_true = ["A"] * 50 + ["B"] * 50
        y_pred = [0] * 45 + [1] * 5 + [1] * 45 + [0] * 5
        wide = bootstrap_metric(
            y_true, y_pred, metric_fn=purity_score, n_bootstrap=300, ci=0.99
        )
        tight = bootstrap_metric(
            y_true, y_pred, metric_fn=purity_score, n_bootstrap=300, ci=0.50
        )
        # 99% CI must be at least as wide as the 50% CI.
        assert (wide["ci_high"] - wide["ci_low"]) >= (
            tight["ci_high"] - tight["ci_low"]
        )

    def test_rejects_invalid_ci(self):
        with pytest.raises(ValueError):
            bootstrap_metric(["A"], [0], metric_fn=purity_score, ci=1.5)
        with pytest.raises(ValueError):
            bootstrap_metric(["A"], [0], metric_fn=purity_score, ci=0.0)

    def test_rejects_misaligned_inputs(self):
        with pytest.raises(ValueError):
            bootstrap_metric(
                ["A", "B"], [0], metric_fn=purity_score, n_bootstrap=10
            )
