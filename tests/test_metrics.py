"""Unit tests for `src.evaluation.metrics`."""

from __future__ import annotations

import math

import pytest

from src.evaluation.metrics import (
    build_confusion_matrix,
    nmi_score,
    purity_score,
)


class TestPurityScore:
    def test_perfect_clusters(self):
        # cluster 0 -> all label 'A', cluster 1 -> all label 'B' => purity = 1
        y_true = ["A", "A", "B", "B"]
        y_pred = [0, 0, 1, 1]
        assert purity_score(y_true, y_pred) == pytest.approx(1.0)

    def test_random_clusters(self):
        # everything in one cluster, but two equal labels => purity = 0.5
        y_true = ["A", "A", "B", "B"]
        y_pred = [0, 0, 0, 0]
        assert purity_score(y_true, y_pred) == pytest.approx(0.5)

    def test_known_majority(self):
        y_true = ["A", "A", "A", "B", "B", "C"]
        y_pred = [0, 0, 0, 0, 1, 1]
        # cluster 0 has 3xA + 1xB => 3 correct
        # cluster 1 has 1xB + 1xC => 1 correct (tie -> max gives 1)
        # purity = (3 + 1) / 6
        assert purity_score(y_true, y_pred) == pytest.approx(4 / 6)


class TestNmiScore:
    def test_perfect_alignment(self):
        y_true = ["A", "A", "B", "B"]
        y_pred = [0, 0, 1, 1]
        assert nmi_score(y_true, y_pred) == pytest.approx(1.0)

    def test_independent_assignment(self):
        # all same predicted cluster, varied truth => NMI = 0
        y_true = ["A", "B", "A", "B"]
        y_pred = [0, 0, 0, 0]
        assert nmi_score(y_true, y_pred) == pytest.approx(0.0, abs=1e-9)

    def test_in_unit_interval(self):
        y_true = ["A", "A", "B", "B", "C", "C"]
        y_pred = [0, 1, 0, 1, 2, 2]
        s = nmi_score(y_true, y_pred)
        assert 0.0 <= s <= 1.0
        assert not math.isnan(s)


class TestBuildConfusionMatrix:
    def test_shape(self):
        y_true = ["A", "A", "B"]
        y_pred = [0, 1, 1]
        cm = build_confusion_matrix(y_true, y_pred)
        assert cm.shape == (2, 2)
        assert set(cm.index) == {0, 1}
        assert set(cm.columns) == {"A", "B"}

    def test_drop_outliers(self):
        y_true = ["A", "A", "B"]
        y_pred = [0, -1, 1]
        cm = build_confusion_matrix(y_true, y_pred, drop_outliers=True)
        assert -1 not in cm.index

    def test_keep_outliers_by_default(self):
        y_true = ["A", "A", "B"]
        y_pred = [0, -1, 1]
        cm = build_confusion_matrix(y_true, y_pred)
        assert -1 in cm.index
