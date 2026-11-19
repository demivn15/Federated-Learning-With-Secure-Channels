from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import f1_score

from evaluation.metrics import binary_metrics, best_f1_threshold, macro_f1_for_thresholds, mean_last_k


def test_vectorized_macro_f1_matches_sklearn():
    rng = np.random.default_rng(0)
    for n_pos in (0, 1, 30, 500):
        y = np.zeros(1000, dtype=int)
        y[:n_pos] = 1
        p = np.clip(rng.random(1000) * 0.6 + 0.4 * y, 0, 1)
        thresholds = np.concatenate([[0.0, 0.5, 1.0], rng.random(40)])
        fast = macro_f1_for_thresholds(y, p, thresholds)
        slow = [f1_score(y, (p > t).astype(int), average="macro", zero_division=0, labels=[0, 1])
                for t in thresholds]
        np.testing.assert_allclose(fast, slow, atol=1e-12)


def test_single_class_prediction_reproduces_d0_table5_pattern():
    # D0 Tabla 5, CN1: Acc 0,8226 · Prec 0,4113 · Rec 0,5000 ⇒ el modelo predice solo la clase mayoritaria
    y = np.array([0] * 8226 + [1] * 1774)
    m = binary_metrics(y, np.zeros_like(y, dtype=float))
    assert m["accuracy"] == pytest.approx(0.8226)
    assert m["recall"] == pytest.approx(0.5)
    assert m["precision"] == pytest.approx(0.8226 / 2)


def test_best_threshold_not_worse_than_default():
    rng = np.random.default_rng(1)
    y = (rng.random(800) < 0.1).astype(int)
    p = np.clip(0.3 * y + rng.random(800) * 0.5, 0, 1)
    t = best_f1_threshold(y, p)
    assert binary_metrics(y, p, t)["f1"] >= binary_metrics(y, p, 0.5)["f1"]


def test_mean_last_k():
    hist = [{"f1": float(i), "recall": 1.0, "threshold": 0.5} for i in range(10)]
    assert mean_last_k(hist, 5) == {"f1": 7.0, "recall": 1.0}
    with pytest.raises(ValueError):
        mean_last_k([])
