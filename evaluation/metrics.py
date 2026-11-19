"""Métricas binarias del protocolo de D0 §3.7.

* Promedio **macro**, como `precision/recall/f1_score(average='macro')` de P1/P2/P3. Coherente con
  D0 Tabla 5: un nodo que predice una sola clase obtiene recall 0,5 y precisión = accuracy/2.
* Umbral: el que maximiza el F1 macro en validación; después se fija para el test.
* Resumen: media de las últimas 5 rondas.
"""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

AVERAGE = "macro"


@torch.no_grad()
def predict_proba(model: nn.Module, X: torch.Tensor) -> np.ndarray:
    model.eval()
    return torch.sigmoid(model(X)).cpu().numpy().ravel()


def binary_metrics(y_true: np.ndarray, proba: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    y_true = np.asarray(y_true).ravel().astype(int)
    y_pred = (np.asarray(proba).ravel() > threshold).astype(int)
    kw = dict(average=AVERAGE, zero_division=0, labels=[0, 1])
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, **kw)),
        "recall": float(recall_score(y_true, y_pred, **kw)),
        "f1": float(f1_score(y_true, y_pred, **kw)),
        "threshold": float(threshold),
    }


def macro_f1_for_thresholds(y_true: np.ndarray, proba: np.ndarray,
                            thresholds: np.ndarray) -> np.ndarray:
    """F1 macro para muchos umbrales a la vez (predicción positiva si proba > t), en O(n log n).

    Equivale a `f1_score(average='macro', zero_division=0, labels=[0, 1])` (verificado en tests).
    """
    y = np.asarray(y_true).ravel().astype(int)
    p = np.asarray(proba).ravel()
    order = np.argsort(p, kind="mergesort")
    p_sorted, y_sorted = p[order], y[order]
    n, n_pos = y.size, int(y.sum())
    pos_suffix = np.concatenate([np.cumsum(y_sorted[::-1])[::-1], [0]])   # positivos en p_sorted[i:]
    idx = np.searchsorted(p_sorted, thresholds, side="right")             # predichos positivos: idx..n-1
    tp = pos_suffix[idx].astype(float)
    pred_pos = (n - idx).astype(float)
    fp = pred_pos - tp
    fn = n_pos - tp
    tn = (n - n_pos) - fp
    with np.errstate(divide="ignore", invalid="ignore"):
        f1_pos = np.where(2 * tp + fp + fn > 0, 2 * tp / (2 * tp + fp + fn), 0.0)
        f1_neg = np.where(2 * tn + fn + fp > 0, 2 * tn / (2 * tn + fn + fp), 0.0)
    return (f1_pos + f1_neg) / 2


def best_f1_threshold(y_true: np.ndarray, proba: np.ndarray) -> float:
    """Umbral que maximiza el F1 macro en validación (D0 §3.7); en empate, el menor.

    Candidatos: 0,5 (umbral heredado) y los puntos medios entre probabilidades distintas consecutivas.
    """
    p = np.unique(np.asarray(proba).ravel())
    mids = (p[:-1] + p[1:]) / 2 if p.size > 1 else np.empty(0)
    candidates = np.unique(np.concatenate([[0.5], mids]))
    scores = macro_f1_for_thresholds(y_true, proba, candidates)
    return float(candidates[int(np.argmax(scores))])


def mean_last_k(history: Sequence[Mapping[str, float]], k: int = 5) -> dict[str, float]:
    """Media de las últimas `k` rondas (D0 §3.7)."""
    if not history:
        raise ValueError("Historial vacío")
    tail = history[-k:]
    keys = [key for key in tail[0] if key != "threshold"]
    return {key: float(np.mean([h[key] for h in tail])) for key in keys}
