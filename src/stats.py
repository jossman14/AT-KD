"""Evaluation statistics: ECE, paired McNemar test, bootstrap F1 CI."""
from __future__ import annotations
import numpy as np
from scipy.stats import chi2
from sklearn.metrics import f1_score


def ece_score(probs: np.ndarray, y: np.ndarray, n_bins: int = 15) -> float:
    """Expected Calibration Error with equal-width bins on max-probability."""
    conf = probs.max(1); pred = probs.argmax(1); correct = (pred == y).astype(float)
    edges = np.linspace(0, 1, n_bins + 1); ece = 0.0
    for i in range(n_bins):
        mask = (conf > edges[i]) & (conf <= edges[i + 1])
        if mask.sum():
            ece += mask.mean() * abs(correct[mask].mean() - conf[mask].mean())
    return float(ece)


def mcnemar_p(y: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> float:
    """Two-sided McNemar with continuity correction on paired predictions."""
    correct_a = (pred_a == y); correct_b = (pred_b == y)
    n01 = int((correct_a & ~correct_b).sum()); n10 = int((~correct_a & correct_b).sum())
    if n01 + n10 == 0:
        return 1.0
    chi = (abs(n01 - n10) - 1) ** 2 / (n01 + n10)
    return float(1 - chi2.cdf(chi, 1))


def bootstrap_f1_ci(y: np.ndarray, pred: np.ndarray, n: int = 2000,
                    alpha: float = 0.05, seed: int = 0) -> tuple:
    rng = np.random.default_rng(seed); m = len(y); arr = np.empty(n)
    for i in range(n):
        idx = rng.integers(0, m, m)
        arr[i] = f1_score(y[idx], pred[idx], average="macro")
    lo, hi = np.percentile(arr, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)
