"""Anomaly scoring and evaluation metrics (frame-level AUC-ROC, EER, PR-AUC,
bootstrap confidence intervals, and effect-size statistics)."""
from __future__ import annotations

import numpy as np
from scipy import stats
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


def smooth(scores: np.ndarray, sigma: float = 3.0) -> np.ndarray:
    return gaussian_filter1d(scores.astype(np.float64), sigma) if sigma > 0 else scores


def minmax(scores: np.ndarray) -> np.ndarray:
    lo, hi = scores.min(), scores.max()
    return (scores - lo) / (hi - lo + 1e-12)


def eer(y: np.ndarray, s: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(y, s)
    fnr = 1 - tpr
    i = np.nanargmin(np.abs(fnr - fpr))
    return float((fpr[i] + fnr[i]) / 2)


def frame_metrics(y: np.ndarray, s: np.ndarray) -> dict:
    return {"auc": float(roc_auc_score(y, s)), "eer": eer(y, s), "pr_auc": float(average_precision_score(y, s))}


def bootstrap_auc(per_video_y: list, per_video_s: list, n: int = 1000, seed: int = 0) -> tuple[float, float]:
    """95% CI of frame-level AUC by resampling whole test videos (frames within a
    video are correlated, so resampling frames would understate uncertainty)."""
    rng = np.random.default_rng(seed)
    k = len(per_video_y)
    aucs = []
    for _ in range(n):
        pick = rng.integers(0, k, k)
        y = np.concatenate([per_video_y[i] for i in pick])
        s = np.concatenate([per_video_s[i] for i in pick])
        if 0 < y.sum() < len(y):
            aucs.append(roc_auc_score(y, s))
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def separation_stats(y: np.ndarray, s: np.ndarray) -> dict:
    """Mann-Whitney U (one-sided: anomalous > normal) and Cohen's d."""
    a, b = s[y == 1], s[y == 0]
    u, p = stats.mannwhitneyu(a, b, alternative="greater")
    pooled = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    return {"mannwhitney_u": float(u), "p_value": float(p), "cohens_d": float((a.mean() - b.mean()) / pooled)}


def threshold_metrics(y: np.ndarray, s: np.ndarray, tau: float) -> dict:
    pred = s > tau
    tp = int((pred & (y == 1)).sum()); fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum()); tn = int((~pred & (y == 0)).sum())
    prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
    return {"tau": float(tau), "precision": prec, "recall": rec,
            "f1": 2 * prec * rec / max(prec + rec, 1e-12), "fpr": fp / max(fp + tn, 1)}
