"""Grading and uncertainty metrics (numpy): QWK, macro-F1, MCC, ECE (15 bins), Brier, NLL,
AURC and the AUROC of uncertainty for separating errors from correct predictions."""
from __future__ import annotations

import math

import numpy as np


def confusion(y, p, k):
    return np.bincount(np.asarray(y) * k + np.asarray(p), minlength=k * k).reshape(k, k)


def qwk(y, p, k):
    o = confusion(y, p, k).astype(float)
    i, j = np.meshgrid(np.arange(k), np.arange(k), indexing="ij")
    w = (i - j) ** 2 / (k - 1) ** 2
    e = np.outer(o.sum(1), o.sum(0)) / max(o.sum(), 1)
    den = (w * e).sum()
    return float(1 - (w * o).sum() / den) if den > 0 else 0.0


def macro_f1(y, p, k):
    c = confusion(y, p, k)
    present = [i for i in range(k) if c[i].sum() + c[:, i].sum() > 0]
    f1 = [2 * c[i, i] / (c[i].sum() + c[:, i].sum()) for i in present]
    return float(np.mean(f1)) if f1 else 0.0


def mcc(y, p, k):
    c = confusion(y, p, k).astype(float)
    t, q, s = c.sum(1), c.sum(0), c.sum()
    num = np.trace(c) * s - t @ q
    den = math.sqrt(max(s * s - q @ q, 0)) * math.sqrt(max(s * s - t @ t, 0))
    return float(num / den) if den > 0 else 0.0


def ece(y, probs, bins: int = 15):
    conf, pred = probs.max(1), probs.argmax(1)
    edges = np.linspace(0, 1, bins + 1)
    out = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = (conf > lo) & (conf <= hi)
        if s.any():
            out += s.mean() * abs((pred[s] == y[s]).mean() - conf[s].mean())
    return float(out)


def brier(y, probs):
    onehot = np.eye(probs.shape[1])[y]
    return float(((probs - onehot) ** 2).sum(1).mean())


def nll(y, probs):
    return float(-np.log(np.clip(probs[np.arange(len(y)), y], 1e-12, 1)).mean())


def aurc(y, probs, unc):
    order = np.argsort(unc, kind="stable")
    err = (probs.argmax(1) != y)[order].astype(float)
    return float((np.cumsum(err) / np.arange(1, len(y) + 1)).mean())


def auroc(scores, labels):
    """Mann-Whitney AUROC with average ranks for ties."""
    scores, labels = np.asarray(scores, float), np.asarray(labels, bool)
    n1, n0 = labels.sum(), (~labels).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores))
    s = scores[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[labels].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def uncertainty_metrics(y, probs, k):
    y, probs = np.asarray(y), np.asarray(probs, float)
    pred = probs.argmax(1)
    u_max = 1 - probs.max(1)
    u_ent = -(np.clip(probs, 1e-12, 1) * np.log(np.clip(probs, 1e-12, 1))).sum(1)
    err = pred != y
    return {"accuracy": float((pred == y).mean()), "qwk": qwk(y, pred, k), "ece": ece(y, probs),
            "brier": brier(y, probs), "nll": nll(y, probs),
            "aurc_1-maxp": aurc(y, probs, u_max), "aurc_entropy": aurc(y, probs, u_ent),
            "err_auroc_1-maxp": auroc(u_max, err), "err_auroc_entropy": auroc(u_ent, err)}
