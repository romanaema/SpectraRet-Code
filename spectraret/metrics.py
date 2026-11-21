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


def mce(y, probs, bins: int = 15):
    """Maximum calibration error over the same 15 equal-width bins as ECE."""
    conf, pred = probs.max(1), probs.argmax(1)
    edges = np.linspace(0, 1, bins + 1)
    gaps = [abs((pred[s] == y[s]).mean() - conf[s].mean())
            for lo, hi in zip(edges[:-1], edges[1:]) if (s := (conf > lo) & (conf <= hi)).any()]
    return float(max(gaps)) if gaps else 0.0


def reliability_bins(y, probs, bins: int = 5, min_count: int = 2):
    """Fig. 7a-b: (mean confidence, accuracy, count) of equal-width bins with >= min_count images."""
    y, probs = np.asarray(y), np.asarray(probs, float)
    conf, pred = probs.max(1), probs.argmax(1)
    edges = np.linspace(0, 1, bins + 1)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = (conf > lo) & (conf <= hi)
        if s.sum() >= min_count:
            out.append((float(conf[s].mean()), float((pred[s] == y[s]).mean()), int(s.sum())))
    return out


def risk_coverage(y, probs, unc):
    """Fig. 7c-d: error rate of the retained cases when the most uncertain are referred first."""
    order = np.argsort(unc, kind="stable")
    err = (np.asarray(probs).argmax(1) != np.asarray(y))[order].astype(float)
    n = len(err)
    return np.arange(1, n + 1) / n, np.cumsum(err) / np.arange(1, n + 1)


def referral(y, probs, unc, coverage: float = 0.8, focus_class=None):
    """Refer the (1 - coverage) most uncertain cases. Relative error reduction of the retained
    cases, and (optionally) the share of misclassified images of focus_class that were referred
    (e.g. Level 4 DR: 0.57 for the head, 0.20 expected under random referral)."""
    y, probs, unc = np.asarray(y), np.asarray(probs, float), np.asarray(unc, float)
    wrong = probs.argmax(1) != y
    n_keep = int(round(coverage * len(y)))
    keep = np.argsort(unc, kind="stable")[:n_keep]
    referred = np.ones(len(y), bool)
    referred[keep] = False
    e_all, e_keep = wrong.mean(), wrong[keep].mean()
    out = {"coverage": coverage, "error_all": float(e_all), "error_retained": float(e_keep),
           "relative_error_reduction": float((e_all - e_keep) / e_all) if e_all > 0 else 0.0}
    if focus_class is not None:
        sel = wrong & (y == focus_class)
        out["referred_share_of_errors_in_class"] = float(referred[sel].mean()) if sel.any() else float("nan")
    return out


def shift_auroc(unc_in, unc_shift):
    """AUROC of the uncertainty score for separating shifted from in-distribution images."""
    s = np.concatenate([np.asarray(unc_in, float), np.asarray(unc_shift, float)])
    lab = np.concatenate([np.zeros(len(unc_in), bool), np.ones(len(unc_shift), bool)])
    return auroc(s, lab)


def uncertainty_metrics(y, probs, k):
    y, probs = np.asarray(y), np.asarray(probs, float)
    pred = probs.argmax(1)
    u_max = 1 - probs.max(1)
    u_ent = -(np.clip(probs, 1e-12, 1) * np.log(np.clip(probs, 1e-12, 1))).sum(1)
    err = pred != y
    return {"accuracy": float((pred == y).mean()), "qwk": qwk(y, pred, k), "ece": ece(y, probs),
            "mce": mce(y, probs), "brier": brier(y, probs), "nll": nll(y, probs),
            "aurc_1-maxp": aurc(y, probs, u_max), "aurc_entropy": aurc(y, probs, u_ent),
            "err_auroc_1-maxp": auroc(u_max, err), "err_auroc_entropy": auroc(u_ent, err)}


def macro_auc(y, probs):
    """One-versus-rest macro-AUC over the classes present in y (Table 10)."""
    y, probs = np.asarray(y), np.asarray(probs, float)
    aucs = [auroc(probs[:, c], y == c) for c in range(probs.shape[1]) if 0 < (y == c).sum() < len(y)]
    return float(np.mean(aucs)) if aucs else float("nan")


def per_class_sensitivity(y, p, k):
    """Recall of each class (e.g. Level 3 DR 0.734 vs 0.498 for the control)."""
    c = confusion(y, p, k)
    return [float(c[i, i] / c[i].sum()) if c[i].sum() else float("nan") for i in range(k)]


def balanced_accuracy(y, p, k):
    s = [v for v in per_class_sensitivity(y, p, k) if not math.isnan(v)]
    return float(np.mean(s)) if s else 0.0


def grading_report(y, probs, k):
    """All grading metrics of the main comparison (Tables 9-10) for one disease."""
    y, probs = np.asarray(y), np.asarray(probs, float)
    p = probs.argmax(1)
    return {"n": len(y), "qwk": qwk(y, p, k), "accuracy": float((p == y).mean()),
            "macro_f1": macro_f1(y, p, k), "mcc": mcc(y, p, k), "macro_auc": macro_auc(y, probs),
            "ece": ece(y, probs), "brier": brier(y, probs), "balanced_accuracy": balanced_accuracy(y, p, k),
            "per_class_sensitivity": per_class_sensitivity(y, p, k)}
