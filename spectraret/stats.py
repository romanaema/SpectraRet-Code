"""Section 6.2-6.3 - statistics and the phase-gate rule.

  bootstrap_mean_ci     95% percentile-bootstrap interval of the mean over fold-seed runs
                        (5,000 resamples)
  paired_wilcoxon       paired differences over runs sharing fold and seed, two-sided
                        Wilcoxon signed-rank test
  holm                  Holm correction (six comparisons for the final model, nine for the
                        earlier model)
  friedman_nemenyi      average ranks, Friedman chi^2, Iman-Davenport F and the Nemenyi
                        critical difference (Demsar, 2006); with six models and three tasks
                        CD = 4.35
  image_bootstrap       95% image-bootstrap interval of a metric on pooled out-of-fold
                        predictions (and of a paired difference between two methods)
  gate_verdict          KEEP / MARGINAL / STOP of one arm on one disease
  component_decision    pass on >= 2 KEEP; otherwise the tie-break added after the results
                        were seen (drop if lower on >= 2 diseases, keep pragmatically if
                        higher on >= 2)
The runs are dependent (shared test images and training images), so intervals are optimistic
and p-values approximate; they are reported descriptively.
"""
from __future__ import annotations

import math

import numpy as np

# two-tailed Nemenyi q_alpha at alpha = 0.05 for k = 2..10 models (Demsar, 2006, Table 5a)
NEMENYI_Q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164}


def bootstrap_mean_ci(values, n_boot: int = 5000, alpha: float = 0.05, seed: int = 0):
    v = np.asarray(values, float)
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), (n_boot, len(v)))].mean(1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"mean": float(v.mean()), "lo": float(lo), "hi": float(hi), "n": len(v)}


def paired_wilcoxon(a, b):
    """a, b: metric per run, aligned on (fold, seed). Returns the paired differences a - b."""
    from scipy.stats import wilcoxon
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    res = {"mean_diff": float(d.mean()), "n_ahead": int((d > 0).sum()), "n": len(d)}
    res["p"] = float(wilcoxon(a, b, alternative="two-sided").pvalue) if np.any(d != 0) else 1.0
    return res


def holm(pvalues):
    """Holm step-down adjusted p-values, in the input order."""
    p = np.asarray(pvalues, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj.tolist()


def average_ranks(scores):
    """scores (n_tasks, k_models), higher is better -> ranks (1 = best), ties averaged."""
    from scipy.stats import rankdata
    s = np.asarray(scores, float)
    return np.vstack([rankdata(-row) for row in s])


def friedman_nemenyi(scores, alpha: float = 0.05):
    from scipy.stats import chi2, f
    r = average_ranks(scores)
    n, k = r.shape
    rbar = r.mean(0)
    chi = 12 * n / (k * (k + 1)) * (np.sum(rbar ** 2) - k * (k + 1) ** 2 / 4)
    p_chi = float(chi2.sf(chi, k - 1))
    ff = (n - 1) * chi / (n * (k - 1) - chi) if n * (k - 1) - chi > 0 else float("inf")
    p_ff = float(f.sf(ff, k - 1, (k - 1) * (n - 1))) if math.isfinite(ff) else 0.0
    if alpha != 0.05:
        raise ValueError("only the alpha = 0.05 Nemenyi table is included")
    cd = NEMENYI_Q05[k] * math.sqrt(k * (k + 1) / (6 * n))
    return {"average_ranks": rbar.tolist(), "chi2": float(chi), "df": k - 1, "p_friedman": p_chi,
            "iman_davenport_F": float(ff), "df_F": (k - 1, (k - 1) * (n - 1)), "p_iman_davenport": p_ff,
            "nemenyi_cd": cd, "max_rank_difference": float(rbar.max() - rbar.min())}


def image_bootstrap(metric, y, probs, n_boot: int = 2000, seed: int = 0, probs_ref=None, alpha: float = 0.05):
    """Interval of metric(y, probs) over images; with probs_ref, of metric(probs) - metric(probs_ref)."""
    y, probs = np.asarray(y), np.asarray(probs)
    rng = np.random.default_rng(seed)

    def stat(idx):
        v = metric(y[idx], probs[idx])
        return v if probs_ref is None else v - metric(y[idx], np.asarray(probs_ref)[idx])

    point = stat(np.arange(len(y)))
    boots = np.array([stat(rng.integers(0, len(y), len(y))) for _ in range(n_boot)])
    lo, hi = np.nanpercentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"value": float(point), "lo": float(lo), "hi": float(hi)}


def gate_verdict(arm_runs, baseline_mean: float, n_boot: int = 5000, seed: int = 0):
    """KEEP if the lower end of the arm's 95% interval clears the baseline mean, STOP if the
    upper end falls below it, MARGINAL otherwise."""
    ci = bootstrap_mean_ci(arm_runs, n_boot, seed=seed)
    verdict = "KEEP" if ci["lo"] > baseline_mean else "STOP" if ci["hi"] < baseline_mean else "MARGINAL"
    return {**ci, "baseline_mean": baseline_mean, "verdict": verdict}


def component_decision(verdicts: dict, arm_means: dict, baseline_means: dict):
    """verdicts / means keyed by disease. Rule fixed before the gates: pass when KEEP on at
    least two diseases. Tie-break (added after the results were seen): drop when KEEP on no
    disease and the mean is lower on >= 2 diseases; keep pragmatically when higher on >= 2."""
    n_keep = sum(v == "KEEP" for v in verdicts.values())
    higher = sum(arm_means[d] > baseline_means[d] for d in verdicts)
    lower = sum(arm_means[d] < baseline_means[d] for d in verdicts)
    if n_keep >= 2:
        return "pass"
    if n_keep == 0 and lower >= 2:
        return "drop"
    if higher >= 2:
        return "keep (pragmatic)"
    return "drop"


def kendall_tau(a, b):
    from scipy.stats import kendalltau
    return float(kendalltau(a, b).statistic)


def spearman(a, b):
    from scipy.stats import spearmanr
    return float(spearmanr(np.ravel(a), np.ravel(b)).statistic)
