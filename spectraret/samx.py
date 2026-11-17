"""Fig. 3, SAM-X box - spatial-spectral attribution, plus band-space Integrated Gradients
and band-deletion faithfulness (Eq. samx, Eq. bandig, Eq. del).

SAM-X is computed after training; it is not part of the forward path. Every method acts on
the band tensor between LSD and the tokenizer (model.forward_from_bands).
"""
from __future__ import annotations

import itertools

import torch
import torch.nn.functional as F


def samx(model, x, form: str = "final"):
    """form="dev":   A^d = w_b^d sum_c |f_b| ds_d/df_b,  s_d = sum of sigmoids of the logits;
                     band profile from the positive part, normalised to sum 1.
    form="final":    A^d = sum_c f_b d log p_d(c_hat | x)/df_b, signed; c_hat = predicted class.
    Returns joint (b, D, B, H, W), band profile (b, D, B), spatial map (b, D, H, W), predictions.
    """
    model.eval()
    with torch.no_grad():
        x_seg, _ = model.segment(x)
        bands = model.bands(x_seg).contiguous()
    bands = bands.detach().requires_grad_(True)
    logits, _, w = model.forward_from_bands(bands)
    joint, profile, preds = [], [], []
    for d, lg in enumerate(logits):
        c_hat = lg.argmax(1)
        preds.append(c_hat)
        if form == "final":
            s = F.log_softmax(lg, 1).gather(1, c_hat[:, None]).sum()
        elif form == "dev":
            s = torch.sigmoid(lg).sum()
        else:
            raise ValueError("form must be final or dev")
        g, = torch.autograd.grad(s, bands, retain_graph=d < len(logits) - 1)
        if form == "final":
            a = (bands * g).sum(2)
            prof = a.sum(dim=(-1, -2))
        else:
            a = w[:, d, :, None, None].detach() * (bands.abs() * g).sum(2)
            prof = a.clamp_min(0).sum(dim=(-1, -2))
            prof = prof / prof.sum(1, keepdim=True).clamp_min(1e-12)
        joint.append(a.detach())
        profile.append(prof.detach())
    joint = torch.stack(joint, 1)
    return {"joint": joint, "profile": torch.stack(profile, 1), "spatial": joint.sum(2),
            "pred": torch.stack(preds, 1), "routing": w.detach()}


def band_integrated_gradients(model, x, d: int, steps: int = 32):
    """IG on the band tensor from the black-image bands; target log p_d(c_hat)."""
    model.eval()
    with torch.no_grad():
        x_seg, _ = model.segment(x)
        bands = model.bands(x_seg).contiguous()
        base = model.bands(torch.full_like(x_seg, -1.0)).contiguous()
        c_hat = model.forward_from_bands(bands)[0][d].argmax(1)
    total = torch.zeros_like(bands)
    for k in range(1, steps + 1):
        fk = (base + k / steps * (bands - base)).requires_grad_(True)
        q = F.log_softmax(model.forward_from_bands(fk)[0][d], 1).gather(1, c_hat[:, None]).sum()
        total += torch.autograd.grad(q, fk)[0]
    a = ((bands - base) * total / steps).sum(2)          # (b, B, H, W)
    return {"joint": a, "profile": a.sum(dim=(-1, -2)), "pred": c_hat}


@torch.no_grad()
def band_subset_scores(model, x, d: int, replacement=None):
    """p_d(c_hat | x) for every subset of deleted bands (all 2^B subsets).
    replacement (B, C, H, W): None = zeroing, or the training-mean bands."""
    model.eval()
    x_seg, _ = model.segment(x)
    bands = model.bands(x_seg).contiguous()
    n_b = bands.shape[1]
    rep = torch.zeros_like(bands[0]) if replacement is None else replacement.to(bands)
    c_hat = model.forward_from_bands(bands)[0][d].argmax(1)
    scores = {}
    for r in range(n_b + 1):
        for subset in itertools.combinations(range(n_b), r):
            fb = bands.clone()
            for b in subset:
                fb[:, b] = rep[b]
            p = F.softmax(model.forward_from_bands(fb)[0][d], 1)
            scores[frozenset(subset)] = p.gather(1, c_hat[:, None]).squeeze(1).cpu()
    return scores


def deletion_ratios(scores: dict, ranking: torch.Tensor):
    """Per-image ratios s(x^(-k)) / s(x), k = 0..B, deleting bands in `ranking` order
    (b, B; most important first). Returns (b, B + 1)."""
    base = scores[frozenset()].clamp_min(1e-12)
    n, n_b = ranking.shape
    cols = [torch.ones(n)]
    for k in range(1, n_b + 1):
        cols.append(torch.stack([scores[frozenset(ranking[i, :k].tolist())][i] for i in range(n)]) / base)
    return torch.stack(cols, 1)


def random_deletion_ratios(scores: dict, n: int, n_b: int):
    """Exact random reference: average of the ratios over all B! deletion orders."""
    perms = list(itertools.permutations(range(n_b)))
    return sum(deletion_ratios(scores, torch.tensor([p] * n)) for p in perms) / len(perms)


def deletion_auc(ratios: torch.Tensor):
    """r_k = mean_i ratio_i(k), AUC = (1/B) sum_k (r_{k-1} + r_k) / 2. Lower = more faithful."""
    r = ratios.mean(0)
    return float(((r[:-1] + r[1:]) / 2).mean()), r.tolist()
