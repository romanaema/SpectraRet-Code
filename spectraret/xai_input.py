"""Sections 6.5 and 7.5 / Tables 17-18 - input-space attribution, localization and sanity checks.

Development runs (Table 17): SAM-X was compared with gradient x input, Integrated Gradients
(32 steps, black baseline) and occlusion (24 x 24 patches), all explaining the summed-sigmoid
score. Spatial methods ranked bands by the overlap sum_{u,v} h(u,v) sum_c |f_b(u,v,c)| of
their heat map h with each band (which favours high-energy bands). Localization used an
oracle-area threshold (keep 1.5 x each mask's area) with IoU and the pointing game.

Final-model repeat (Table 18): every method explains log p(c_hat | x) of the class predicted for
the unperturbed image; thresholds are chosen on a 30% validation part of each mask set; central
and random maps and the Stage-1 mask used as a heat map are reference maps; model- and
label-randomization checks, seed-to-seed map stability (Spearman rho) and band-ranking stability
(Kendall tau) are added.

Images are in [-1, 1], so the black image is -1 (zero would be mid-grey).
"""
from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn.functional as F

from .stats import kendall_tau, spearman


# ----------------------------------------------------------------------------- targets
def target_score(model, x, d: int, target: str = "logprob", c_hat=None):
    """Scalar per image: "logprob" = log p_d(c_hat | x) (final protocol), "sigmoid" = sum of the
    sigmoids of the logits (development protocol, a score tied to no grade)."""
    lg = model(x).logits[d]
    if target == "sigmoid":
        return torch.sigmoid(lg).sum(1)
    if c_hat is None:
        c_hat = lg.argmax(1)
    return F.log_softmax(lg, 1).gather(1, c_hat[:, None]).squeeze(1)


@torch.no_grad()
def predicted_class(model, x, d: int):
    model.eval()
    return model(x).logits[d].argmax(1)


# ----------------------------------------------------------------------------- input-space maps
def grad_x_input(model, x, d: int, target: str = "logprob"):
    """(b, H, W) signed map sum_c x * ds/dx."""
    model.eval()
    c_hat = predicted_class(model, x, d) if target == "logprob" else None
    x = x.clone().requires_grad_(True)
    g, = torch.autograd.grad(target_score(model, x, d, target, c_hat).sum(), x)
    return (x * g).sum(1).detach()


def integrated_gradients(model, x, d: int, steps: int = 32, target: str = "logprob", baseline: float = -1.0):
    """(b, H, W) Integrated Gradients from the black image (Riemann right sum, `steps` steps)."""
    model.eval()
    c_hat = predicted_class(model, x, d) if target == "logprob" else None
    base = torch.full_like(x, baseline)
    total = torch.zeros_like(x)
    for k in range(1, steps + 1):
        xk = (base + k / steps * (x - base)).requires_grad_(True)
        total += torch.autograd.grad(target_score(model, xk, d, target, c_hat).sum(), xk)[0]
    return ((x - base) * total / steps).sum(1).detach()


@torch.no_grad()
def occlusion(model, x, d: int, patch: int = 24, stride: int = 24, target: str = "logprob", fill: float = -1.0):
    """(b, H, W) drop of the target score when a patch x patch square is set to `fill`."""
    model.eval()
    c_hat = predicted_class(model, x, d) if target == "logprob" else None
    ref = target_score(model, x, d, target, c_hat)
    h, w = x.shape[-2:]
    heat = torch.zeros(x.shape[0], h, w)
    count = torch.zeros(1, h, w)
    for i in range(0, h, stride):
        for j in range(0, w, stride):
            xo = x.clone()
            xo[..., i:i + patch, j:j + patch] = fill
            drop = (ref - target_score(model, xo, d, target, c_hat)).cpu()
            heat[:, i:i + patch, j:j + patch] += drop[:, None, None]
            count[:, i:i + patch, j:j + patch] += 1
    return heat / count.clamp_min(1)


def band_ranking_from_heatmap(heat, bands):
    """Development protocol: rank bands by sum_{u,v} h(u,v) sum_c |f_b(u,v,c)| (most first).
    heat (b, H, W), bands (b, B, C, H, W) -> (b, B) ranking."""
    h = heat.clamp_min(0).to(bands.device)
    overlap = (h[:, None] * bands.abs().sum(2)).sum(dim=(-1, -2))
    return overlap.argsort(1, descending=True).cpu()


# ----------------------------------------------------------------------------- reference maps
def central_map(size: int, sigma_frac: float = 0.15):
    yy, xx = torch.meshgrid(torch.arange(size), torch.arange(size), indexing="ij")
    c, s = (size - 1) / 2, sigma_frac * size
    return torch.exp(-((yy - c) ** 2 + (xx - c) ** 2) / (2 * s * s))


def random_map(size: int, seed: int = 0):
    return torch.rand(size, size, generator=torch.Generator().manual_seed(seed))


@torch.no_grad()
def stage1_mask_map(model, x, channels=None):
    """Stage-1 segmentation probability used as a heat map (max over `channels`)."""
    _, logits = model.segment(x)
    p = torch.sigmoid(logits)
    if channels is not None:
        p = p[:, list(channels)]
    return p.amax(1).cpu()


# ----------------------------------------------------------------------------- localization
def oracle_area_binarize(heat, mask, factor: float = 1.5):
    """Keep the top round(factor x mask area) pixels of each heat map (oracle-area threshold)."""
    b = heat.shape[0]
    out = torch.zeros_like(mask, dtype=torch.bool)
    flat = heat.reshape(b, -1)
    for i in range(b):
        k = int(round(factor * float(mask[i].sum())))
        if k > 0:
            idx = flat[i].topk(min(k, flat.shape[1])).indices
            out[i].view(-1)[idx] = True
    return out


def threshold_binarize(heat, thr: float):
    """Per-image min-max normalised heat map >= thr."""
    b = heat.shape[0]
    f = heat.reshape(b, -1)
    lo, hi = f.min(1, keepdim=True).values, f.max(1, keepdim=True).values
    return ((f - lo) / (hi - lo).clamp_min(1e-12) >= thr).reshape(heat.shape)


def iou(pred, mask):
    pred, mask = pred.bool(), mask.bool()
    inter = (pred & mask).flatten(1).sum(1).float()
    union = (pred | mask).flatten(1).sum(1).float()
    return torch.where(union > 0, inter / union.clamp_min(1), torch.ones_like(inter))


def select_threshold(heat_val, mask_val, grid=np.linspace(0.05, 0.95, 19)):
    """Threshold maximising mean IoU on the validation part (30%) of a mask set."""
    scores = [float(iou(threshold_binarize(heat_val, t), mask_val).mean()) for t in grid]
    return float(grid[int(np.argmax(scores))]), scores


def pointing_game(heat, mask, tolerance: int = 0):
    """Hit when the maximum of the heat map falls inside the mask (dilated by `tolerance` px)."""
    m = mask.float()[:, None]
    if tolerance > 0:
        m = F.max_pool2d(m, 2 * tolerance + 1, stride=1, padding=tolerance)
    b, w = heat.shape[0], heat.shape[-1]
    idx = heat.reshape(b, -1).argmax(1)
    r, c = idx // w, idx % w
    return m[torch.arange(b), 0, r, c].bool().float()


def split_validation(n: int, frac: float = 0.3, seed: int = 0):
    """Indices of the validation part (for threshold choice) and the evaluation part."""
    perm = np.random.default_rng(seed).permutation(n)
    k = int(round(frac * n))
    return perm[:k], perm[k:]


# ----------------------------------------------------------------------------- sanity checks
def randomize_model(model, seed: int = 0):
    """Model randomization (Adebayo et al.): a copy whose Stage 2 layers are re-initialised."""
    torch.manual_seed(seed)
    m = copy.deepcopy(model)
    for part in (m.tok, m.csf, m.dsr, m.heads):
        if part is None:
            continue
        for mod in part.modules():
            if hasattr(mod, "reset_parameters") and mod is not part:
                mod.reset_parameters()
    if hasattr(m.tok, "band_embed"):
        torch.nn.init.trunc_normal_(m.tok.band_embed, std=0.02)
    return m.eval()


def shuffle_labels_csv(csv_in, csv_out, seed: int = 0):
    """Label randomization: permute the grade columns within the training split (the model
    trained on this CSV gives the 'shuffled-label' maps)."""
    import csv
    import random

    from .config import DISEASES
    with open(csv_in, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    rng = random.Random(seed)
    tr = [r for r in rows if r.get("split", "").lower() == "train"]
    for d in DISEASES:
        vals = [r.get(d, "") for r in tr]
        rng.shuffle(vals)
        for r, v in zip(tr, vals):
            r[d] = v
    with open(csv_out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def map_similarity(maps_a, maps_b, absolute: bool = True):
    """Mean per-image Spearman rho between two sets of maps (b, H, W) or band profiles (b, B)."""
    a, b = (maps_a.abs(), maps_b.abs()) if absolute else (maps_a, maps_b)
    return float(np.mean([spearman(a[i].numpy(), b[i].numpy()) for i in range(a.shape[0])]))


def band_order_stability(profiles_a, profiles_b):
    """Mean Kendall tau between the band profiles of two runs (b, B)."""
    return float(np.mean([kendall_tau(profiles_a[i].numpy(), profiles_b[i].numpy())
                          for i in range(profiles_a.shape[0])]))
