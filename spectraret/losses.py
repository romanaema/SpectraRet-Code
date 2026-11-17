"""Losses: masked cross-entropy (Eq. loss) for Stage 2, masked BCE + Dice for Stage 1."""
from __future__ import annotations

import torch.nn.functional as F

from .config import Config
from .dsr import routing_entropy


def masked_ce(logits_list, y, mask):
    """Cross-entropy averaged over the labelled (image, disease) pairs; mask (b, D) bool."""
    total, n = 0.0, 0
    for d, lg in enumerate(logits_list):
        m = mask[:, d]
        if m.any():
            total = total + F.cross_entropy(lg[m], y[m, d], reduction="sum")
            n += int(m.sum())
    return total / max(n, 1)


def spectraret_loss(out, y, mask, cfg: Config):
    """Masked CE + reconstruction check (zero up to rounding) + routing entropy (learned DSR)."""
    loss = masked_ce(out.logits, y, mask)
    if not cfg.raw_control and cfg.lambda_recon > 0:
        loss = loss + cfg.lambda_recon * (out.segmented - out.bands.sum(1)).pow(2).mean()
    if cfg.dsr == "learned" and cfg.lambda_sparsity > 0:
        loss = loss + cfg.lambda_sparsity * routing_entropy(out.routing)
    return loss


def masked_seg_loss(logits, target, sup, smooth: float = 1.0):
    """BCE + soft Dice on the annotated (image, channel) pairs only. sup (b, C) is 1 where
    channel c of image i was annotated."""
    m = sup[:, :, None, None]
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none").mean(dim=(2, 3))
    bce = (bce * sup).sum() / sup.sum().clamp_min(1e-6)
    p = logits.sigmoid()
    inter = (p * target * m).sum(dim=(0, 2, 3))
    denom = ((p + target) * m).sum(dim=(0, 2, 3))
    dice = (2 * inter + smooth) / (denom + smooth)
    valid = (sup.sum(0) > 0).float()
    return bce + ((1 - dice) * valid).sum() / valid.sum().clamp_min(1e-6)
