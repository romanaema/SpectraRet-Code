"""Section 7.3 (final model) - pixel-level MC dropout in the Stage-1 decoder.

A decoder variant of the modified DeepLabV3+ with spatial dropout before the classifier, run
for 30 stochastic passes. Reported: MC-dropout variance ranked wrong pixels (AUROC 0.846 vs
0.812 for a single pass), image-level uncertainty correlated with Dice (rho = -0.47), and
referring the 10% most uncertain images raised the mean Dice of the rest by 0.034; the MC mean
map did not improve Dice. Thirty decoder passes added 31% to the model time.
The dropout rate (0.3) is an implementation choice; the paper does not state it.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .metrics import auroc
from .segmentation import ModifiedDeepLabV3Plus
from .stats import spearman


class MCDropoutDeepLabV3Plus(ModifiedDeepLabV3Plus):
    """ModifiedDeepLabV3Plus with Dropout2d between the decoder and the classifier."""

    def __init__(self, *args, p: float = 0.3, **kwargs):
        super().__init__(*args, **kwargs)
        self.mc_drop = nn.Dropout2d(p)
        old = self.classifier
        self.classifier = nn.Sequential(self.mc_drop, old)

    @torch.no_grad()
    def mc_predict(self, x, passes: int = 30):
        """-> mean probability (b, C, H, W), variance (b, C, H, W), single-pass probability."""
        self.eval()
        single = torch.sigmoid(self(x))
        self.mc_drop.train()                              # only the dropout layer is stochastic
        probs = torch.stack([torch.sigmoid(self(x)) for _ in range(passes)])
        self.mc_drop.eval()
        return probs.mean(0), probs.var(0), single


def pixel_error_auroc(prob, score, target, sup, thr: float = 0.5, max_pixels: int = 2_000_000, seed: int = 0):
    """AUROC of a pixel score (variance, or 1 - |2p - 1| for a single pass) for separating wrong
    pixels from correct ones, over the annotated channels only."""
    m = sup[:, :, None, None].expand_as(prob).bool()
    wrong = ((prob >= thr).float() != target)[m].numpy()
    s = score[m].numpy()
    if len(s) > max_pixels:
        idx = np.random.default_rng(seed).choice(len(s), max_pixels, replace=False)
        s, wrong = s[idx], wrong[idx]
    return auroc(s, wrong)


def dice_per_image(prob, target, sup, thr: float = 0.5, smooth: float = 1.0):
    pred = (prob >= thr).float()
    m = sup[:, :, None, None]
    inter = (pred * target * m).sum(dim=(1, 2, 3))
    denom = ((pred + target) * m).sum(dim=(1, 2, 3))
    return ((2 * inter + smooth) / (denom + smooth)).numpy()


def image_uncertainty(var, sup):
    m = sup[:, :, None, None].expand_as(var)
    return ((var * m).sum(dim=(1, 2, 3)) / m.sum(dim=(1, 2, 3)).clamp_min(1)).numpy()


def segmentation_referral(unc, dice, refer_frac: float = 0.1):
    """Mean Dice of the retained images after referring the most uncertain fraction."""
    keep = np.argsort(unc, kind="stable")[: int(round((1 - refer_frac) * len(unc)))]
    return {"dice_all": float(np.mean(dice)), "dice_retained": float(np.mean(dice[keep])),
            "gain": float(np.mean(dice[keep]) - np.mean(dice)),
            "spearman_unc_dice": spearman(unc, dice)}
