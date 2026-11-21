"""Ordinal CORAL head (Cao, Mirjalili & Raschka, 2020) - Section 5.5.

Tried in place of the softmax heads; it collapsed to the class prior on the hospital sources,
so every reported model uses cross-entropy (preset "coral" re-runs it).

The head shares one weight vector across the K - 1 binary tasks "y > k" and learns K - 1
biases, which gives rank-consistent cumulative probabilities P(y > k) = sigmoid(g(e) + b_k).
forward() returns class log-probabilities, so softmax(output) is the class distribution and
all softmax-based code (metrics, SAM-X, uncertainty) works unchanged; the cumulative logits of
the last call are kept in .last_cum for the CORAL loss.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CoralHead(nn.Module):
    """(b, dim) -> (b, k) class log-probabilities; .last_cum (b, k - 1) cumulative logits."""

    def __init__(self, dim: int, n_classes: int, hidden: int = 128, drop: float = 0.3):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(), nn.Dropout(drop))
        self.score = nn.Linear(hidden, 1, bias=False)
        self.bias = nn.Parameter(torch.arange(n_classes - 1, 0, -1, dtype=torch.float32) / n_classes)
        self.last_cum = None

    def cumulative_logits(self, e):
        return self.score(self.body(e)) + self.bias                 # (b, k - 1)

    def forward(self, e):
        cum = self.cumulative_logits(e)
        self.last_cum = cum
        p_gt = torch.sigmoid(cum)                                   # P(y > k), k = 0..K-2
        ones, zeros = torch.ones_like(p_gt[:, :1]), torch.zeros_like(p_gt[:, :1])
        upper = torch.cat([ones, p_gt], 1)                           # P(y > k - 1)
        lower = torch.cat([p_gt, zeros], 1)                          # P(y > k)
        probs = (upper - lower).clamp_min(1e-8)
        return (probs / probs.sum(1, keepdim=True)).log()


def coral_targets(y, n_classes: int):
    """Extended binary labels: t_k = 1[y > k], k = 0..K-2."""
    return (y[:, None] > torch.arange(n_classes - 1, device=y.device)[None]).float()


def coral_loss(cum_logits, y, n_classes: int, reduction: str = "sum"):
    """Sum of the K - 1 binary cross-entropies per image."""
    loss = F.binary_cross_entropy_with_logits(cum_logits, coral_targets(y, n_classes), reduction="none").sum(1)
    return loss.sum() if reduction == "sum" else loss.mean()
