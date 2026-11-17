"""Fig. 3, CSF box - cross-spectral fusion.

The four band tokens attend to each other through two pre-norm transformer blocks
(four heads, MLP ratio two). Tested and dropped: off in the final model (preset "final"),
on in preset "full".
"""
from __future__ import annotations

import torch.nn as nn


class CrossBandBlock(nn.Module):
    """Pre-norm transformer block: every band token attends to all band tokens."""

    def __init__(self, dim: int, heads: int = 4, mlp_ratio: float = 2.0, drop: float = 0.0):
        super().__init__()
        self.n1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=drop, batch_first=True)
        self.n2 = nn.LayerNorm(dim, eps=1e-6)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(drop), nn.Linear(hidden, dim))

    def forward(self, z):                               # (b, B, dim)
        h = self.n1(z)
        z = z + self.attn(h, h, h, need_weights=False)[0]
        return z + self.mlp(self.n2(z))


class CSF(nn.Sequential):
    """(b, B, dim) -> (b, B, dim)."""

    def __init__(self, dim: int, depth: int = 2, heads: int = 4, mlp_ratio: float = 2.0):
        super().__init__(*[CrossBandBlock(dim, heads, mlp_ratio) for _ in range(depth)])
