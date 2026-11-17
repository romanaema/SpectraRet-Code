"""Fig. 3, DSR box - disease-spectral router:  e_d = sum_b w_b^d z_b.

uniform (final model): w_b^d = 1/B, so e is the average of the band tokens.
learned (preset "full"): w^d = softmax_b(q_d . K z_b / T) with learned disease queries q_d,
temperature 1 and an entropy penalty (weight 0.05) that pushes each disease to few bands.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class DiseaseSpectralRouter(nn.Module):
    """(b, B, dim) band tokens -> e (b, D, dim) evidence vectors, w (b, D, B) routing weights."""

    def __init__(self, n_diseases: int, n_bands: int, dim: int, mode: str = "uniform",
                 temperature: float = 1.0):
        super().__init__()
        if mode not in ("uniform", "learned"):
            raise ValueError("dsr must be uniform or learned")
        self.mode, self.n_diseases, self.n_bands, self.temperature = mode, n_diseases, n_bands, temperature
        if mode == "learned":
            self.queries = nn.Parameter(nn.init.trunc_normal_(torch.empty(n_diseases, dim), std=0.02))
            self.key = nn.Linear(dim, dim)

    def forward(self, z):
        if self.mode == "uniform":
            w = z.new_full((z.shape[0], self.n_diseases, self.n_bands), 1.0 / self.n_bands)
        else:
            logits = torch.einsum("dc,bkc->bdk", self.queries, self.key(z))
            w = torch.softmax(logits / self.temperature, dim=-1)
        return torch.einsum("bdk,bkc->bdc", w, z), w


def routing_entropy(w, eps: float = 1e-8):
    """Mean entropy of the per-disease band distributions (the sparsity penalty)."""
    return -(w * (w + eps).log()).sum(-1).mean()
