"""Fig. 3, uncertainty heads on e_d (Eq. gp, Eq. mf).

All heads use the same frozen evidence vector, separating the method from the encoder:
  deterministic   the grading head with dropout off
  temperature     softmax(logits / T), T fitted on validation logits only
  MC dropout      GradingHead (128, dropout 0.3) trained on e, 30 stochastic passes
  SNGP-style      standardised e, 512 random Fourier features, Laplace precision,
                  mean-field softmax (no spectral normalisation of the encoder)
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SNGPStyleHead(nn.Module):
    """phi(e) = sqrt(2/M) cos(W e~ / l + beta); Sigma = (I + sum_i p_i(1-p_i) phi_i phi_i^T)^-1;
    sigma^2 = phi^T Sigma phi; p = softmax(mu / sqrt(1 + pi sigma^2 / 8)).
    l = median pairwise distance of the standardised training embeddings,
    p_i = maximum softmax probability."""

    def __init__(self, dim: int, n_classes: int, n_rff: int = 512, seed: int = 0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.n_rff = n_rff
        self.register_buffer("W", torch.randn(dim, n_rff, generator=g))
        self.register_buffer("b", torch.rand(n_rff, generator=g) * 2 * math.pi)
        self.register_buffer("mu", torch.zeros(dim))
        self.register_buffer("sd", torch.ones(dim))
        self.register_buffer("ell", torch.tensor(1.0))
        self.register_buffer("cov", torch.eye(n_rff))
        self.beta = nn.Linear(n_rff, n_classes, bias=False)

    @torch.no_grad()
    def adapt(self, e, max_pairs: int = 2000, seed: int = 0):
        """Standardisation and median-heuristic length scale from the training embeddings."""
        self.mu.copy_(e.mean(0))
        self.sd.copy_(e.std(0) + 1e-6)
        z = (e - self.mu) / self.sd
        g = torch.Generator().manual_seed(seed)
        i, j = torch.randint(0, len(z), (2, max_pairs), generator=g)
        keep = i != j
        if keep.any():
            self.ell.fill_(float((z[i[keep]] - z[j[keep]]).norm(dim=1).median()))

    def phi(self, e):
        z = (e - self.mu) / (self.sd * self.ell)
        return math.sqrt(2.0 / self.n_rff) * torch.cos(z @ self.W + self.b)

    def forward(self, e):
        return self.beta(self.phi(e))

    @torch.no_grad()
    def fit_laplace(self, e):
        phi = self.phi(e)
        p = torch.softmax(self.beta(phi), 1).amax(1, keepdim=True)
        precision = torch.eye(self.n_rff) + (phi * p * (1 - p)).T @ phi
        self.cov.copy_(torch.linalg.inv(precision))

    @torch.no_grad()
    def predict(self, e):
        phi = self.phi(e)
        var = ((phi @ self.cov) * phi).sum(1, keepdim=True)
        return torch.softmax(self.beta(phi) / torch.sqrt(1 + math.pi * var / 8), 1)


def fit_head(head: nn.Module, e, y, epochs: int = 40, lr: float = 1e-3, batch: int = 32, seed: int = 0):
    """Adam, lr 1e-3, batch 32, 40 epochs - the setting used for the uncertainty heads."""
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam([p for p in head.parameters() if p.requires_grad], lr=lr)
    head.train()
    for _ in range(epochs):
        perm = torch.randperm(len(e), generator=g)
        for s in range(0, len(e), batch):
            idx = perm[s:s + batch]
            loss = F.cross_entropy(head(e[idx]), y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    return head.eval()


@torch.no_grad()
def mc_dropout_predict(head: nn.Module, e, passes: int = 30):
    head.train()                                         # dropout stays on
    probs = torch.stack([torch.softmax(head(e), 1) for _ in range(passes)]).mean(0)
    head.eval()
    return probs


def fit_temperature(logits, y):
    """Temperature scaling fitted by NLL on validation logits only."""
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=200)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(logits / log_t.exp(), y)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.exp())
