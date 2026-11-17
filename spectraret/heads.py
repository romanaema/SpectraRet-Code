"""Fig. 3, softmax heads - DR head (4 levels), Glaucoma head (4), AMD head (3).

Each disease has a two-layer MLP softmax head on its evidence vector e_d:
Linear(256, 128) -> ReLU -> Dropout(0.3) -> Linear(128, k_d). The same head structure is
reused as the MC-dropout uncertainty head (uncertainty.py).
"""
from __future__ import annotations

import torch.nn as nn


class GradingHead(nn.Sequential):
    """(b, dim) -> (b, k) logits."""

    def __init__(self, dim: int, n_classes: int, hidden: int = 128, drop: float = 0.3):
        super().__init__(nn.Linear(dim, hidden), nn.ReLU(), nn.Dropout(drop), nn.Linear(hidden, n_classes))
