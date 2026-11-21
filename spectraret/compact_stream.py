"""Earlier model, transformer stream - compact transformer trained from scratch (Section 5.6 (iv)).

It preceded SwinECAT: a small Vision-Transformer-style encoder with 96 x 96 input and two
pre-norm transformer blocks (Vaswani et al., 2017), trained from scratch together with the
rest of Stage 2. Bands are resized to 96 x 96, cut into 8 x 8 patches (144 tokens), passed
through the blocks, average-pooled and projected to a 256-d token, so it is a drop-in
replacement for SwinECATStream in the dual-stream tokenizer.

The paper fixes the input size and the number of blocks; width 128, four heads, MLP ratio 4
and patch size 8 are implementation choices (they give the reported 0.90 M trained
parameters of the earlier model together with the CNN-stream projection and the heads).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .csf import CrossBandBlock


class CompactTransformerStream(nn.Module):
    """(n, 3, H, W) band images -> (n, dim) tokens v_b(f_b)."""

    def __init__(self, dim: int = 256, size: int = 96, width: int = 128, depth: int = 2,
                 heads: int = 4, patch: int = 8, mlp_ratio: float = 4.0):
        super().__init__()
        if size % patch:
            raise ValueError("compact transformer: size must be a multiple of the patch size")
        self.size = size
        self.patch_embed = nn.Conv2d(3, width, patch, stride=patch)
        n_tokens = (size // patch) ** 2
        self.pos = nn.Parameter(nn.init.trunc_normal_(torch.empty(1, n_tokens, width), std=0.02))
        self.blocks = nn.Sequential(*[CrossBandBlock(width, heads, mlp_ratio) for _ in range(depth)])
        self.norm = nn.LayerNorm(width, eps=1e-6)
        self.proj = nn.Linear(width, dim)

    def backbone_parameters(self):
        """Trained from scratch with the other Stage 2 layers (Adam, lr 1e-3): no separate group."""
        return iter(())

    def forward(self, x):
        if x.shape[-1] != self.size:
            x = F.interpolate(x, size=(self.size, self.size), mode="bilinear", align_corners=False)
        t = self.patch_embed(x).flatten(2).transpose(1, 2) + self.pos      # (n, tokens, width)
        t = self.norm(self.blocks(t))
        return self.proj(t.mean(1))
