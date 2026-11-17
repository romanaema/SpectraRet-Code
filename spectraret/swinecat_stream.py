"""Fig. 3, DSP - SwinECAT stream: Swin-V2-Tiny + ECA on bands B3 and B4.

ImageNet Swin-V2-Tiny, efficient channel attention before pooling (as in the SwinECAT
code), fully fine-tuned on bands resized to 256 x 256. The pooled 768-d output is
projected to a 256-d token.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbones import swin_v2_t


class ECA(nn.Module):
    """Efficient channel attention (Wang et al., 2020)."""

    def __init__(self, channels: int, gamma: int = 2, b: int = 1):
        super().__init__()
        k = int(abs((math.log2(channels) + b) / gamma))
        k = max(k if k % 2 else k + 1, 3)
        self.conv = nn.Conv1d(1, 1, k, padding=(k - 1) // 2, bias=False)

    def forward(self, x):                               # (b, C, H, W)
        y = x.mean(dim=(2, 3)).unsqueeze(1)             # (b, 1, C)
        y = torch.sigmoid(self.conv(y)).transpose(1, 2).unsqueeze(-1)   # (b, C, 1, 1)
        return x * y


class SwinECATStream(nn.Module):
    """(n, 3, H, W) band images -> (n, dim) tokens v_b(f_b)."""

    def __init__(self, dim: int = 256, size: int = 256, pretrained: bool = True):
        super().__init__()
        m = swin_v2_t(pretrained)
        self.size = size
        self.features, self.norm, self.permute = m.features, m.norm, m.permute
        self.eca = ECA(m.head.in_features)
        self.proj = nn.Linear(m.head.in_features, dim)

    def backbone_parameters(self):
        """Parameters trained with AdamW (lr 3e-5) - everything except the projection."""
        for mod in (self.features, self.norm, self.eca):
            yield from mod.parameters()

    def forward(self, x):
        if x.shape[-1] != self.size:
            x = F.interpolate(x, size=(self.size, self.size), mode="bilinear", align_corners=False)
        h = self.permute(self.norm(self.features(x)))  # (n, 768, H/32, W/32)
        return self.proj(self.eca(h).mean(dim=(2, 3)))
