"""Fig. 3, DSP - CNN stream: frozen ImageNet MobileNetV2 on bands B1, B2, B3.

A 1x1 identity-initialised adapter feeds each band at 128 x 128; the pooled stride-32
features (1280) are projected to a 256-d token. The backbone stays frozen and in eval mode.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbones import mobilenet_v2


class ChannelAdapter(nn.Module):
    """1x1 convolution initialised to the identity."""

    def __init__(self, channels: int = 3):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 1)
        with torch.no_grad():
            self.conv.weight.copy_(torch.eye(channels).view(channels, channels, 1, 1))
            self.conv.bias.zero_()

    def forward(self, x):
        return self.conv(x)


class CNNStream(nn.Module):
    """(n, 3, H, W) band images -> (n, dim) tokens c_b(f_b)."""

    def __init__(self, dim: int = 256, size: int = 128, pretrained: bool = True):
        super().__init__()
        self.size = size
        self.adapter = ChannelAdapter(3)
        self.backbone = mobilenet_v2(pretrained).features
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.proj = nn.Linear(1280, dim)

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()                            # frozen: BatchNorm statistics stay fixed
        return self

    def forward(self, x):
        x = self.adapter(x)
        if x.shape[-1] != self.size:
            x = F.interpolate(x, size=(self.size, self.size), mode="bilinear", align_corners=False)
        return self.proj(self.backbone(x).mean(dim=(2, 3)))
