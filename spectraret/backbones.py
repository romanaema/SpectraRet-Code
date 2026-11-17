"""ImageNet backbones from torchvision, shared by Stage 1 and the two DSP streams.

Weights are downloaded on first use (or read from ~/.cache/torch/hub/checkpoints).
"""
from __future__ import annotations

import torch.nn as nn


def mobilenet_v2(pretrained: bool = True) -> nn.Module:
    from torchvision.models import MobileNet_V2_Weights
    from torchvision.models import mobilenet_v2 as build
    return build(weights=MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None)


def swin_v2_t(pretrained: bool = True) -> nn.Module:
    from torchvision.models import Swin_V2_T_Weights
    from torchvision.models import swin_v2_t as build
    return build(weights=Swin_V2_T_Weights.IMAGENET1K_V1 if pretrained else None)
