"""Fig. 3, box 2 - Stage 1 segmentation: modified DeepLabV3+ (r77).

MobileNetV2 encoder, spatial attention at three points (between the encoder and ASPP,
after the first upsampling, before the second upsampling), ASPP, and a two-step decoder.
Its predicted mask is applied to the fundus image to give the segmented image (box 3).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbones import mobilenet_v2
from .config import SEG_CHANNELS


def conv_bn_relu(cin: int, cout: int, k: int = 1, dilation: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, k, padding=dilation * (k // 2), dilation=dilation, bias=False),
        nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class SpatialAttention(nn.Module):
    """Y = X * sigmoid(conv7x7([mean_c(X), max_c(X)]))."""

    def __init__(self, kernel: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel, padding=kernel // 2)

    def forward(self, x):
        a = torch.cat([x.mean(1, keepdim=True), x.amax(1, keepdim=True)], dim=1)
        return x * torch.sigmoid(self.conv(a))


class ASPP(nn.Module):
    """1x1 branch + three atrous branches + image-level pooling, fused by a 1x1 conv."""

    def __init__(self, cin: int, cout: int = 256, rates=(6, 12, 18)):
        super().__init__()
        self.branches = nn.ModuleList([conv_bn_relu(cin, cout, 1)] +
                                      [conv_bn_relu(cin, cout, 3, r) for r in rates])
        # no BatchNorm on the 1x1 pooled branch, so a batch of one still trains
        self.pool = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(cin, cout, 1), nn.ReLU(inplace=True))
        self.project = nn.Sequential(conv_bn_relu(cout * (len(rates) + 2), cout, 1), nn.Dropout(0.1))

    def forward(self, x):
        outs = [b(x) for b in self.branches]
        outs.append(F.interpolate(self.pool(x), size=x.shape[-2:], mode="bilinear", align_corners=False))
        return self.project(torch.cat(outs, dim=1))


class ModifiedDeepLabV3Plus(nn.Module):
    """Input (b, 3, H, W) in [-1, 1]; output (b, 11, H, W) logits, one sigmoid channel per
    structure in SEG_CHANNELS (6 DR lesions, 3 AMD lesions, optic disc, optic cup).

    Encoder output stride 16. Decoder: ASPP -> x4 upsampling -> concat with the stride-4
    features -> two 3x3 refinement convs -> x4 upsampling to the input size.
    """

    def __init__(self, n_classes: int = len(SEG_CHANNELS), pretrained: bool = True):
        super().__init__()
        feats = mobilenet_v2(pretrained).features
        for i in range(14, 18):                  # output stride 32 -> 16, dilated blocks
            dw = feats[i].conv[1][0]
            if i == 14:
                dw.stride = (1, 1)
            dw.dilation, dw.padding = (2, 2), (2, 2)
        self.low = feats[:4]                     # stride 4, 24 channels
        self.high = feats[4:18]                  # stride 16, 320 channels
        self.att_enc = SpatialAttention()
        self.aspp = ASPP(320, 256)
        self.att_up1 = SpatialAttention()
        self.low_proj = conv_bn_relu(24, 48, 1)
        self.refine = nn.Sequential(conv_bn_relu(256 + 48, 256, 3), conv_bn_relu(256, 256, 3))
        self.att_up2 = SpatialAttention()
        self.classifier = nn.Conv2d(256, n_classes, 1)
        nn.init.constant_(self.classifier.bias, -4.0)   # lesions are sparse: start near "absent"

    def forward(self, x):
        size = x.shape[-2:]
        low = self.low(x)
        h = self.aspp(self.att_enc(self.high(low)))
        h = F.interpolate(h, size=low.shape[-2:], mode="bilinear", align_corners=False)  # step 1
        h = self.att_up1(h)
        h = self.refine(torch.cat([h, self.low_proj(low)], dim=1))
        h = self.att_up2(h)
        return F.interpolate(self.classifier(h), size=size, mode="bilinear", align_corners=False)  # step 2


def apply_stage1_mask(x, seg_logits, mode: str = "hard", threshold: float = 0.5):
    """Segmented image. x in [-1, 1]; pixels outside the predicted structures become black (-1).
    The union over all channels is used because the disease is not known at test time.
    If no pixel passes the threshold the image is passed on unmasked (implementation choice:
    the paper does not describe this case)."""
    p = torch.sigmoid(seg_logits).amax(1, keepdim=True)
    if mode == "hard":
        m = (p >= threshold).to(x.dtype)
        empty = m.flatten(1).amax(1) == 0
        m = torch.where(empty.view(-1, 1, 1, 1), torch.ones_like(m), m)
    elif mode == "soft":
        m = p.to(x.dtype)
    else:
        raise ValueError(f"mask_mode must be hard or soft, got {mode}")
    return (x + 1.0) * m - 1.0
