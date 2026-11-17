"""Shared constants and the model configuration.

The presets are the paper's model and its control / ablation arms. Every module of Fig. 3
can be switched by a flag here, which is how the ablations were run.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch.nn as nn

DISEASES = ("DR", "Glaucoma", "AMD")
N_CLASSES = (4, 4, 3)                     # DR 4 levels, Glaucoma 4, AMD 3
SEG_CHANNELS = ("MA", "HEM", "HE", "SE", "IRMA", "NV", "Drusen", "GA", "CNV", "OD", "OC")
SEG_GROUPS = {"DR": (0, 1, 2, 3, 4, 5), "AMD": (6, 7, 8), "Glaucoma": (9, 10)}
BAND_NAMES = ("B1 finest", "B2 mid-high", "B3 mid-low", "B4 coarsest")

# which bands each stream reads (0-based: 0 = B1 finest ... 3 = B4 coarsest)
ASSIGNMENTS = {
    "default": {"cnn": (0, 1, 2), "swin": (2, 3)},
    "inverted": {"cnn": (2, 3), "swin": (0, 1, 2)},
    "all_cnn": {"cnn": (0, 1, 2, 3), "swin": ()},
}


@dataclass
class Config:
    image_size: int = 128          # Stage 1, LSD and CNN-stream resolution
    swin_size: int = 256           # bands are resized to this for SwinECAT
    n_bands: int = 4
    dim: int = 256                 # band-token width
    # Stage 1
    use_seg: bool = True
    mask_mode: str = "hard"        # hard: keep pixels with p >= threshold | soft: weight by p
    mask_threshold: float = 0.5
    # LSD
    lsd_kernel: int = 9
    lsd_learnable: bool = False    # final model: fixed filters with U_k = 0
    raw_control: bool = False      # raw image in place of every band
    # DSP
    assignment: str = "default"
    # CSF and DSR
    use_csf: bool = False
    csf_depth: int = 2
    csf_heads: int = 4
    csf_mlp_ratio: float = 2.0
    dsr: str = "uniform"           # uniform (final model) | learned
    dsr_temperature: float = 1.0
    # heads and loss
    head_hidden: int = 128
    head_dropout: float = 0.3
    lambda_recon: float = 1e-3     # numerical check only: the term is zero up to rounding
    lambda_sparsity: float = 0.05  # routing-entropy penalty, learned DSR only
    pretrained: bool = True        # ImageNet weights for MobileNetV2 and Swin-V2-T


PRESETS = {
    "final": {},                                        # reported model
    "full": {"use_csf": True, "dsr": "learned"},        # every box of Fig. 3
    "learnable_lsd": {"lsd_learnable": True},
    "inverted": {"assignment": "inverted"},
    "all_cnn": {"assignment": "all_cnn"},
    "raw_control": {"raw_control": True, "use_seg": False},
}


def count_params(module: nn.Module) -> tuple[int, int]:
    """(total, trained) parameter counts."""
    total = sum(p.numel() for p in module.parameters())
    trained = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trained
