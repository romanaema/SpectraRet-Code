"""SpectraRet - all Fig. 3 modules connected into one model.

    fundus image x (b, 3, 128, 128) in [-1, 1]
        |  segmentation.py   ModifiedDeepLabV3Plus + apply_stage1_mask   (Stage 1, fixed)
        v
    segmented image
        |  lsd.py            LSD                                         -> f_1..f_4
        v
    band images (b, 4, 3, 128, 128)
        |  dsp.py            DualStreamTokenizer
        |                      cnn_stream.py      CNNStream       on B1, B2, B3
        |                      swinecat_stream.py SwinECATStream  on B3, B4
        v
    band tokens z_1..z_4 (b, 4, 256)
        |  csf.py            CSF                  (off in the final model)
        |  dsr.py            DiseaseSpectralRouter  e_d = sum_b w_b^d z_b
        v
    evidence e (b, 3, 256)
        |  heads.py          GradingHead x 3  -> DR (4), Glaucoma (4), AMD (3) logits
        v
    after training:  samx.py (SAM-X)  and  uncertainty.py (uncertainty heads on e_d)
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .config import DISEASES, N_CLASSES, Config
from .csf import CSF
from .dsp import DualStreamTokenizer
from .dsr import DiseaseSpectralRouter
from .heads import GradingHead
from .lsd import LSD
from .segmentation import ModifiedDeepLabV3Plus, apply_stage1_mask


@dataclass
class Output:
    logits: list               # per disease (b, k_d)
    embedding: torch.Tensor    # e (b, D, dim)
    routing: torch.Tensor      # w (b, D, B)
    bands: torch.Tensor        # f (b, B, C, H, W)
    segmented: torch.Tensor    # LSD input (b, C, H, W)
    seg_logits: torch.Tensor | None


class SpectraRet(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        # Stage 1 (box 2): trained first with train.train_stage1, then fixed
        self.seg = ModifiedDeepLabV3Plus(pretrained=cfg.pretrained) if cfg.use_seg else None
        if self.seg is not None:
            for p in self.seg.parameters():
                p.requires_grad = False
        # Stage 2
        self.lsd = LSD(cfg.n_bands, cfg.lsd_kernel, cfg.lsd_learnable)                      # box 4
        self.tok = DualStreamTokenizer(cfg)                                                 # DSP + band tokens
        self.csf = CSF(cfg.dim, cfg.csf_depth, cfg.csf_heads, cfg.csf_mlp_ratio) if cfg.use_csf else None
        self.dsr = DiseaseSpectralRouter(len(DISEASES), cfg.n_bands, cfg.dim, cfg.dsr, cfg.dsr_temperature)
        self.heads = nn.ModuleList([GradingHead(cfg.dim, k, cfg.head_hidden, cfg.head_dropout) for k in N_CLASSES])

    def train(self, mode: bool = True):
        super().train(mode)
        if self.seg is not None:
            self.seg.eval()
        return self

    def load_stage1(self, path):
        ck = torch.load(path, map_location="cpu", weights_only=True)
        self.seg.load_state_dict(ck["state"])

    # --- the pipeline, step by step ------------------------------------------------------
    def segment(self, x):
        """Fundus image -> segmented image (Stage 1, no gradient)."""
        if self.seg is None:
            return x, None
        with torch.no_grad():
            logits = self.seg(x)
        return apply_stage1_mask(x, logits, self.cfg.mask_mode, self.cfg.mask_threshold), logits

    def bands(self, x_seg):
        """Segmented image -> band images (LSD), or copies of the raw image for the control."""
        if self.cfg.raw_control:
            return x_seg.unsqueeze(1).expand(-1, self.cfg.n_bands, -1, -1, -1)
        return self.lsd(x_seg)

    def forward_from_bands(self, bands):
        """Band images -> tokens -> (CSF) -> DSR -> heads. SAM-X, band IG and band deletion enter here."""
        z = self.tok(bands)
        if self.csf is not None:
            z = self.csf(z)
        e, w = self.dsr(z)
        return [head(e[:, d]) for d, head in enumerate(self.heads)], e, w

    def forward(self, x) -> Output:
        x_seg, seg_logits = self.segment(x)
        f = self.bands(x_seg)
        logits, e, w = self.forward_from_bands(f)
        return Output(logits, e, w, f, x_seg, seg_logits)
