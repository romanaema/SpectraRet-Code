"""Fig. 3, DSP box + band tokens - dual-stream tokenizer (Eq. tokens).

    t_1 = c(f_1),  t_2 = c(f_2),  t_3 = (c(f_3) + v(f_3)) / 2,  t_4 = v(f_4)
    z_b = t_b + a_b          (a_b = learned band embedding)

c = CNN stream (cnn_stream.py), v = SwinECAT stream (swinecat_stream.py). Each stream
shares its weights across the bands it reads; the bands are folded into the batch so each
stream runs once per forward pass.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .cnn_stream import CNNStream
from .config import ASSIGNMENTS, Config
from .swinecat_stream import SwinECATStream


class DualStreamTokenizer(nn.Module):
    """(b, B, C, H, W) band images -> (b, B, dim) band tokens z_1..z_B."""

    def __init__(self, cfg: Config):
        super().__init__()
        a = ASSIGNMENTS[cfg.assignment]
        self.cnn_idx, self.swin_idx = tuple(a["cnn"]), tuple(a["swin"])
        missing = set(range(cfg.n_bands)) - set(self.cnn_idx) - set(self.swin_idx)
        if missing:
            raise ValueError(f"bands {sorted(missing)} are read by neither stream")
        self.cnn = CNNStream(cfg.dim, cfg.image_size, cfg.pretrained) if self.cnn_idx else None
        self.swin = SwinECATStream(cfg.dim, cfg.swin_size, cfg.pretrained) if self.swin_idx else None
        self.band_embed = nn.Parameter(nn.init.trunc_normal_(torch.empty(1, cfg.n_bands, cfg.dim), std=0.02))

    @staticmethod
    def _run(stream, bands, idx):
        b = bands.shape[0]
        x = bands[:, list(idx)].flatten(0, 1)           # (b * len(idx), C, H, W)
        out = stream(x).view(b, len(idx), -1)
        return {i: out[:, j] for j, i in enumerate(idx)}

    def forward(self, bands):
        c = self._run(self.cnn, bands, self.cnn_idx) if self.cnn is not None else {}
        v = self._run(self.swin, bands, self.swin_idx) if self.swin is not None else {}
        tokens = [torch.stack([t[b] for t in (c, v) if b in t]).mean(0) for b in range(bands.shape[1])]
        return torch.stack(tokens, dim=1) + self.band_embed
