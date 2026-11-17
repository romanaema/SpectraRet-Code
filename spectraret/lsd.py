"""Fig. 3, box 4 - LSD: undecimated lifting decomposition (Eq. lift and Eq. synth).

Step k:   p_k = P_k(s_k),  d_k = s_k - p_k,  s_{k+1} = p_k + U_k(d_k),   s_0 = x
Inverse:  s_k = (s_{k+1} - U_k(d_k)) + d_k

P_k and U_k are depthwise 9x9 convolutions dilated by 2^k (1, 2, 4). P_k starts as the
separable db4 low-pass filter with unit DC gain, U_k as zero. Each band image f_b is the
synthesis of one coefficient on its own, so sum_b f_b = x exactly. In the final model the
filters are fixed and U_k = 0, so f_b = s_{b-1} - s_b (b = 1, 2, 3) and f_4 = s_3.
Output bands: B1 finest, B2 mid-high, B3 mid-low, B4 coarsest.
"""
from __future__ import annotations

import torch
import torch.nn as nn

# pywt.Wavelet("db4").dec_lo, written out so that PyWavelets is not needed
DB4_DEC_LO = (-0.010597401785069032, 0.0328830116668852, 0.030841381835560764,
              -0.18703481171909309, -0.027983769416859854, 0.6308807679298589,
              0.7148465705529157, 0.2303778133088965)


def db4_lowpass_2d(size: int = 9) -> torch.Tensor:
    """Separable 2-D low-pass kernel from db4 with unit DC gain. The 8 taps sit at positions
    0..7 of the 9-tap window (off-centre, as in the paper)."""
    lo = torch.tensor(DB4_DEC_LO, dtype=torch.float64)
    lo = lo / lo.sum()
    k1 = torch.zeros(size, dtype=torch.float64)
    off = (size - len(lo)) // 2
    k1[off:off + len(lo)] = lo
    k2 = torch.outer(k1, k1)
    return k2 / k2.sum()


class LiftingStep(nn.Module):
    """One undecimated lifting step with zero-padded depthwise predict / update filters."""

    def __init__(self, channels: int = 3, kernel: int = 9, dilation: int = 1):
        super().__init__()
        pad = dilation * (kernel - 1) // 2
        kw = dict(kernel_size=kernel, padding=pad, dilation=dilation, groups=channels, bias=False)
        self.predict = nn.Conv2d(channels, channels, **kw)
        self.update = nn.Conv2d(channels, channels, **kw)
        with torch.no_grad():
            self.predict.weight.copy_(db4_lowpass_2d(kernel).to(torch.float32).expand(channels, 1, kernel, kernel))
            self.update.weight.zero_()

    def forward(self, s):
        p = self.predict(s)
        d = s - p
        return p + self.update(d), d

    def inverse(self, approx, d):
        return approx - self.update(d) + d


class LSD(nn.Module):
    """(b, C, H, W) -> (b, B, C, H, W) band images, B1 finest ... B4 coarsest, summing to x."""

    def __init__(self, n_bands: int = 4, kernel: int = 9, learnable: bool = False, channels: int = 3):
        super().__init__()
        self.steps = nn.ModuleList([LiftingStep(channels, kernel, 2 ** k) for k in range(n_bands - 1)])
        self.learnable = learnable
        for p in self.parameters():
            p.requires_grad = learnable

    def analysis(self, x):
        coeffs, s = [], x
        for step in self.steps:
            s, d = step(s)
            coeffs.append(d)
        coeffs.append(s)
        return coeffs                                   # (d_0, d_1, d_2, s_3)

    def synthesis(self, coeffs):
        s = coeffs[-1]
        for k in range(len(self.steps) - 1, -1, -1):
            s = self.steps[k].inverse(s, coeffs[k])
        return s

    def forward(self, x):
        coeffs = self.analysis(x)
        if not self.learnable and all(not bool(st.update.weight.any()) for st in self.steps):
            # U = 0: telescoping pyramid, the band images are the coefficients themselves
            return torch.stack(coeffs, dim=1)
        bands = [self.synthesis([c if i == b else torch.zeros_like(c) for i, c in enumerate(coeffs)])
                 for b in range(len(coeffs))]
        return torch.stack(bands, dim=1)
