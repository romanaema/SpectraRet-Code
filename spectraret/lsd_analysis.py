"""Section 5.2 / Fig. 4 - analysis of the fixed LSD operators.

  reconstruction_error       max |sum_b f_b - x| in double precision (reported 3e-16)
  band_impulse_responses     response of each band operator to a centred unit impulse
  radial_magnitude_response  radially averaged |FFT| of each band operator (Fig. 4b); band 1
                             passes most energy above 0.15 cycles/pixel and band 4 is low-pass
                             below about 0.06
  energy_above               fraction of a band's squared response above a frequency
  band_images                example band images averaged over colour channels (Fig. 4a)
The eight-tap db4 filter sits one tap off centre in the 9 x 9 support and borders are
zero-padded, so responses are phase dependent; this analysis uses a centred impulse.
"""
from __future__ import annotations

import numpy as np
import torch

from .lsd import LSD


@torch.no_grad()
def reconstruction_error(lsd: LSD | None = None, size: int = 128, seed: int = 0):
    lsd = (lsd or LSD(4, 9, learnable=False)).double()
    x = torch.rand(2, 3, size, size, dtype=torch.float64, generator=torch.Generator().manual_seed(seed)) * 2 - 1
    return float((lsd(x).sum(1) - x).abs().max())


@torch.no_grad()
def band_impulse_responses(lsd: LSD | None = None, size: int = 128):
    """(B, H, W) responses of the band operators to a unit impulse at the image centre."""
    lsd = (lsd or LSD(4, 9, learnable=False)).double()
    x = torch.zeros(1, 3, size, size, dtype=torch.float64)
    x[..., size // 2, size // 2] = 1.0
    return lsd(x)[0, :, 0].numpy()


def radial_magnitude_response(responses, n_bins: int = 64):
    """-> frequencies (cycles/pixel, 0..0.5) and (B, n_bins) radially averaged |FFT|."""
    b, h, w = responses.shape
    mag = np.abs(np.fft.fftshift(np.fft.fft2(responses), axes=(-2, -1)))
    fy = np.fft.fftshift(np.fft.fftfreq(h))
    fx = np.fft.fftshift(np.fft.fftfreq(w))
    r = np.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)
    edges = np.linspace(0, 0.5, n_bins + 1)
    centres = (edges[:-1] + edges[1:]) / 2
    curves = np.zeros((b, n_bins))
    for i in range(n_bins):
        m = (r >= edges[i]) & (r < edges[i + 1])
        if m.any():
            curves[:, i] = mag[:, m].mean(1)
    return centres, curves


def energy_above(freqs, curve, f0: float):
    e = np.asarray(curve) ** 2
    return float(e[np.asarray(freqs) >= f0].sum() / max(e.sum(), 1e-30))


@torch.no_grad()
def band_images(x, lsd: LSD | None = None):
    """(b, 3, H, W) images in [-1, 1] -> (b, B, H, W) band images averaged over colour channels."""
    lsd = lsd or LSD(4, 9, learnable=False)
    return lsd(x).mean(2)
