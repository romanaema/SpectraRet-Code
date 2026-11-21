"""Section 6.6 / Table 19 - computational cost.

Total and trainable parameters, MACs (FLOPs taken as 2 x MACs) and CPU throughput on one CPU
with four threads, float32, batch size 8, 20 timed passes after five warm-up passes. MACs follow
the calls actually executed (SpectraRet: segmentation once, CNN on three bands, transformer on
two; the raw-image control runs the same passes on copies of the raw image). fvcore is used
when installed, otherwise torch.utils.flop_counter. Uncertainty heads are timed separately on
the evidence vectors and added to the encoder time. Reported times are batch throughput, not
single-image latency.
"""
from __future__ import annotations

import time

import torch

from .config import count_params


@torch.no_grad()
def count_macs(model, x):
    """Multiply-accumulate operations of one forward pass of x, per image."""
    model.eval()
    try:
        from fvcore.nn import FlopCountAnalysis               # counts one MAC as one "flop"
        macs = FlopCountAnalysis(model, (x,)).unsupported_ops_warnings(False).uncalled_modules_warnings(False).total()
    except ImportError:
        from torch.utils.flop_counter import FlopCounterMode   # counts one MAC as two FLOPs
        with FlopCounterMode(display=False) as fc:
            model(x)
        macs = fc.get_total_flops() / 2
    return macs / x.shape[0]


@torch.no_grad()
def cpu_throughput_ms(fn, x, warmup: int = 5, timed: int = 20, threads: int = 4):
    """Milliseconds per image of fn(x) for a batch x (batch throughput)."""
    torch.set_num_threads(threads)
    for _ in range(warmup):
        fn(x)
    t0 = time.perf_counter()
    for _ in range(timed):
        fn(x)
    return 1000.0 * (time.perf_counter() - t0) / (timed * x.shape[0])


def cost_report(model, image_size: int = 128, batch: int = 8, threads: int = 4):
    model.eval()
    x = torch.rand(batch, 3, image_size, image_size) * 2 - 1
    total, trained = count_params(model)
    macs = count_macs(model, x)
    return {"params_M": total / 1e6, "trained_M": trained / 1e6, "GMACs": macs / 1e9, "GFLOPs": 2 * macs / 1e9,
            "cpu_ms_per_image": cpu_throughput_ms(model, x, threads=threads), "batch": batch, "threads": threads}


def head_cost_ms(head_fn, dim: int = 256, batch: int = 8, threads: int = 4):
    """Extra time of an uncertainty head on the evidence vectors (e.g. 30 MC-dropout passes)."""
    return cpu_throughput_ms(head_fn, torch.randn(batch, dim), threads=threads)
