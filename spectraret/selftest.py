"""Self-test on random tensors: every module of Fig. 3, connected, without any data files."""
from __future__ import annotations

from dataclasses import asdict

import torch

from .config import PRESETS, Config, count_params
from .heads import GradingHead
from .losses import spectraret_loss
from .lsd import LSD
from .metrics import uncertainty_metrics
from .model import SpectraRet
from .samx import (band_integrated_gradients, band_subset_scores, deletion_auc, deletion_ratios,
                   random_deletion_ratios, samx)
from .train import set_seed
from .uncertainty import SNGPStyleHead, fit_head, fit_temperature, mc_dropout_predict


def run_demo(cfg: Config, preset_name: str = "final"):
    set_seed(0)
    print("1) LSD exactness, float64")
    x64 = torch.rand(2, 3, 128, 128, dtype=torch.float64) * 2 - 1
    lsd = LSD(4, 9, learnable=False).double()
    print(f"   fixed (U = 0):      max |sum_b f_b - x| = {(lsd(x64).sum(1) - x64).abs().max():.2e}")
    lsd_l = LSD(4, 9, learnable=True).double()
    with torch.no_grad():
        for st in lsd_l.steps:
            st.update.weight.normal_(0, 0.05)
    print(f"   learnable (U != 0): max |sum_b f_b - x| = {(lsd_l(x64).sum(1) - x64).abs().max():.2e}")

    print(f"2) SpectraRet, preset '{preset_name}' (ImageNet weights: {cfg.pretrained})")
    model = SpectraRet(cfg)
    if model.seg is not None:
        s_total, _ = count_params(model.seg)
        print(f"   Stage 1 segmentation network: {s_total / 1e6:.2f} M parameters (fixed in Stage 2)")
    total, trained = count_params(model)
    seg_n = count_params(model.seg)[0] if model.seg is not None else 0
    print(f"   Stage 2 (without Stage 1): {(total - seg_n) / 1e6:.2f} M parameters, {trained / 1e6:.2f} M trained")
    b = 2
    x = torch.rand(b, 3, cfg.image_size, cfg.image_size) * 2 - 1
    y = torch.tensor([[2, 0, 0], [0, 1, 2]])
    m = torch.tensor([[True, False, False], [False, True, True]])
    model.train()
    out = model(x)
    loss = spectraret_loss(out, y, m, cfg)
    loss.backward()
    print(f"   segmented image {tuple(out.segmented.shape)}, bands {tuple(out.bands.shape)}, "
          f"e {tuple(out.embedding.shape)}, w {tuple(out.routing.shape)}")
    print(f"   logits {[tuple(lg.shape) for lg in out.logits]}, loss {float(loss):.4f}")
    grads = {n: p.grad is not None for n, p in model.named_parameters() if p.requires_grad}
    print(f"   trainable tensors with a gradient: {sum(grads.values())} / {len(grads)}")

    print("3) SAM-X, band IG and band deletion (disease DR)")
    for form in ("dev", "final"):
        r = samx(model, x, form)
        print(f"   SAM-X {form:5s}: joint {tuple(r['joint'].shape)}, band profile image 0 DR = "
              f"{[round(v, 4) for v in r['profile'][0, 0].tolist()]}")
    ig = band_integrated_gradients(model, x, 0, steps=4)
    print(f"   band IG (4 steps): profile image 0 = {[round(v, 4) for v in ig['profile'][0].tolist()]}")
    sc = band_subset_scores(model, x, 0)
    rank = samx(model, x, "final")["profile"][:, 0].argsort(1, descending=True)
    print(f"   deletion AUC: SAM-X order {deletion_auc(deletion_ratios(sc, rank))[0]:.4f}, "
          f"random reference {deletion_auc(random_deletion_ratios(sc, b, cfg.n_bands))[0]:.4f}")

    print("4) Preset 'full' (CSF + learned DSR)")
    full = SpectraRet(Config(**{**asdict(cfg), **PRESETS["full"]}))
    full.train()
    o = full(x)
    print(f"   routing weights w {tuple(o.routing.shape)}, rows sum to "
          f"{o.routing.sum(-1).mean():.3f}, loss {float(spectraret_loss(o, y, m, full.cfg)):.4f}")

    print("5) Uncertainty heads on synthetic embeddings (4 classes)")
    g = torch.Generator().manual_seed(0)
    centres = torch.randn(4, cfg.dim, generator=g)
    yy = torch.randint(0, 4, (400,), generator=g)
    ee = centres[yy] + 12.0 * torch.randn(400, cfg.dim, generator=g)
    tr, va, te = slice(0, 240), slice(240, 320), slice(320, 400)
    det = fit_head(GradingHead(cfg.dim, 4), ee[tr], yy[tr])
    with torch.no_grad():
        lg_va, lg_te = det(ee[va]), det(ee[te])
    t = fit_temperature(lg_va, yy[va])
    sn = SNGPStyleHead(cfg.dim, 4, 512)
    sn.adapt(ee[tr])
    fit_head(sn, ee[tr], yy[tr])
    sn.fit_laplace(ee[tr])
    heads = {"deterministic": torch.softmax(lg_te, 1), "temperature": torch.softmax(lg_te / t, 1),
             "mc_dropout": mc_dropout_predict(det, ee[te], 30), "sngp": sn.predict(ee[te])}
    for name, p in heads.items():
        r = uncertainty_metrics(yy[te].numpy(), p.numpy(), 4)
        print(f"   {name:13s} acc {r['accuracy']:.3f}  ECE {r['ece']:.3f}  Brier {r['brier']:.3f}  "
              f"NLL {r['nll']:.3f}  AURC {r['aurc_1-maxp']:.3f}")
    print("demo finished")
