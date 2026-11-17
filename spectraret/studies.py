"""Post-training studies on a Stage 2 checkpoint (the final-model protocol of the paper).

run_explain:      SAM-X (final form) and band IG band profiles on held-out test images;
                  band-deletion AUC for both rankings, with zeroing and training-mean
                  replacement, against the exact random reference over all 24 orders.
run_uncertainty:  deterministic, temperature, MC-dropout and SNGP-style heads on the same
                  frozen evidence vector e_d of one disease.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .config import BAND_NAMES, DISEASES, N_CLASSES
from .data import GradeDataset, loader, read_csv, row_grades, split_rows
from .heads import GradingHead
from .metrics import uncertainty_metrics
from .samx import (band_integrated_gradients, band_subset_scores, deletion_auc, deletion_ratios,
                   random_deletion_ratios, samx)
from .train import load_model, predict
from .uncertainty import SNGPStyleHead, fit_head, fit_temperature, mc_dropout_predict


@torch.no_grad()
def mean_train_bands(model, rows, cfg, device, n_max: int = 200):
    """Training-mean band images (B, C, H, W), used as the deletion replacement control."""
    dl = loader(GradeDataset(rows[:n_max], cfg.image_size), 16, False)
    acc, n = None, 0
    for x, _, _ in dl:
        x_seg, _ = model.segment(x.to(device))
        f = model.bands(x_seg).sum(0)
        acc = f if acc is None else acc + f
        n += len(x)
    return acc / max(n, 1)


def run_explain(args):
    device = torch.device(args.device)
    model, cfg = load_model(args.ckpt, device)
    rows = read_csv(args.csv)
    test, train = split_rows(rows, "test"), split_rows(rows, "train")
    mean_bands = mean_train_bands(model, train, cfg, device) if train else None
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary, arrays = {}, {}
    for d, name in enumerate(DISEASES):
        sel = [r for r in test if row_grades(r)[d] is not None][:args.n]
        if not sel:
            continue
        dl = loader(GradeDataset(sel, cfg.image_size), args.batch, False)
        prof, spatial, ig_prof = [], [], []
        ratios = {"samx_zero": [], "ig_zero": [], "random_zero": [], "samx_mean": [], "random_mean": []}
        for x, _, _ in dl:
            x = x.to(device)
            sx = samx(model, x, "final")
            sx_d = sx["profile"][:, d].cpu()
            ig = band_integrated_gradients(model, x, d, args.ig_steps)["profile"].cpu()
            prof.append(sx_d)
            ig_prof.append(ig)
            spatial.append(sx["spatial"][:, d].cpu())
            rank_sx, rank_ig = sx_d.argsort(1, descending=True), ig.argsort(1, descending=True)
            sc = band_subset_scores(model, x, d)
            ratios["samx_zero"].append(deletion_ratios(sc, rank_sx))
            ratios["ig_zero"].append(deletion_ratios(sc, rank_ig))
            ratios["random_zero"].append(random_deletion_ratios(sc, len(x), cfg.n_bands))
            if mean_bands is not None:
                scm = band_subset_scores(model, x, d, mean_bands)
                ratios["samx_mean"].append(deletion_ratios(scm, rank_sx))
                ratios["random_mean"].append(random_deletion_ratios(scm, len(x), cfg.n_bands))
        prof = torch.cat(prof)
        res = {"n": len(prof),
               "samx_band_profile_mean": dict(zip(BAND_NAMES, prof.mean(0).tolist())),
               "samx_top_band_counts": dict(zip(BAND_NAMES, np.bincount(prof.argmax(1).numpy(),
                                                                         minlength=cfg.n_bands).tolist()))}
        for key, val in ratios.items():
            if val:
                auc, curve = deletion_auc(torch.cat(val))
                res[f"deletion_auc_{key}"] = auc
                res[f"deletion_curve_{key}"] = curve
        summary[name] = res
        arrays[f"{name}_samx_profile"] = prof.numpy()
        arrays[f"{name}_ig_profile"] = torch.cat(ig_prof).numpy()
        arrays[f"{name}_samx_spatial"] = torch.cat(spatial).numpy()
        print(name, json.dumps({k: v for k, v in res.items() if not k.startswith("deletion_curve")}, indent=1))
    (out_dir / "xai_summary.json").write_text(json.dumps(summary, indent=1))
    np.savez_compressed(out_dir / "xai_maps.npz", **arrays)


@torch.no_grad()
def disease_embeddings(model, rows, d, cfg, device, batch):
    """e_d, grading logits and grades of the rows labelled for disease d."""
    sel = [r for r in rows if row_grades(r)[d] is not None]
    logits, y, _, e = predict(model, loader(GradeDataset(sel, cfg.image_size), batch, False), device)
    return e[:, d], logits[d], y[:, d]


def run_uncertainty(args):
    device = torch.device(args.device)
    model, cfg = load_model(args.ckpt, device)
    d = DISEASES.index(args.disease)
    k = N_CLASSES[d]
    rows = read_csv(args.csv)
    e_tr, _, y_tr = disease_embeddings(model, split_rows(rows, "train"), d, cfg, device, args.batch)
    e_va, lg_va, y_va = disease_embeddings(model, split_rows(rows, "val"), d, cfg, device, args.batch)
    e_te, lg_te, y_te = disease_embeddings(model, split_rows(rows, "test"), d, cfg, device, args.batch)
    torch.manual_seed(args.seed)

    probs = {"deterministic": torch.softmax(lg_te, 1)}
    t = fit_temperature(lg_va, y_va)
    probs["temperature"] = torch.softmax(lg_te / t, 1)
    mc_head = fit_head(GradingHead(cfg.dim, k, 128, 0.3), e_tr, y_tr, seed=args.seed)
    probs["mc_dropout"] = mc_dropout_predict(mc_head, e_te, args.passes)
    sngp = SNGPStyleHead(cfg.dim, k, args.n_rff, args.seed)
    sngp.adapt(e_tr, seed=args.seed)
    fit_head(sngp, e_tr, y_tr, seed=args.seed)
    sngp.fit_laplace(e_tr)
    probs["sngp"] = sngp.predict(e_te)

    y = y_te.numpy()
    report = {"disease": args.disease, "n_train": len(e_tr), "n_test": len(e_te), "temperature": t,
              "heads": {name: uncertainty_metrics(y, p.numpy(), k) for name, p in probs.items()}}
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(json.dumps(report, indent=1))
    keys = {"deterministic": "det", "temperature": "temp", "mc_dropout": "mc", "sngp": "sngp"}
    np.savez_compressed(out_dir / "samples.npz", y=y,
                        **{f"{keys[n]}_probs": p.numpy() for n, p in probs.items()},
                        **{f"{keys[n]}_unc": 1 - p.numpy().max(1) for n, p in probs.items()})
    print(json.dumps(report, indent=1))
