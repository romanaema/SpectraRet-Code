"""Training (Table 7 of the paper) and evaluation.

Stage 1: modified DeepLabV3+, Adam lr 1e-3, batch 16, up to 25 epochs, early stopping on
         validation mean Dice (patience 8, min gain 0.003).
Stage 2: Stage 1 fixed. SwinECAT backbone AdamW lr 3e-5, weight decay 1e-4, cosine schedule;
         all other trained layers Adam lr 1e-3, constant; batch 32; up to 25 epochs; early
         stopping on the 3-epoch moving average of validation macro-F1 over the three
         diseases (patience 8, min gain 0.003).
"""
from __future__ import annotations

import json
import random
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from .config import DISEASES, N_CLASSES, SEG_CHANNELS, Config, count_params
from .data import GradeDataset, SegDataset, loader, read_csv, split_rows
from .losses import masked_seg_loss, spectraret_loss
from .metrics import ece, macro_f1, mcc, qwk
from .model import SpectraRet
from .segmentation import ModifiedDeepLabV3Plus


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_model(path, device="cpu"):
    """Rebuild a Stage 2 checkpoint (it includes the Stage 1 weights)."""
    ck = torch.load(path, map_location="cpu", weights_only=True)
    cfg = Config(**{**ck["cfg"], "pretrained": False})
    model = SpectraRet(cfg)
    model.load_state_dict(ck["state"])
    return model.to(device).eval(), cfg


@torch.no_grad()
def predict(model, dl, device):
    """-> per-disease logits, grades (N, 3), label mask (N, 3), embeddings e (N, 3, dim)."""
    model.eval()
    logits = [[] for _ in DISEASES]
    ys, ms, es = [], [], []
    for x, y, m in dl:
        out = model(x.to(device))
        for d in range(len(DISEASES)):
            logits[d].append(out.logits[d].cpu())
        ys.append(y)
        ms.append(m)
        es.append(out.embedding.cpu())
    return [torch.cat(lg) for lg in logits], torch.cat(ys), torch.cat(ms), torch.cat(es)


def grading_metrics(logits, y, m):
    res = {}
    for d, (name, k) in enumerate(zip(DISEASES, N_CLASSES)):
        sel = m[:, d]
        if not sel.any():
            continue
        probs = torch.softmax(logits[d][sel], 1).numpy()
        yt, pr = y[sel, d].numpy(), probs.argmax(1)
        res[name] = {"n": int(sel.sum()), "qwk": qwk(yt, pr, k), "macro_f1": macro_f1(yt, pr, k),
                     "mcc": mcc(yt, pr, k), "accuracy": float((pr == yt).mean()), "ece": ece(yt, probs)}
    res["mean_macro_f1"] = float(np.mean([v["macro_f1"] for v in res.values()])) if res else 0.0
    return res


# ---------------------------------------------------------------------------------------
# Stage 1
# ---------------------------------------------------------------------------------------
@torch.no_grad()
def seg_dice(net, dl, device):
    net.eval()
    c = len(SEG_CHANNELS)
    inter, denom, seen = torch.zeros(c), torch.zeros(c), torch.zeros(c)
    for x, t, s in dl:
        p = (torch.sigmoid(net(x.to(device))) >= 0.5).float().cpu()
        mm = s[:, :, None, None]
        inter += (p * t * mm).sum(dim=(0, 2, 3))
        denom += ((p + t) * mm).sum(dim=(0, 2, 3))
        seen += s.sum(0)
    dice = (2 * inter + 1) / (denom + 1)
    return float(dice[seen > 0].mean()) if (seen > 0).any() else 0.0


def train_stage1(args):
    set_seed(args.seed)
    device = torch.device(args.device)
    rows = read_csv(args.csv)
    tr = loader(SegDataset(split_rows(rows, "train"), args.image_size, True), args.batch, True, args.workers, True)
    va = loader(SegDataset(split_rows(rows, "val"), args.image_size), args.batch, False, args.workers)
    net = ModifiedDeepLabV3Plus(pretrained=not args.no_pretrained).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    best, wait, log = -1.0, 0, []
    for ep in range(1, args.epochs + 1):
        net.train()
        t0, tot = time.time(), 0.0
        for x, t, s in tr:
            loss = masked_seg_loss(net(x.to(device)), t.to(device), s.to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss) * len(x)
        dice = seg_dice(net, va, device)
        log.append({"epoch": ep, "loss": tot / max(len(tr.dataset), 1), "val_dice": dice, "sec": time.time() - t0})
        print(json.dumps(log[-1]), flush=True)
        if dice > best + args.min_gain:
            best, wait = dice, 0
            torch.save({"state": net.state_dict(), "epoch": ep, "val_dice": dice}, out_dir / "stage1_best.pt")
        else:
            wait += 1
            if wait >= args.patience:
                break
    (out_dir / "stage1_log.json").write_text(json.dumps(log, indent=1))
    print(f"best validation Dice {best:.4f} -> {out_dir / 'stage1_best.pt'}")


# ---------------------------------------------------------------------------------------
# Stage 2
# ---------------------------------------------------------------------------------------
def train_stage2(args, cfg: Config):
    set_seed(args.seed)
    device = torch.device(args.device)
    rows = read_csv(args.csv)
    tr = loader(GradeDataset(split_rows(rows, "train"), cfg.image_size, True), args.batch, True, args.workers)
    va = loader(GradeDataset(split_rows(rows, "val"), cfg.image_size), args.batch, False, args.workers)
    te_rows = split_rows(rows, "test")
    model = SpectraRet(cfg)
    if cfg.use_seg:
        if not args.seg_ckpt:
            raise SystemExit("Stage 2 needs the Stage 1 checkpoint (--seg-ckpt), or use --no-seg.")
        model.load_stage1(args.seg_ckpt)
    model.to(device)
    total, trained = count_params(model)
    print(f"parameters: {total / 1e6:.2f} M total, {trained / 1e6:.2f} M trained", flush=True)

    swin = list(model.tok.swin.backbone_parameters()) if model.tok.swin is not None else []
    swin_ids = {id(p) for p in swin}
    other = [p for p in model.parameters() if p.requires_grad and id(p) not in swin_ids]
    opts = [torch.optim.Adam(other, lr=1e-3)]
    scheds = []
    if swin:
        opt_swin = torch.optim.AdamW(swin, lr=3e-5, weight_decay=1e-4)
        opts.append(opt_swin)
        scheds.append(torch.optim.lr_scheduler.CosineAnnealingLR(opt_swin, T_max=max(args.epochs * len(tr), 1)))

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    best, wait, hist, log = -1.0, 0, [], []
    for ep in range(1, args.epochs + 1):
        model.train()
        t0, tot, n = time.time(), 0.0, 0
        for x, y, m in tr:
            out = model(x.to(device))
            loss = spectraret_loss(out, y.to(device), m.to(device), cfg)
            for o in opts:
                o.zero_grad()
            loss.backward()
            for o in opts:
                o.step()
            for s in scheds:
                s.step()
            tot, n = tot + float(loss) * len(x), n + len(x)
        val = grading_metrics(*predict(model, va, device)[:3])
        hist.append(val["mean_macro_f1"])
        ma = float(np.mean(hist[-3:]))                   # 3-epoch moving average
        log.append({"epoch": ep, "loss": tot / max(n, 1), "val_macro_f1": hist[-1], "val_f1_ma3": ma,
                    "sec": time.time() - t0})
        print(json.dumps(log[-1]), flush=True)
        if ma > best + args.min_gain:
            best, wait = ma, 0
            torch.save({"cfg": asdict(cfg), "state": model.state_dict(), "epoch": ep, "val_f1_ma3": ma},
                       out_dir / "best.pt")
        else:
            wait += 1
            if wait >= args.patience:
                break
    (out_dir / "train_log.json").write_text(json.dumps(log, indent=1))

    model, _ = load_model(out_dir / "best.pt", device)
    report = {"config": asdict(cfg), "best_val_f1_ma3": best}
    if te_rows:
        te = loader(GradeDataset(te_rows, cfg.image_size), args.batch, False, args.workers)
        report["test"] = grading_metrics(*predict(model, te, device)[:3])
        print(json.dumps(report["test"], indent=1))
    (out_dir / "report.json").write_text(json.dumps(report, indent=1))
