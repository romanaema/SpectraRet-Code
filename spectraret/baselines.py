"""Section 6.1 / Tables 7, 9, 11 - comparison baselines and the Gate P1 network.

Frozen ImageNet CNNs (one model per disease): VGG16, ResNet50, ResNet101, InceptionV3,
InceptionResNetV2, Xception, DenseNet121, MobileNetV2, EfficientNetB0, ConvNeXt-Tiny as frozen
feature extractors (224 x 224, global average pooling) with a two-layer head
(256 units, dropout 0.4), Adam lr 1e-3, batch 32, up to 40 epochs, early stopping on validation
QWK (patience 8). The head is trained on cached frozen features, so no augmentation.
InceptionResNetV2 and Xception come from timm (optional dependency).

The matched references of the final comparison - standalone SwinECAT on the raw image and
the raw-image dual-stream control - are presets of SpectraRet ("swinecat_only", "raw_control").

Gate P1 (Table 11): a small four-layer CNN trained from scratch at 160 x 160 on the raw image,
on fixed db4 bands, or on learnable LSD bands (three folds, one seed).
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import DISEASES, N_CLASSES
from .lsd import LSD
from .metrics import qwk

IMAGENET_MEAN = torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
BACKBONES = ("VGG16", "ResNet50", "ResNet101", "InceptionV3", "InceptionResNetV2", "Xception",
             "DenseNet121", "MobileNetV2", "EfficientNetB0", "ConvNeXt-Tiny")


class _Pooled(nn.Module):
    def __init__(self, body: nn.Module, post: nn.Module | None = None):
        super().__init__()
        self.body, self.post = body, post

    def forward(self, x):
        h = self.body(x)
        if h.dim() == 4:
            h = h.mean(dim=(2, 3))
        return self.post(h) if self.post is not None else h


def frozen_backbone(name: str, pretrained: bool = True):
    """-> (module: image in [0, 1] ImageNet-normalised -> pooled features, feature dim)."""
    import torchvision.models as tvm
    W = "DEFAULT" if pretrained else None
    if name == "VGG16":
        return _Pooled(tvm.vgg16(weights=W).features), 512
    if name == "ResNet50":
        m = tvm.resnet50(weights=W)
        m.fc = nn.Identity()
        return m, 2048
    if name == "ResNet101":
        m = tvm.resnet101(weights=W)
        m.fc = nn.Identity()
        return m, 2048
    if name == "InceptionV3":
        m = tvm.inception_v3(weights=W, aux_logits=True, init_weights=not pretrained)
        m.aux_logits, m.AuxLogits, m.fc = False, None, nn.Identity()
        return m, 2048
    if name == "DenseNet121":
        m = tvm.densenet121(weights=W)
        return _Pooled(nn.Sequential(m.features, nn.ReLU())), 1024
    if name == "MobileNetV2":
        return _Pooled(tvm.mobilenet_v2(weights=W).features), 1280
    if name == "EfficientNetB0":
        return _Pooled(tvm.efficientnet_b0(weights=W).features), 1280
    if name == "ConvNeXt-Tiny":
        m = tvm.convnext_tiny(weights=W)
        return _Pooled(m.features, nn.LayerNorm(768, eps=1e-6)), 768
    if name in ("InceptionResNetV2", "Xception"):
        import timm
        m = timm.create_model({"InceptionResNetV2": "inception_resnet_v2", "Xception": "legacy_xception"}[name],
                              pretrained=pretrained, num_classes=0)
        return m, m.num_features
    raise ValueError(f"unknown backbone {name}")


@torch.no_grad()
def extract_features(name, rows, size: int = 224, batch: int = 32, device: str = "cpu", pretrained: bool = True):
    """Cached frozen features (N, dim) of the rows' images."""
    from .data import load_rgb
    net, _ = frozen_backbone(name, pretrained)
    net = net.to(device).eval()
    out = []
    for s in range(0, len(rows), batch):
        imgs = np.stack([load_rgb(r["image"], size) for r in rows[s:s + batch]]) / 255.0
        x = (torch.from_numpy(imgs).permute(0, 3, 1, 2).float() - IMAGENET_MEAN) / IMAGENET_STD
        out.append(net(x.to(device)).cpu())
    return torch.cat(out)


class BaselineHead(nn.Sequential):
    """2-layer MLP (256, dropout 0.4) on frozen features."""

    def __init__(self, dim: int, n_classes: int):
        super().__init__(nn.Linear(dim, 256), nn.ReLU(), nn.Dropout(0.4), nn.Linear(256, n_classes))


def train_head_on_features(f_tr, y_tr, f_va, y_va, k, epochs: int = 40, patience: int = 8,
                           lr: float = 1e-3, batch: int = 32, seed: int = 0):
    """Adam on cached features; best validation QWK with patience 8."""
    torch.manual_seed(seed)
    head = BaselineHead(f_tr.shape[1], k)
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    best, best_state, wait = -2.0, None, 0
    g = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        head.train()
        perm = torch.randperm(len(f_tr), generator=g)
        for s in range(0, len(f_tr), batch):
            idx = perm[s:s + batch]
            loss = F.cross_entropy(head(f_tr[idx]), y_tr[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        head.eval()
        with torch.no_grad():
            score = qwk(y_va.numpy(), head(f_va).argmax(1).numpy(), k)
        if score > best:
            best, best_state, wait = score, copy.deepcopy(head.state_dict()), 0
        else:
            wait += 1
            if wait >= patience:
                break
    head.load_state_dict(best_state)
    return head.eval(), best


def run_frozen_baselines(args):
    """python main.py baselines --csv fold_0.csv --out runs/baselines_f0 [--backbones ...] [--seeds 0 1 2]"""
    from .data import read_csv, row_grades, split_rows
    from .metrics import grading_report
    rows = read_csv(args.csv)
    parts = {s: split_rows(rows, s) for s in ("train", "val", "test")}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report = {}
    for name in args.backbones:
        feats = {s: extract_features(name, rs, device=args.device) for s, rs in parts.items()}
        report[name] = {}
        for d, (dis, k) in enumerate(zip(DISEASES, N_CLASSES)):        # one model per disease
            sel = {s: [i for i, r in enumerate(rs) if row_grades(r)[d] is not None] for s, rs in parts.items()}
            if not sel["train"] or not sel["val"] or not sel["test"]:
                continue
            ys = {s: torch.tensor([row_grades(parts[s][i])[d] for i in sel[s]]) for s in sel}
            runs = []
            for seed in args.seeds:
                head, _ = train_head_on_features(feats["train"][sel["train"]], ys["train"],
                                                 feats["val"][sel["val"]], ys["val"], k, seed=seed)
                with torch.no_grad():
                    probs = torch.softmax(head(feats["test"][sel["test"]]), 1).numpy()
                runs.append(grading_report(ys["test"].numpy(), probs, k))
            report[name][dis] = {"runs": runs, "qwk_mean": float(np.mean([r["qwk"] for r in runs])),
                                 "qwk_sd": float(np.std([r["qwk"] for r in runs]))}
            print(name, dis, f"QWK {report[name][dis]['qwk_mean']:.3f} +- {report[name][dis]['qwk_sd']:.3f}", flush=True)
    (out / "baselines_report.json").write_text(json.dumps(report, indent=1))


# ----------------------------------------------------------------------------- Gate P1
class SmallCNN(nn.Module):
    """Four conv blocks (32-64-128-256, 3x3, BN, ReLU, max-pool), GAP, one head per disease.
    input: raw | fixed | learnable  (raw image, or the four LSD bands stacked as 12 channels)."""

    def __init__(self, input_mode: str = "raw", size: int = 160, widths=(32, 64, 128, 256)):
        super().__init__()
        if input_mode not in ("raw", "fixed", "learnable"):
            raise ValueError("input_mode must be raw, fixed or learnable")
        self.input_mode, self.size = input_mode, size
        self.lsd = None if input_mode == "raw" else LSD(4, 9, learnable=input_mode == "learnable")
        cin = 3 if input_mode == "raw" else 12
        layers = []
        for w in widths:
            layers += [nn.Conv2d(cin, w, 3, padding=1, bias=False), nn.BatchNorm2d(w), nn.ReLU(inplace=True),
                       nn.MaxPool2d(2)]
            cin = w
        self.features = nn.Sequential(*layers)
        self.heads = nn.ModuleList([nn.Linear(cin, k) for k in N_CLASSES])

    def forward(self, x):
        if x.shape[-1] != self.size:
            x = F.interpolate(x, size=(self.size, self.size), mode="bilinear", align_corners=False)
        if self.lsd is not None:
            x = self.lsd(x).flatten(1, 2)                 # (b, 4 * 3, H, W)
        h = self.features(x).mean(dim=(2, 3))
        return [head(h) for head in self.heads]


def train_gate_p1(args):
    """python main.py gate-p1 --csv fold_0.csv --input fixed --out runs/p1_fixed
    Masked cross-entropy over the three diseases, Adam lr 1e-3, early stopping on the mean
    validation QWK; reports test QWK per disease."""
    from .data import GradeDataset, loader, read_csv, split_rows
    from .losses import masked_ce
    from .train import set_seed
    set_seed(args.seed)
    device = torch.device(args.device)
    rows = read_csv(args.csv)
    tr = loader(GradeDataset(split_rows(rows, "train"), args.image_size, True), args.batch, True, args.workers)
    va = loader(GradeDataset(split_rows(rows, "val"), args.image_size), args.batch, False, args.workers)
    te = loader(GradeDataset(split_rows(rows, "test"), args.image_size), args.batch, False, args.workers)
    model = SmallCNN(args.input, args.image_size).to(device)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-3)

    @torch.no_grad()
    def scores(dl):
        model.eval()
        lg, ys, ms = [[] for _ in DISEASES], [], []
        for x, y, m in dl:
            out = model(x.to(device))
            for d in range(len(DISEASES)):
                lg[d].append(out[d].argmax(1).cpu())
            ys.append(y)
            ms.append(m)
        y, m = torch.cat(ys), torch.cat(ms)
        return {n: qwk(y[m[:, d], d].numpy(), torch.cat(lg[d])[m[:, d]].numpy(), k)
                for d, (n, k) in enumerate(zip(DISEASES, N_CLASSES)) if m[:, d].any()}

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    best, wait, best_state = -2.0, 0, None
    for ep in range(1, args.epochs + 1):
        model.train()
        for x, y, m in tr:
            loss = masked_ce(model(x.to(device)), y.to(device), m.to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
        val = scores(va)
        mean = float(np.mean(list(val.values()))) if val else -1.0
        print(json.dumps({"epoch": ep, "val_qwk": val}), flush=True)
        if mean > best + args.min_gain:
            best, wait, best_state = mean, 0, copy.deepcopy(model.state_dict())
        else:
            wait += 1
            if wait >= args.patience:
                break
    model.load_state_dict(best_state)
    report = {"input": args.input, "best_val_mean_qwk": best, "test_qwk": scores(te)}
    (out / "gate_p1_report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
