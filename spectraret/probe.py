"""Section 3.2 - linear probe on frozen MobileNetV2 features (learnability screening).

Used before model comparison to check whether a grading folder carries a learnable signal
(APTOS-2019, FGADR and REFUGE folders: probe QWK -0.02, 0.03, 0.06 -> excluded) and to screen
the BMRD_DR centres (575 images, QWK 0.06 -> excluded from development; 336 images, 0.33).
A low probe score does not by itself show poor images or wrong labels.

Features: ImageNet MobileNetV2 (frozen, eval mode), 224 x 224, ImageNet normalization,
global average pooling (1280-d). Probe: standardized features + multinomial logistic
regression, four folds grouped by source-image file stem; QWK of the out-of-fold predictions.

Input CSV columns: image, label (integer grade), optional group (defaults to the file stem).
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch

from .metrics import qwk

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@torch.no_grad()
def mobilenet_features(paths, size: int = 224, batch: int = 32, device: str = "cpu", pretrained: bool = True):
    from PIL import Image

    from .backbones import mobilenet_v2
    net = mobilenet_v2(pretrained).features.to(device).eval()
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    feats = []
    for s in range(0, len(paths), batch):
        imgs = [np.asarray(Image.open(p).convert("RGB").resize((size, size), Image.BILINEAR), np.float32) / 255.0
                for p in paths[s:s + batch]]
        x = (torch.from_numpy(np.stack(imgs)).permute(0, 3, 1, 2) - mean) / std
        feats.append(net(x.to(device)).mean(dim=(2, 3)).cpu())
    return torch.cat(feats).numpy()


def grouped_folds(labels, groups, n_folds: int = 4, seed: int = 0):
    from sklearn.model_selection import StratifiedGroupKFold
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    return list(sgkf.split(np.zeros(len(labels)), labels, groups))


def linear_probe(features, labels, groups, n_folds: int = 4, seed: int = 0, C: float = 1.0):
    """Out-of-fold predictions of a multinomial logistic-regression probe; QWK per fold."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    labels = np.asarray(labels)
    k = int(labels.max()) + 1
    pred = np.full(len(labels), -1)
    per_fold = []
    for tr, te in grouped_folds(labels, groups, n_folds, seed):
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000))
        clf.fit(features[tr], labels[tr])
        pred[te] = clf.predict(features[te])
        per_fold.append(qwk(labels[te], pred[te], k))
    return {"qwk_oof": qwk(labels, pred, k), "qwk_folds": per_fold,
            "qwk_mean": float(np.mean(per_fold)), "qwk_sd": float(np.std(per_fold)),
            "class_counts": np.bincount(labels, minlength=k).tolist(), "pred": pred.tolist()}


def run_probe(csv_path, out_dir, n_folds: int = 4, seed: int = 0, device: str = "cpu"):
    with open(csv_path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    root = Path(csv_path).resolve().parent
    paths = [str(Path(r["image"]) if Path(r["image"]).is_absolute() else root / r["image"]) for r in rows]
    labels = np.array([int(float(r["label"])) for r in rows])
    groups = [r.get("group") or Path(p).stem.lower() for r, p in zip(rows, paths)]
    feats = mobilenet_features(paths, device=device)
    res = linear_probe(feats, labels, groups, n_folds, seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "probe_report.json").write_text(json.dumps({k: v for k, v in res.items() if k != "pred"}, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "pred"}, indent=1))
    return res
