"""CSV datasets for Stage 1 (segmentation) and Stage 2 (grading), and the augmentation.

Paths in the CSV are relative to the CSV file or absolute.

Grading CSV:        image, split, DR, Glaucoma, AMD
    split  = train | val | test
    grades = 0..3 for DR and Glaucoma, 0..2 for AMD; empty (or -1) = no label for that disease
Segmentation CSV:   image, split, mask, group
    mask  = 16-bit PNG, bit c set where channel c is present (config.SEG_CHANNELS)
    group = DR | AMD | Glaucoma - the channels the annotator marked; the other channels of
            that image are unknown, not absent, and are left out of the loss.

Augmentation (training only): H/V flips (joint with the mask), brightness +-12,
contrast [0.9, 1.1]. Images are resized to 128 x 128 and scaled to [-1, 1].
"""
from __future__ import annotations

import csv
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .config import DISEASES, N_CLASSES, SEG_CHANNELS, SEG_GROUPS


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    root = Path(path).resolve().parent
    for r in rows:
        for key in ("image", "mask"):
            if r.get(key):
                p = Path(r[key])
                r[key] = str(p if p.is_absolute() else root / p)
    return rows


def split_rows(rows, split):
    return [r for r in rows if r.get("split", "").strip().lower() == split]


def parse_grade(value, k):
    v = (value or "").strip()
    if v.lower() in ("", "-1", "nan", "na", "none"):
        return None
    g = int(float(v))
    if not 0 <= g < k:
        raise ValueError(f"grade {g} outside 0..{k - 1}")
    return g


def row_grades(row):
    """[DR, Glaucoma, AMD] grades of one CSV row, None where unlabelled."""
    return [parse_grade(row.get(name, row.get(name.lower(), "")), k) for name, k in zip(DISEASES, N_CLASSES)]


def load_rgb(path, size):
    from PIL import Image
    img = Image.open(path).convert("RGB").resize((size, size), Image.BOX)
    return np.asarray(img, dtype=np.float32)


def augment(img, mask=None):
    """img in 0..255, HWC; mask CHW."""
    if random.random() < 0.5:
        img = img[:, ::-1]
        mask = None if mask is None else mask[:, :, ::-1]
    if random.random() < 0.5:
        img = img[::-1]
        mask = None if mask is None else mask[:, ::-1]
    img = img + random.uniform(-12, 12)
    mean = img.mean(axis=(0, 1), keepdims=True)
    img = (img - mean) * random.uniform(0.9, 1.1) + mean
    return np.clip(img, 0, 255), mask


def to_tensor(img):
    return torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1)) / 127.5 - 1.0).float()


class GradeDataset(torch.utils.data.Dataset):
    """-> image (3, H, W) in [-1, 1], grades y (3,), label mask m (3,) bool."""

    def __init__(self, rows, size: int = 128, train: bool = False):
        self.rows, self.size, self.train = rows, size, train

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        img = load_rgb(r["image"], self.size)
        if self.train:
            img, _ = augment(img)
        g = row_grades(r)
        y = torch.tensor([0 if v is None else v for v in g])
        m = torch.tensor([v is not None for v in g])
        return to_tensor(img), y, m


class SegDataset(torch.utils.data.Dataset):
    """-> image (3, H, W) in [-1, 1], target (11, H, W) in {0, 1}, supervised channels (11,)."""

    def __init__(self, rows, size: int = 128, train: bool = False):
        self.rows, self.size, self.train = rows, size, train

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        from PIL import Image
        r = self.rows[i]
        img = load_rgb(r["image"], self.size)
        bits = np.asarray(Image.open(r["mask"])).astype(np.int64)
        chans = ((bits[None] >> np.arange(len(SEG_CHANNELS))[:, None, None]) & 1).astype(np.float32)
        chans = F.interpolate(torch.from_numpy(chans)[None], size=(self.size, self.size), mode="nearest")[0].numpy()
        if self.train:
            img, chans = augment(img, chans)
        sup = torch.zeros(len(SEG_CHANNELS))
        sup[list(SEG_GROUPS[r["group"]])] = 1.0
        return to_tensor(img), torch.from_numpy(np.ascontiguousarray(chans)), sup


def loader(ds, batch, shuffle, workers=0, drop_last=False):
    return torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=shuffle, num_workers=workers,
                                       drop_last=drop_last)
