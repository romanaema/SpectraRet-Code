"""Sections 3.2-3.3 / Table 4 - label encoding, grouped folds and the split check.

Label encoding (training codes 0..3 = categories 1..4 of the paper):
  BMRD DR / glaucoma   Level 1..4                       -> 0..3
  Messidor             grade 0, 1, 2, 3                 -> 0, 1, 2, 3   (ordinal encoding only)
  IDRiD (ICDR)         0, 1, 2, 3, 4                    -> 0, 1, 1, 2, 3 (original study mapping)
  DRISHTI-GS           mean CDR in [0,.3) [.3,.5) [.5,.7) [.7,1] -> 0..3 (boundaries go up)
  BMRD AMD             no / dry / wet                   -> 0, 1, 2      (categorical)

Splitting: stratified five-fold splitting grouped by source-image file stem; 15% of each
training fold (by group, stratified) is held out for validation and early stopping. The split
check verifies that no file-stem group crosses partitions. It cannot detect duplicates with
different file names or two photographs of the same patient.
"""
from __future__ import annotations

import csv
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

from .config import DISEASES


# ----------------------------------------------------------------------------- encodings
def bmrd_level_to_code(level: int) -> int:
    if not 1 <= int(level) <= 4:
        raise ValueError("BMRD levels are 1..4")
    return int(level) - 1


def messidor_to_code(grade: int) -> int:
    if not 0 <= int(grade) <= 3:
        raise ValueError("Messidor grades are 0..3")
    return int(grade)


IDRID_ICDR_TO_CODE = {0: 0, 1: 1, 2: 1, 3: 2, 4: 3}


def idrid_to_code(icdr: int) -> int:
    return IDRID_ICDR_TO_CODE[int(icdr)]


CDR_EDGES = (0.3, 0.5, 0.7)


def cdr_category(mean_cdr: float) -> int:
    """[0, 0.3) -> 0, [0.3, 0.5) -> 1, [0.5, 0.7) -> 2, [0.7, 1] -> 3; boundary values go to the
    higher category."""
    return int(np.searchsorted(CDR_EDGES, float(mean_cdr), side="right"))


AMD_TO_CODE = {"no": 0, "dry": 1, "wet": 2}


# ----------------------------------------------------------------------------- folds
def file_stem(path: str) -> str:
    return Path(path).stem.lower()


def strat_key(row) -> str:
    """Stratification label: the observed task and its grade (each image has one task label)."""
    for d in DISEASES:
        v = (row.get(d) or "").strip()
        if v not in ("", "-1"):
            return f"{d}:{int(float(v))}"
    return "none"


def make_folds(rows, n_folds: int = 5, val_frac: float = 0.15, seed: int = 0, group_fn=file_stem):
    """-> list of n_folds dicts {"train": idx, "val": idx, "test": idx} (indices into rows)."""
    from sklearn.model_selection import StratifiedGroupKFold
    y = [strat_key(r) for r in rows]
    groups = [group_fn(r["image"]) for r in rows]
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = []
    for f, (trval, test) in enumerate(sgkf.split(np.zeros(len(rows)), y, groups)):
        val = _grouped_stratified_holdout([rows[i] for i in trval], [groups[i] for i in trval],
                                          [y[i] for i in trval], val_frac, seed + f)
        val_idx = trval[val]
        train_idx = np.setdiff1d(trval, val_idx)
        folds.append({"train": train_idx.tolist(), "val": val_idx.tolist(), "test": test.tolist()})
    return folds


def _grouped_stratified_holdout(rows, groups, y, frac, seed):
    """Pick whole groups per stratum until each stratum has ~frac of its images held out."""
    rng = random.Random(seed)
    by_stratum = defaultdict(lambda: defaultdict(list))
    for i, (g, s) in enumerate(zip(groups, y)):
        by_stratum[s][g].append(i)
    chosen = []
    for s, gmap in sorted(by_stratum.items()):
        target = frac * sum(len(v) for v in gmap.values())
        names = sorted(gmap)
        rng.shuffle(names)
        n = 0
        for g in names:
            if n >= target:
                break
            chosen += gmap[g]
            n += len(gmap[g])
    # a group spanning several strata must not be split between train and val
    chosen_groups = {groups[i] for i in chosen}
    return np.array(sorted(i for i, g in enumerate(groups) if g in chosen_groups), dtype=int)


def check_split(rows_by_split: dict, group_fn=file_stem):
    """Raise if a file-stem group occurs in more than one partition; return group counts."""
    seen = {}
    for split, rows in rows_by_split.items():
        for r in rows:
            g = group_fn(r["image"])
            if g in seen and seen[g] != split:
                raise AssertionError(f"group {g!r} is in both {seen[g]} and {split}")
            seen[g] = split
    return {s: len({group_fn(r["image"]) for r in rs}) for s, rs in rows_by_split.items()}


def write_fold_csvs(csv_path, out_dir, n_folds: int = 5, val_frac: float = 0.15, seed: int = 0):
    """Read a grading CSV (image, DR, Glaucoma, AMD, ...) and write fold_<k>.csv files with a
    split column (train | val | test), checking group separation for every fold."""
    with open(csv_path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    root = Path(csv_path).resolve().parent
    fields = [k for k in rows[0].keys() if k != "split"] + ["split"]
    for f, fold in enumerate(make_folds(rows, n_folds, val_frac, seed)):
        parts = {s: [rows[i] for i in idx] for s, idx in fold.items()}
        counts = check_split(parts)
        with open(out / f"fold_{f}.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            for s, rs in parts.items():
                for r in rs:
                    img = Path(r["image"])
                    w.writerow({**r, "image": str(img if img.is_absolute() else root / img), "split": s})
        print(f"fold {f}: " + ", ".join(f"{s} {len(rs)} images / {counts[s]} groups" for s, rs in parts.items()))
