"""Section 4 / Table 5 / Fig. 2 - spectral pre-study (run before any model code).

Per image: resize to 512 x 512, CLAHE on the lightness channel (clip limit 2.0, 8 x 8 tiles),
z-score per channel, convert to luma, four-level db4 DWT (PyWavelets). Band energy is the sum
of squared coefficients of the approximation A and of the detail levels D1 (finest) .. D4
(coarsest); because A holds about 95% of the energy the detail levels are also renormalized
among themselves ("detail-relative" energy, rel.).

Differences are tested with the Kruskal-Wallis test and summarized by the effect size
eta^2 = (H - k + 1) / (n - k):
  * between task-specific collections (DR vs glaucoma vs AMD), all images
  * between collections within each BMRD centre that supplied more than one disease
  * between acquisition sources within each disease (each public dataset and each BMRD
    centre is one source)
Collection membership is the grouping variable; disease status is not verified.

Input CSV columns: image, collection (DR | Glaucoma | AMD), source, bmrd (1 = BMRD image),
centre (BMRD collection centre, empty for public images).
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

BANDS_ABS = ("A", "D1", "D2", "D3", "D4")
BANDS_REL = ("D1 (rel.)", "D2 (rel.)", "D3 (rel.)", "D4 (rel.)")


def preprocess(path, size: int = 512, clip: float = 2.0, tiles: int = 8):
    """RGB -> 512 x 512 -> CLAHE on L (Lab) -> per-channel z-score -> luma (float64)."""
    import cv2
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    bgr = cv2.resize(bgr, (size, size), interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    lab[..., 0] = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tiles, tiles)).apply(lab[..., 0])
    rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB).astype(np.float64)
    rgb = (rgb - rgb.mean(axis=(0, 1))) / (rgb.std(axis=(0, 1)) + 1e-8)
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]


def band_energies(luma, wavelet: str = "db4", level: int = 4):
    """Energy fractions of A, D1..D4 and detail-relative energies of D1..D4."""
    import pywt
    coeffs = pywt.wavedec2(luma, wavelet, level=level)       # [cA_L, (H,V,D)_L, ..., (H,V,D)_1]
    e_a = float(np.sum(coeffs[0] ** 2))
    e_d = {f"D{level - i}": float(sum(np.sum(c ** 2) for c in coeffs[1 + i])) for i in range(level)}
    total = e_a + sum(e_d.values())
    det = sum(e_d.values())
    out = {"A": e_a / total}
    out.update({f"D{j}": e_d[f"D{j}"] / total for j in range(1, level + 1)})
    out.update({f"D{j} (rel.)": e_d[f"D{j}"] / det for j in range(1, level + 1)})
    return out


def kruskal_eta2(groups):
    """Kruskal-Wallis H, p and eta^2 = (H - k + 1) / (n - k) over a list of 1-D samples."""
    from scipy.stats import kruskal
    groups = [np.asarray(g, float) for g in groups if len(g) > 0]
    k, n = len(groups), sum(len(g) for g in groups)
    if k < 2 or n <= k:
        return {"H": float("nan"), "p": float("nan"), "eta2": float("nan"), "k": k, "n": n}
    h, p = kruskal(*groups)
    return {"H": float(h), "p": float(p), "eta2": float((h - k + 1) / (n - k)), "k": k, "n": n}


def compare(rows, group_key, bands):
    """Kruskal-Wallis per band between the groups defined by row[group_key]."""
    labels = sorted({r[group_key] for r in rows})
    res = {}
    for b in bands:
        res[b] = kruskal_eta2([[r[b] for r in rows if r[group_key] == g] for g in labels])
    best = max(res, key=lambda b: -np.inf if np.isnan(res[b]["eta2"]) else res[b]["eta2"])
    return {"groups": labels, "n": len(rows), "n_bmrd": sum(int(r["bmrd"]) for r in rows),
            "per_band": res, "largest_band": best, "largest_eta2": res[best]["eta2"]}


def collection_medians(rows, bands=BANDS_REL):
    """Fig. 2a: median band energy per collection as % difference from the mean of the medians."""
    cols = sorted({r["collection"] for r in rows})
    out = {}
    for b in bands:
        med = {c: float(np.median([r[b] for r in rows if r["collection"] == c])) for c in cols}
        mean = np.mean(list(med.values()))
        out[b] = {c: 100.0 * (v - mean) / mean for c, v in med.items()}
    return out


def run_prestudy(csv_path, out_dir, size: int = 512):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(csv_path, newline="", encoding="utf-8") as fh:
        meta = list(csv.DictReader(fh))
    root = Path(csv_path).resolve().parent
    rows = []
    for m in meta:
        p = Path(m["image"])
        e = band_energies(preprocess(p if p.is_absolute() else root / p, size))
        rows.append({**m, "bmrd": int(m.get("bmrd") or 0), **e})
    with open(out_dir / "band_energy_per_image.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    bands = BANDS_ABS + BANDS_REL
    report = {"between_collections_all": compare(rows, "collection", bands),
              "within_centre": {}, "between_sources": {}, "fig2a_median_pct_diff": collection_medians(rows)}
    for c in sorted({r["centre"] for r in rows if r.get("centre")}):
        sub = [r for r in rows if r.get("centre") == c]
        if len({r["collection"] for r in sub}) > 1:          # centres that supplied > 1 disease
            report["within_centre"][c] = compare(sub, "collection", bands)
    for dis in sorted({r["collection"] for r in rows}):
        sub = [r for r in rows if r["collection"] == dis]
        for r in sub:                                        # BMRD centres count as separate sources
            r["_source"] = f"BMRD:{r['centre']}" if r["bmrd"] and r.get("centre") else r["source"]
        if len({r["_source"] for r in sub}) > 1:
            report["between_sources"][dis] = compare(sub, "_source", bands)
    (out_dir / "prestudy_report.json").write_text(json.dumps(report, indent=1))
    return report
