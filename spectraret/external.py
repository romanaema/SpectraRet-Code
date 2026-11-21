"""Sections 6.4 and 7.4 / Table 16 - external evaluation and transfer scoring.

The earlier model (fold 0, seed 0 checkpoint) was scored without training on ACRIMA, G1020 and
RIM-ONE DL (binary glaucoma), ARMD (all AMD-positive), STARE (no AMD labels) and Messidor-2:
  binary glaucoma   predicted grade 0 = non-glaucomatous, higher grades = glaucomatous;
                    AUC score = 1 - P(grade 0)
  ARMD / STARE      a predicted dry or wet grade counts as an AMD detection; with only positive
                    images (ARMD) only sensitivity can be computed, and STARE without verified
                    AMD labels gives a detection rate, not specificity
  Messidor-2        DR QWK, referable-DR AUC and the QWK of the reversed predicted grades (a
                    warning check about the grade mapping, not proof of inversion)
Zero-shot transfer (Messidor -> BMRD_DR, DRISHTI-GS -> BMRD_Glaucoma) is ordinary training on
the source CSV and evaluation on the target CSV (train.predict + metrics.grading_report).
"""
from __future__ import annotations

import numpy as np

from .metrics import auroc, mcc, qwk


def binary_report(y_true, y_pred, score=None):
    y_true, y_pred = np.asarray(y_true, bool), np.asarray(y_pred, bool)
    tp, tn = int((y_true & y_pred).sum()), int((~y_true & ~y_pred).sum())
    fp, fn = int((~y_true & y_pred).sum()), int((y_true & ~y_pred).sum())
    out = {"n": len(y_true), "positive": int(y_true.sum()), "accuracy": (tp + tn) / max(len(y_true), 1),
           "sensitivity": tp / (tp + fn) if tp + fn else float("nan"),
           "specificity": tn / (tn + fp) if tn + fp else float("nan"),
           "mcc": mcc(y_true.astype(int), y_pred.astype(int), 2)}
    if score is not None:
        out["auc"] = auroc(score, y_true)
    return out


def glaucoma_binary(probs_glaucoma, y_binary):
    """probs (N, 4) of the glaucoma head; y_binary 1 = glaucomatous."""
    p = np.asarray(probs_glaucoma, float)
    return binary_report(y_binary, p.argmax(1) != 0, 1.0 - p[:, 0])


def amd_detection(probs_amd, y_binary=None):
    """probs (N, 3) of the AMD head (no, dry, wet). Without labels: detection rate only."""
    det = np.asarray(probs_amd, float).argmax(1) != 0
    if y_binary is None:
        return {"n": len(det), "detected": int(det.sum()), "detection_rate": float(det.mean())}
    return binary_report(y_binary, det, 1.0 - np.asarray(probs_amd, float)[:, 0])


def dr_external(probs_dr, y_codes, referable_from: int = 2):
    """QWK on the training codes, referable-DR AUC (score P(code >= referable_from)), and the
    QWK after reversing the predicted grades."""
    p = np.asarray(probs_dr, float)
    y = np.asarray(y_codes)
    k = p.shape[1]
    pred = p.argmax(1)
    return {"qwk": qwk(y, pred, k), "referable_auc": auroc(p[:, referable_from:].sum(1), y >= referable_from),
            "qwk_reversed_predictions": qwk(y, k - 1 - pred, k)}
