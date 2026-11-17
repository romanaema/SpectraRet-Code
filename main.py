#!/usr/bin/env python3
"""SpectraRet (Fig. 3) - entry point. The modules live in spectraret/, one file each,
and are connected in spectraret/model.py.

    python main.py demo                         # self-test on random tensors, no data needed
    python main.py train-seg   --csv seg.csv    --out runs/stage1
    python main.py train       --csv grades.csv --seg-ckpt runs/stage1/stage1_best.pt --out runs/final
    python main.py train       --csv grades.csv --preset raw_control --out runs/raw
    python main.py explain     --csv grades.csv --ckpt runs/final/best.pt --out runs/xai
    python main.py uncertainty --csv grades.csv --ckpt runs/final/best.pt --disease DR --out runs/unc

Presets:  final (default, the reported model) | full (+ CSF + learned DSR, every box of Fig. 3)
          | learnable_lsd | inverted | all_cnn | raw_control
CSV formats: see spectraret/data.py.
Requirements: python >= 3.9, torch >= 2.0, torchvision >= 0.15, numpy, pillow.
"""
from __future__ import annotations

import argparse

import torch

from spectraret.config import ASSIGNMENTS, DISEASES, PRESETS, Config
from spectraret.selftest import run_demo
from spectraret.studies import run_explain, run_uncertainty
from spectraret.train import train_stage1, train_stage2


def make_config(args) -> Config:
    """Preset first, then the individual switches."""
    cfg = Config(**PRESETS[args.preset])
    if args.no_seg:
        cfg.use_seg = False
    if args.csf:
        cfg.use_csf = True
    if args.dsr:
        cfg.dsr = args.dsr
    if args.learnable_lsd:
        cfg.lsd_learnable = True
    if args.assignment:
        cfg.assignment = args.assignment
    if args.mask_mode:
        cfg.mask_mode = args.mask_mode
    if args.image_size:
        cfg.image_size = args.image_size
    if args.no_pretrained:
        cfg.pretrained = False
    return cfg


def build_parser():
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    common.add_argument("--seed", type=int, default=0)
    common.add_argument("--batch", type=int, default=32)
    common.add_argument("--workers", type=int, default=0)

    model_opts = argparse.ArgumentParser(add_help=False)
    model_opts.add_argument("--preset", choices=sorted(PRESETS), default="final")
    model_opts.add_argument("--no-seg", action="store_true", help="skip Stage 1 (LSD reads the raw image)")
    model_opts.add_argument("--csf", action="store_true", help="switch on cross-spectral fusion")
    model_opts.add_argument("--dsr", choices=["uniform", "learned"])
    model_opts.add_argument("--learnable-lsd", action="store_true")
    model_opts.add_argument("--assignment", choices=sorted(ASSIGNMENTS))
    model_opts.add_argument("--mask-mode", choices=["hard", "soft"])
    model_opts.add_argument("--image-size", type=int)
    model_opts.add_argument("--no-pretrained", action="store_true")

    train_opts = argparse.ArgumentParser(add_help=False)
    train_opts.add_argument("--epochs", type=int, default=25)
    train_opts.add_argument("--patience", type=int, default=8)
    train_opts.add_argument("--min-gain", type=float, default=0.003)

    p = argparse.ArgumentParser(description="SpectraRet (Fig. 3) - training, explanation and uncertainty.")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("demo", parents=[model_opts], help="self-test on random tensors")
    d.add_argument("--pretrained", action="store_true", help="load ImageNet weights (default: random init)")

    s1 = sub.add_parser("train-seg", parents=[common, train_opts], help="Stage 1: segmentation")
    s1.add_argument("--csv", required=True)
    s1.add_argument("--out", required=True)
    s1.add_argument("--image-size", type=int, default=128)
    s1.add_argument("--no-pretrained", action="store_true")
    s1.set_defaults(batch=16)

    s2 = sub.add_parser("train", parents=[common, model_opts, train_opts], help="Stage 2: grading")
    s2.add_argument("--csv", required=True)
    s2.add_argument("--out", required=True)
    s2.add_argument("--seg-ckpt")

    ex = sub.add_parser("explain", parents=[common], help="SAM-X, band IG and band deletion")
    ex.add_argument("--csv", required=True)
    ex.add_argument("--ckpt", required=True)
    ex.add_argument("--out", required=True)
    ex.add_argument("--n", type=int, default=100, help="test images per disease")
    ex.add_argument("--ig-steps", type=int, default=32)
    ex.set_defaults(batch=8)

    un = sub.add_parser("uncertainty", parents=[common], help="deterministic / temperature / MC dropout / SNGP-style")
    un.add_argument("--csv", required=True)
    un.add_argument("--ckpt", required=True)
    un.add_argument("--out", required=True)
    un.add_argument("--disease", choices=DISEASES, required=True)
    un.add_argument("--passes", type=int, default=30)
    un.add_argument("--n-rff", type=int, default=512)
    return p


def main():
    args = build_parser().parse_args()
    if args.cmd == "demo":
        cfg = make_config(args)
        cfg.pretrained = args.pretrained
        run_demo(cfg, args.preset)
    elif args.cmd == "train-seg":
        train_stage1(args)
    elif args.cmd == "train":
        train_stage2(args, make_config(args))
    elif args.cmd == "explain":
        run_explain(args)
    elif args.cmd == "uncertainty":
        run_uncertainty(args)


if __name__ == "__main__":
    main()
