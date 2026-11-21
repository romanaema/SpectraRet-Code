#!/usr/bin/env python3
"""SpectraRet (Fig. 3) - entry point. The modules live in spectraret/, one file each,
and are connected in spectraret/model.py.

    python main.py demo                         # self-test on random tensors, no data needed
    python main.py train-seg   --csv seg.csv    --out runs/stage1
    python main.py train       --csv grades.csv --seg-ckpt runs/stage1/stage1_best.pt --out runs/final
    python main.py train       --csv grades.csv --preset raw_control --out runs/raw
    python main.py explain     --csv grades.csv --ckpt runs/final/best.pt --out runs/xai
    python main.py uncertainty --csv grades.csv --ckpt runs/final/best.pt --disease DR --out runs/unc

    python main.py prestudy    --csv prestudy.csv --out runs/prestudy         # Section 4, Table 5
    python main.py probe       --csv folder.csv   --out runs/probe_aptos      # Section 3.2 linear probe
    python main.py folds       --csv grades.csv   --out folds/                # 5 grouped folds, 15% val
    python main.py baselines   --csv folds/fold_0.csv --out runs/cnn_f0       # frozen ImageNet CNNs
    python main.py gate-p1     --csv folds/fold_0.csv --input fixed --out runs/p1_fixed
    python main.py joint       --csv joint.csv --mode C3 --decoder rmha --out runs/joint_C3
    python main.py cost        --preset final                                 # Table 19
    python main.py lsd-check                                                  # Fig. 4b, exactness

Presets:  final (default, the reported model) | full (+ CSF + learned DSR, every box of Fig. 3)
          | learnable_lsd | inverted | all_cnn | raw_control | swinecat_only | coral
          | earlier_control | earlier_lsd | earlier | earlier_csf | earlier_dsr
          | earlier_inverted_dsr | earlier_inverted          (gate cascade of the earlier model)
CSV formats: see spectraret/data.py (and the module docstrings of the new commands).
Requirements: python >= 3.9, torch >= 2.0, torchvision >= 0.15, numpy, pillow;
              scipy, scikit-learn, PyWavelets, opencv-python for the analysis commands;
              timm for the InceptionResNetV2 / Xception baselines; fvcore optional for MACs.
"""
from __future__ import annotations

import argparse
import json

import torch

from spectraret.baselines import BACKBONES
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

    ps = sub.add_parser("prestudy", help="spectral pre-study: band energy, Kruskal-Wallis, eta^2")
    ps.add_argument("--csv", required=True)
    ps.add_argument("--out", required=True)
    ps.add_argument("--size", type=int, default=512)

    pr = sub.add_parser("probe", parents=[common], help="linear probe on frozen MobileNetV2 features")
    pr.add_argument("--csv", required=True)
    pr.add_argument("--out", required=True)
    pr.add_argument("--folds", type=int, default=4)

    fo = sub.add_parser("folds", parents=[common], help="stratified grouped folds + split check")
    fo.add_argument("--csv", required=True)
    fo.add_argument("--out", required=True)
    fo.add_argument("--n-folds", type=int, default=5)
    fo.add_argument("--val-frac", type=float, default=0.15)

    bl = sub.add_parser("baselines", parents=[common], help="frozen ImageNet CNN baselines")
    bl.add_argument("--csv", required=True)
    bl.add_argument("--out", required=True)
    bl.add_argument("--backbones", nargs="+", default=list(BACKBONES), choices=BACKBONES)
    bl.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])

    p1 = sub.add_parser("gate-p1", parents=[common, train_opts], help="Gate P1 small CNN")
    p1.add_argument("--csv", required=True)
    p1.add_argument("--out", required=True)
    p1.add_argument("--input", choices=["raw", "fixed", "learnable"], required=True)
    p1.add_argument("--image-size", type=int, default=160)

    jo = sub.add_parser("joint", parents=[common, train_opts], help="joint grading + segmentation")
    jo.add_argument("--csv", required=True)
    jo.add_argument("--out", required=True)
    jo.add_argument("--mode", choices=["C1", "C2", "C3", "C3a"], required=True)
    jo.add_argument("--decoder", choices=["unet", "rmha"], default="rmha")
    jo.add_argument("--image-size", type=int, default=256)
    jo.add_argument("--unfreeze-from", type=int, default=None)
    jo.add_argument("--lesion-balanced", action="store_true")
    jo.add_argument("--no-pretrained", action="store_true")

    co = sub.add_parser("cost", parents=[model_opts], help="parameters, MACs, CPU throughput")
    co.add_argument("--threads", type=int, default=4)
    co.add_argument("--pretrained", action="store_true")

    sub.add_parser("lsd-check", help="LSD exactness and band-operator frequency responses")
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
    elif args.cmd == "prestudy":
        from spectraret.prestudy import run_prestudy
        rep = run_prestudy(args.csv, args.out, args.size)
        print(json.dumps({k: v for k, v in rep.items() if k != "fig2a_median_pct_diff"}, indent=1))
    elif args.cmd == "probe":
        from spectraret.probe import run_probe
        run_probe(args.csv, args.out, args.folds, args.seed, args.device)
    elif args.cmd == "folds":
        from spectraret.splits import write_fold_csvs
        write_fold_csvs(args.csv, args.out, args.n_folds, args.val_frac, args.seed)
    elif args.cmd == "baselines":
        from spectraret.baselines import run_frozen_baselines
        run_frozen_baselines(args)
    elif args.cmd == "gate-p1":
        from spectraret.baselines import train_gate_p1
        train_gate_p1(args)
    elif args.cmd == "joint":
        from spectraret.jointseg import train_joint
        train_joint(args)
    elif args.cmd == "cost":
        from spectraret.cost import cost_report
        from spectraret.model import SpectraRet
        cfg = make_config(args)
        cfg.pretrained = args.pretrained
        print(json.dumps(cost_report(SpectraRet(cfg), cfg.image_size, threads=args.threads), indent=1))
    elif args.cmd == "lsd-check":
        from spectraret.lsd_analysis import (band_impulse_responses, energy_above,
                                             radial_magnitude_response, reconstruction_error)
        freqs, curves = radial_magnitude_response(band_impulse_responses())
        print(f"max |sum_b f_b - x| (float64): {reconstruction_error():.2e}")
        for b, c in enumerate(curves):
            print(f"band {b + 1}: energy above 0.15 c/px {energy_above(freqs, c, 0.15):.3f}, "
                  f"above 0.06 c/px {energy_above(freqs, c, 0.06):.3f}")


if __name__ == "__main__":
    main()
