"""SpectraRet (Fig. 3), one file per module.

    config.py            constants, Config, presets (final model and control arms)
    backbones.py         ImageNet MobileNetV2 and Swin-V2-Tiny from torchvision
    segmentation.py      Stage 1: modified DeepLabV3+ and the mask applied to the image
    lsd.py               LSD: undecimated lifting decomposition into bands B1..B4
    cnn_stream.py        DSP: frozen MobileNetV2 stream (B1, B2, B3)
    swinecat_stream.py   DSP: SwinECAT stream, Swin-V2-Tiny + ECA (B3, B4)
    dsp.py               DSP: dual-stream tokenizer -> band tokens z_1..z_4 + band embedding
    csf.py               CSF: cross-spectral fusion
    dsr.py               DSR: disease-spectral router, e_d = sum_b w_b^d z_b
    heads.py             DR / Glaucoma / AMD softmax heads
    model.py             SpectraRet: all of the above connected
    losses.py            masked cross-entropy, Stage 1 BCE + Dice
    samx.py              SAM-X, band Integrated Gradients, band deletion
    uncertainty.py       temperature scaling, MC dropout, SNGP-style head
    metrics.py           QWK, macro-F1, MCC, ECE, Brier, NLL, AURC, error AUROC
    data.py              CSV datasets and augmentation
    train.py             Stage 1 and Stage 2 training, evaluation, checkpoint loading
    studies.py           explanation and uncertainty studies on a trained model
    selftest.py          random-tensor self-test of the connected model

    compact_stream.py    earlier model: compact transformer (96 px, two blocks, from scratch)
    coral.py             ordinal CORAL head and loss (tried, collapsed)
    prestudy.py          spectral pre-study: CLAHE, db4 DWT band energy, Kruskal-Wallis, eta^2
    probe.py             linear probe on frozen MobileNetV2 features (learnability screening)
    splits.py            label encodings, stratified grouped folds, split check
    stats.py             bootstrap CIs, Wilcoxon + Holm, Friedman / Nemenyi, gate rule
    xai_input.py         grad x input, IG, occlusion, localization, reference maps, sanity checks
    seg_uncertainty.py   MC-dropout Stage-1 decoder: pixel AUROC, uncertainty vs Dice, referral
    jointseg.py          joint grading + segmentation (C1, C2, C3, C3a; U-Net / RMHA decoders)
    baselines.py         frozen ImageNet CNN baselines and the Gate P1 small CNN
    external.py          external evaluation: binary glaucoma, AMD detection, Messidor-2 checks
    cost.py              parameters, MACs, CPU throughput
    lsd_analysis.py      LSD exactness and band-operator frequency responses
"""
from .config import DISEASES, N_CLASSES, PRESETS, Config
from .model import Output, SpectraRet

__all__ = ["Config", "PRESETS", "DISEASES", "N_CLASSES", "SpectraRet", "Output"]
