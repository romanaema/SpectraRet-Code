# SpectraRet - Python code

All algorithms of the paper, in PyTorch. `python main.py demo` runs a self-test on random
tensors (no data needed); `python main.py -h` lists every command.

| Paper | What | File |
|---|---|---|
| Sec. 3.2, Table 4 | label encodings (BMRD, Messidor, IDRiD, DRISHTI-GS CDR bins, AMD) | `spectraret/splits.py` |
| Sec. 3.2 | linear probe on frozen MobileNetV2 features (folder / centre screening) | `spectraret/probe.py` |
| Sec. 3.3 | stratified 5-fold split grouped by file stem, 15% validation, split check | `spectraret/splits.py` |
| Sec. 4, Table 5, Fig. 2 | spectral pre-study: CLAHE, db4 DWT band energy, Kruskal-Wallis, eta^2 | `spectraret/prestudy.py` |
| Sec. 5.1, Fig. 3 | SpectraRet model (all modules connected) | `spectraret/model.py` |
| Sec. 5.2, Eqs. 1-2, Fig. 4 | LSD lifting decomposition; exactness and operator responses | `spectraret/lsd.py`, `spectraret/lsd_analysis.py` |
| Sec. 5.3, Eq. 3 | DSP: frozen MobileNetV2 stream, SwinECAT stream, band tokens | `cnn_stream.py`, `swinecat_stream.py`, `dsp.py` |
| Sec. 5.4 | Stage 1 modified DeepLabV3+ and lesion masking | `spectraret/segmentation.py` |
| Sec. 5.5, Eq. 4 | softmax heads, masked cross-entropy; ordinal CORAL head | `heads.py`, `losses.py`, `coral.py` |
| Sec. 5.6 | learnable LSD, CSF, learned DSR + entropy penalty, compact transformer | `lsd.py`, `csf.py`, `dsr.py`, `compact_stream.py` |
| Sec. 5.7, Eqs. 5-7 | SAM-X (development and final forms), band-space IG, band deletion | `spectraret/samx.py` |
| Sec. 5.8, Eqs. 8-9 | temperature scaling, MC dropout, SNGP-style head | `spectraret/uncertainty.py` |
| Sec. 6.1, Tables 7, 9 | training (Stage 1, Stage 2); frozen ImageNet CNN baselines | `train.py`, `baselines.py` |
| Sec. 6.2 | QWK, macro-F1, MCC, macro-AUC, ECE, MCE, Brier, NLL, AURC, error AUROC | `spectraret/metrics.py` |
| Sec. 6.2-6.3, Table 8 | bootstrap CIs, Wilcoxon + Holm, Friedman / Nemenyi, gate rule | `spectraret/stats.py` |
| Sec. 7.2, Tables 11-12 | Gate P1 small CNN; gate cascade of the earlier model (presets) | `baselines.py`, `config.py` |
| Sec. 6.5, 7.5, Tables 17-18 | grad x input, IG, occlusion, localization, sanity checks | `xai_input.py`, `studies.py` |
| Sec. 7.3, Tables 13-14 | uncertainty study; referral, risk-coverage, shift; MC-dropout decoder | `studies.py`, `metrics.py`, `seg_uncertainty.py` |
| Sec. 6.4, 7.4, Table 16 | external evaluation (binary glaucoma, AMD detection, Messidor-2) | `spectraret/external.py` |
| Sec. 6.6, 7.6, Table 19 | parameters, MACs, CPU throughput | `spectraret/cost.py` |
| Sec. 7.7, Tables 20-21 | joint grading + segmentation (C1, C2, C3, C3a; U-Net / RMHA decoders) | `spectraret/jointseg.py` |

Model presets (`--preset`): `final`, `raw_control`, `swinecat_only`, `full`, `learnable_lsd`,
`inverted`, `all_cnn`, `coral`, and the earlier-model cascade `earlier_control`, `earlier_lsd`,
`earlier`, `earlier_csf`, `earlier_dsr`, `earlier_inverted_dsr`, `earlier_inverted`.

Not included: the third-party comparators MedViT V2, MSTNet, AES-Net and XV-AMD (re-implement
from their papers), the faithfulness loss (never activated; its form is not given in the paper)
and the blinded clinician rating (a human study).
