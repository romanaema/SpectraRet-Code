"""Section 7.7 / Tables 20-21 - joint grading and lesion segmentation on one shared encoder.

Configurations (fold 0 of the graded cohort for grading, LabelMe lesion masks for segmentation):
  C1   grading only
  C2   segmentation only
  C3   grading + segmentation with a gated lesion-attention link
  C3a  grading + segmentation without the link
Shared encoder: ImageNet MobileNetV2 at 256 x 256, frozen (unfreezing from block 13 made
grading collapse to the majority class within three epochs - `unfreeze_from=13` re-runs it).
Decoders: a small U-Net-style decoder (Ronneberger et al., 2015) trained with BCE + soft Dice,
then a larger RMHA-Net-style decoder (residual blocks, multi-head self-attention at the
bottleneck, attention-gated skips; after Zedan et al.) with the two losses weighted by learned
uncertainty (Kendall, Gal & Cipolla, 2018). The link multiplies the pooled grading features by
(1 + tanh(g) * a), a = lesion probability map pooled to the encoder grid, g learned and
initialised at 0; it stayed closed (|tanh g| < 0.07). Lesion-balanced batches oversample
images with rare lesion types.
Decoder widths are implementation choices.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbones import mobilenet_v2
from .config import DISEASES, N_CLASSES, SEG_CHANNELS, SEG_GROUPS
from .heads import GradingHead
from .losses import masked_ce, masked_seg_loss
from .metrics import qwk
from .segmentation import conv_bn_relu

# MobileNetV2 feature indices that end each stride (2, 4, 8, 16, 32) and their channels
_TAPS = (1, 3, 6, 13, 18)
_TAP_CH = (16, 24, 32, 96, 1280)


class SharedEncoder(nn.Module):
    def __init__(self, pretrained: bool = True, unfreeze_from: int | None = None):
        super().__init__()
        self.features = mobilenet_v2(pretrained).features
        for i, block in enumerate(self.features):
            for p in block.parameters():
                p.requires_grad = unfreeze_from is not None and i >= unfreeze_from
        self.frozen_all = unfreeze_from is None

    def train(self, mode: bool = True):
        super().train(mode)
        if self.frozen_all:
            self.features.eval()
        return self

    def forward(self, x):
        skips, h = [], x
        for i, block in enumerate(self.features):
            h = block(h)
            if i in _TAPS:
                skips.append(h)
        return skips                                      # strides 2, 4, 8, 16, 32


class UNetDecoder(nn.Module):
    def __init__(self, n_out: int, widths=(256, 128, 64, 32, 16)):
        super().__init__()
        ch = _TAP_CH[::-1]                                # 1280, 96, 32, 24, 16
        self.blocks = nn.ModuleList()
        cin = ch[0]
        for i, w in enumerate(widths[:-1]):
            self.blocks.append(nn.Sequential(conv_bn_relu(cin + ch[i + 1], w, 3), conv_bn_relu(w, w, 3)))
            cin = w
        self.last = nn.Sequential(conv_bn_relu(cin, widths[-1], 3), nn.Conv2d(widths[-1], n_out, 1))
        nn.init.constant_(self.last[-1].bias, -4.0)

    def forward(self, skips, size):
        rev = skips[::-1]
        h = rev[0]
        for blk, skip in zip(self.blocks, rev[1:]):
            h = F.interpolate(h, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            h = blk(torch.cat([h, skip], 1))
        return F.interpolate(self.last(h), size=size, mode="bilinear", align_corners=False)


class ResBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.body = nn.Sequential(conv_bn_relu(cin, cout, 3), nn.Conv2d(cout, cout, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(cout))
        self.skip = nn.Conv2d(cin, cout, 1, bias=False) if cin != cout else nn.Identity()

    def forward(self, x):
        return F.relu(self.body(x) + self.skip(x))


class AttentionGate(nn.Module):
    """Additive attention gate on a skip connection (Oktay et al. style)."""

    def __init__(self, g_ch, x_ch, mid):
        super().__init__()
        self.wg, self.wx = nn.Conv2d(g_ch, mid, 1), nn.Conv2d(x_ch, mid, 1)
        self.psi = nn.Conv2d(mid, 1, 1)

    def forward(self, g, x):
        g = F.interpolate(g, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return x * torch.sigmoid(self.psi(F.relu(self.wg(g) + self.wx(x))))


class RMHADecoder(nn.Module):
    """RMHA-Net-style decoder: MHSA bottleneck + attention-gated skips + residual blocks."""

    def __init__(self, n_out: int, widths=(256, 192, 128, 64, 32), heads: int = 8):
        super().__init__()
        ch = _TAP_CH[::-1]
        self.reduce = nn.Conv2d(ch[0], widths[0], 1)
        self.mhsa = nn.MultiheadAttention(widths[0], heads, batch_first=True)
        self.norm = nn.LayerNorm(widths[0])
        self.gates, self.blocks = nn.ModuleList(), nn.ModuleList()
        cin = widths[0]
        for i, w in enumerate(widths[1:]):
            self.gates.append(AttentionGate(cin, ch[i + 1], max(ch[i + 1] // 2, 8)))
            self.blocks.append(ResBlock(cin + ch[i + 1], w))
            cin = w
        self.last = nn.Conv2d(cin, n_out, 1)
        nn.init.constant_(self.last.bias, -4.0)

    def forward(self, skips, size):
        rev = skips[::-1]
        h = self.reduce(rev[0])
        b, c, hh, ww = h.shape
        t = h.flatten(2).transpose(1, 2)
        t = self.norm(t + self.mhsa(t, t, t, need_weights=False)[0])
        h = t.transpose(1, 2).reshape(b, c, hh, ww)
        for gate, blk, skip in zip(self.gates, self.blocks, rev[1:]):
            s = gate(h, skip)
            h = F.interpolate(h, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            h = blk(torch.cat([h, s], 1))
        return F.interpolate(self.last(h), size=size, mode="bilinear", align_corners=False)


class JointNet(nn.Module):
    """mode: C1 | C2 | C3 | C3a; decoder: unet | rmha."""

    def __init__(self, mode: str = "C3", decoder: str = "rmha", pretrained: bool = True,
                 unfreeze_from: int | None = None, learned_weighting: bool | None = None):
        super().__init__()
        if mode not in ("C1", "C2", "C3", "C3a"):
            raise ValueError("mode must be C1, C2, C3 or C3a")
        self.mode = mode
        self.encoder = SharedEncoder(pretrained, unfreeze_from)
        self.grade = mode != "C2"
        self.seg = mode != "C1"
        self.link = mode == "C3"
        if self.grade:
            self.heads = nn.ModuleList([GradingHead(_TAP_CH[-1], k) for k in N_CLASSES])
        if self.seg:
            dec = {"unet": UNetDecoder, "rmha": RMHADecoder}[decoder]
            self.decoder = dec(len(SEG_CHANNELS))
        if self.link:
            self.g = nn.Parameter(torch.zeros(1))
        use_lw = decoder == "rmha" if learned_weighting is None else learned_weighting
        self.log_vars = nn.Parameter(torch.zeros(2)) if (use_lw and self.grade and self.seg) else None

    def forward(self, x):
        skips = self.encoder(x)
        seg_logits = self.decoder(skips, x.shape[-2:]) if self.seg else None
        logits = None
        if self.grade:
            f = skips[-1]
            if self.link:
                a = torch.sigmoid(seg_logits).amax(1, keepdim=True)
                a = F.adaptive_max_pool2d(a, f.shape[-2:])
                f = f * (1 + torch.tanh(self.g) * a)
            pooled = f.mean(dim=(2, 3))
            logits = [h(pooled) for h in self.heads]
        return logits, seg_logits

    def gate_value(self):
        return float(torch.tanh(self.g).abs()) if self.link else float("nan")


def joint_loss(model: JointNet, logits, seg_logits, y, ymask, target, sup):
    """Grading masked CE and segmentation BCE + soft Dice; summed, or weighted by learned
    uncertainty L = sum_i exp(-s_i) L_i + s_i (Kendall et al., 2018)."""
    parts = []
    if logits is not None and ymask.any():
        parts.append(masked_ce(logits, y, ymask))
    if seg_logits is not None and sup.any():
        parts.append(masked_seg_loss(seg_logits, target, sup))
    if not parts:
        return seg_logits.sum() * 0 if seg_logits is not None else sum(l.sum() for l in logits) * 0
    if model.log_vars is not None and len(parts) == 2:
        s = model.log_vars
        return sum(torch.exp(-s[i]) * parts[i] + s[i] for i in range(2))
    return sum(parts)


def lesion_balanced_weights(supervision, targets):
    """Sampling weights that up-weight images containing rare lesion channels.
    supervision (N, C) and per-image channel presence (N, C) as 0/1 arrays."""
    pres = np.asarray(targets, float) * np.asarray(supervision, float)
    freq = pres.sum(0) + 1.0
    w = 1.0 + (pres / freq).sum(1) * len(freq)
    return w / w.mean()


@torch.no_grad()
def evaluate_joint(model: JointNet, dl, device):
    """Grading QWK per disease and macro-QWK; Dice per lesion (pixels pooled over images,
    threshold 0.5), mean DR, mean AMD and mean of all lesions."""
    model.eval()
    preds = [[] for _ in DISEASES]
    ys = [[] for _ in DISEASES]
    inter = torch.zeros(len(SEG_CHANNELS))
    denom = torch.zeros(len(SEG_CHANNELS))
    seen = torch.zeros(len(SEG_CHANNELS))
    for x, y, ym, t, s in dl:
        logits, seg = model(x.to(device))
        if logits is not None:
            for d in range(len(DISEASES)):
                sel = ym[:, d]
                if sel.any():
                    preds[d].append(logits[d][sel.to(device)].argmax(1).cpu())
                    ys[d].append(y[sel, d])
        if seg is not None:
            p = (torch.sigmoid(seg) >= 0.5).float().cpu()
            m = s[:, :, None, None]
            inter += (p * t * m).sum(dim=(0, 2, 3))
            denom += ((p + t) * m).sum(dim=(0, 2, 3))
            seen += s.sum(0)
    res = {}
    if model.grade:
        q = {}
        for d, (name, k) in enumerate(zip(DISEASES, N_CLASSES)):
            if preds[d]:
                q[name] = qwk(torch.cat(ys[d]).numpy(), torch.cat(preds[d]).numpy(), k)
        res["qwk"] = q
        res["macro_qwk"] = float(np.mean(list(q.values()))) if q else float("nan")
    if model.seg:
        dice = torch.where(denom > 0, 2 * inter / denom.clamp_min(1e-12), torch.full_like(denom, float("nan")))
        dice[seen == 0] = float("nan")
        res["dice"] = dict(zip(SEG_CHANNELS, dice.tolist()))
        res["mean_dr"] = float(np.nanmean([dice[i] for i in SEG_GROUPS["DR"]]))
        res["mean_amd"] = float(np.nanmean([dice[i] for i in SEG_GROUPS["AMD"]]))
        lesions = list(SEG_GROUPS["DR"]) + list(SEG_GROUPS["AMD"])
        res["mean_all_lesions"] = float(np.nanmean([dice[i] for i in lesions]))
    if model.link:
        res["abs_tanh_g"] = model.gate_value()
    return res


class JointDataset(torch.utils.data.Dataset):
    """CSV rows with any of: DR / Glaucoma / AMD grades, mask (+ group). Returns
    image, grades, grade mask, seg target (11, H, W), seg supervision (11,)."""

    def __init__(self, rows, size: int = 256, train: bool = False):
        self.rows, self.size, self.train = rows, size, train

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        from PIL import Image

        from .data import augment, load_rgb, row_grades, to_tensor
        r = self.rows[i]
        img = load_rgb(r["image"], self.size)
        chans = np.zeros((len(SEG_CHANNELS), self.size, self.size), np.float32)
        sup = torch.zeros(len(SEG_CHANNELS))
        if r.get("mask"):
            bits = np.asarray(Image.open(r["mask"])).astype(np.int64)
            c = ((bits[None] >> np.arange(len(SEG_CHANNELS))[:, None, None]) & 1).astype(np.float32)
            chans = F.interpolate(torch.from_numpy(c)[None], size=(self.size, self.size), mode="nearest")[0].numpy()
            sup[list(SEG_GROUPS[r["group"]])] = 1.0
        if self.train:
            img, chans = augment(img, chans)
        g = row_grades(r)
        y = torch.tensor([0 if v is None else v for v in g])
        m = torch.tensor([v is not None for v in g])
        return to_tensor(img), y, m, torch.from_numpy(np.ascontiguousarray(chans)), sup


def train_joint(args):
    """python main.py joint --csv joint.csv --mode C3 --decoder rmha --out runs/joint_C3"""
    from .data import loader, read_csv, split_rows
    from .train import set_seed
    set_seed(args.seed)
    device = torch.device(args.device)
    rows = read_csv(args.csv)
    tr_rows = split_rows(rows, "train")
    tr_ds = JointDataset(tr_rows, args.image_size, True)
    if args.lesion_balanced:
        sup = np.stack([tr_ds[i][4].numpy() for i in range(len(tr_ds))])
        pres = np.stack([(tr_ds[i][3].flatten(1).amax(1) > 0).numpy() for i in range(len(tr_ds))])
        w = lesion_balanced_weights(sup, pres)
        sampler = torch.utils.data.WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double), len(w))
        tr = torch.utils.data.DataLoader(tr_ds, batch_size=args.batch, sampler=sampler, num_workers=args.workers)
    else:
        tr = loader(tr_ds, args.batch, True, args.workers)
    va = loader(JointDataset(split_rows(rows, "val"), args.image_size), args.batch, False, args.workers)
    te_rows = split_rows(rows, "test")
    model = JointNet(args.mode, args.decoder, not args.no_pretrained, args.unfreeze_from).to(device)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    best, wait, log = -1e9, 0, []
    for ep in range(1, args.epochs + 1):
        model.train()
        tot = 0.0
        for x, y, ym, t, s in tr:
            logits, seg = model(x.to(device))
            loss = joint_loss(model, logits, seg, y.to(device), ym.to(device), t.to(device), s.to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss) * len(x)
        val = evaluate_joint(model, va, device)
        score = (val.get("macro_qwk", 0.0) if model.grade else 0.0) + (val.get("mean_all_lesions", 0.0) if model.seg else 0.0)
        log.append({"epoch": ep, "loss": tot / max(len(tr.dataset), 1), "val": val})
        print(json.dumps(log[-1]), flush=True)
        if score > best + args.min_gain:
            best, wait = score, 0
            torch.save({"state": model.state_dict(), "mode": args.mode, "decoder": args.decoder}, out / "joint_best.pt")
        else:
            wait += 1
            if wait >= args.patience:
                break
    model.load_state_dict(torch.load(out / "joint_best.pt", map_location="cpu", weights_only=True)["state"])
    report = {"mode": args.mode, "decoder": args.decoder, "log": log}
    if te_rows:
        report["test"] = evaluate_joint(model, loader(JointDataset(te_rows, args.image_size), args.batch, False), device)
        print(json.dumps(report["test"], indent=1))
    (out / "joint_report.json").write_text(json.dumps(report, indent=1))
