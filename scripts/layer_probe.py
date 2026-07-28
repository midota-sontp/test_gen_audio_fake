"""Diagnostic: which WavLM layer carries the spoof signal?

Runs ONE frozen forward pass per clip, mean-pools EACH transformer layer, then
fits a quick logistic-regression probe per layer on train and reports val
ROC-AUC / EER. This separates two hypotheses cheaply:

  * If some middle layer beats the last layer clearly -> representation choice
    matters (switch `extract.output_layer` or learn a multi-layer weighted sum).
  * If every layer plateaus at the same ~0.80 AUC -> the frozen features are the
    ceiling; real gains need fine-tuning (or a spectro-temporal model), not head
    or pooling tweaks. It also flags train/test attack-type mismatch (H2): if
    train AUC is high on all layers but val is flat low, it's domain shift.

Usage (inside the container):
    python scripts/layer_probe.py --config configs/mvp.yaml \
        --n-train 800 --n-val 594
"""
from __future__ import annotations

import argparse
import csv
import random

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve

import _bootstrap  # noqa: F401  (adds repo root to sys.path)
from src.utils.config import load_config, resolve
from src.utils.device import select_device
from src.wavlm.extractor import WavLMExtractor


def _eer(y, s):
    fpr, tpr, _ = roc_curve(y, s)
    fnr = 1 - tpr
    i = int(np.nanargmin(np.abs(fnr - fpr)))
    return float((fpr[i] + fnr[i]) / 2)


def _read_manifest(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def _subsample(rows, n, rng):
    """Balanced subsample of up to n rows (n/2 per class where possible)."""
    if n is None or n >= len(rows):
        return rows
    by = {0: [], 1: []}
    for r in rows:
        by[int(r["label"])].append(r)
    per = n // 2
    out = []
    for c in (0, 1):
        rng.shuffle(by[c])
        out += by[c][:per]
    rng.shuffle(out)
    return out


def _all_layer_means(ex: WavLMExtractor, wav: torch.Tensor) -> np.ndarray:
    """Return [L, D] — time-mean of every transformer layer for one clip."""
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    wav = wav.to(ex.device).float()
    if getattr(ex.cfg, "normalize", False):
        wav = F.layer_norm(wav, wav.shape[1:])
    # layer_results is only populated when a tgt_layer is set, so ask for the last
    # layer explicitly — the encoder then records every layer on the way there.
    n_layers = len(ex.model.encoder.layers)
    with torch.no_grad():
        (_, layer_results), _ = ex.model.extract_features(
            wav, output_layer=n_layers, ret_conv=False, ret_layer_results=True
        )
    reps = []
    for xl, _ in layer_results[1:]:      # [0] is the input embedding; keep the L layer outputs
        xl = xl.transpose(0, 1)          # [T,B,D] -> [B,T,D]
        reps.append(xl.mean(dim=1).squeeze(0).detach().cpu().numpy())
    return np.stack(reps, axis=0)        # [L, D]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/mvp.yaml")
    ap.add_argument("--n-train", type=int, default=800)
    ap.add_argument("--n-val", type=int, default=None)   # None => all val
    args = ap.parse_args()

    cfg = load_config(args.config)
    rng = random.Random(int(cfg.get("seed", 42)))
    device = select_device(cfg.extract.device)
    ex = WavLMExtractor(resolve(cfg.paths.wavlm_checkpoint), device=device)

    mdir = resolve(cfg.paths.manifest_dir)
    train = _subsample(_read_manifest(mdir / "train.csv"), args.n_train, rng)
    val = _subsample(_read_manifest(mdir / "val.csv"), args.n_val, rng)
    print(f"probe: train={len(train)} val={len(val)} device={device}")

    def encode(rows):
        X, y = [], []
        for i, r in enumerate(rows):
            wav, _ = sf.read(str(resolve(r["path"])), dtype="float32")
            X.append(_all_layer_means(ex, torch.from_numpy(wav)))
            y.append(int(r["label"]))
            if (i + 1) % 100 == 0:
                print(f"  encoded {i + 1}/{len(rows)}")
        return np.stack(X, 0), np.array(y)   # X: [N, L, D]

    Xtr, ytr = encode(train)
    Xva, yva = encode(val)
    L = Xtr.shape[1]
    print(f"\n{'layer':>5} {'train_AUC':>10} {'val_AUC':>9} {'val_EER':>9}")
    best = (None, -1.0)
    for l in range(L):
        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit(Xtr[:, l, :], ytr)
        s_tr = clf.predict_proba(Xtr[:, l, :])[:, 1]
        s_va = clf.predict_proba(Xva[:, l, :])[:, 1]
        a_tr = roc_auc_score(ytr, s_tr)
        a_va = roc_auc_score(yva, s_va)
        e_va = _eer(yva, s_va)
        flag = ""
        if a_va > best[1]:
            best = (l, a_va)
            flag = " <-"
        print(f"{l + 1:>5} {a_tr:>10.3f} {a_va:>9.3f} {e_va:>9.3f}{flag}")

    # multi-layer: mean over all layers, and best-layer mean+std as references
    def probe(feat_tr, feat_va, name):
        clf = LogisticRegression(max_iter=2000, C=1.0).fit(feat_tr, ytr)
        s = clf.predict_proba(feat_va)[:, 1]
        print(f"{name:>18}  val_AUC={roc_auc_score(yva, s):.3f}  val_EER={_eer(yva, s):.3f}")

    print()
    probe(Xtr.mean(1), Xva.mean(1), "mean-all-layers")
    bl = best[0]
    probe(np.concatenate([Xtr[:, bl], Xtr.std(1)], 1),
          np.concatenate([Xva[:, bl], Xva.std(1)], 1), "bestlayer+layerstd")
    print(f"\nbest single layer = {best[0] + 1} (val_AUC={best[1]:.3f})")


if __name__ == "__main__":
    main()
