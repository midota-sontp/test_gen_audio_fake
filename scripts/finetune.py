"""Partial WavLM fine-tuning stage (CPU-friendly).

Pipeline:
  1. cache encoder-input (CNN) features once per clip  -> data/ft_feats/{split}
  2. fine-tune the top N of the first `n_keep` WavLM layers + an MLP head
  3. evaluate the best checkpoint on test and write the same reports/status/
     registry artifacts as the frozen pipeline, so the dashboard shows this run.

Run inside the container:
    python scripts/finetune.py --config configs/mvp.yaml
"""
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn

import _bootstrap  # noqa: F401  (adds repo root to sys.path)
from src.evaluator.metrics import compute_metrics, save_plots
from src.models.wavlm_finetune import WavLMPartialFinetune, encoder_input
from src.monitoring import overfitting as of
from src.monitoring.status import RunStatus
from src.utils.config import load_config, resolve
from src.utils.device import select_device
from src.utils.logging import get_logger
from src.utils.seed import set_seed
from src.wavlm.extractor import WavLMExtractor

log = get_logger("finetune")


def _read_manifest(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def _cache_split(ex, rows, out_dir: Path):
    """Cache encoder-input features for a split (idempotent)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, r in enumerate(rows):
        out = out_dir / f"sample_{i:04d}.pt"
        if out.exists():
            continue
        wav, _ = sf.read(str(resolve(r["path"])), dtype="float32")
        x = encoder_input(ex.model, torch.from_numpy(wav).to(ex.device)).cpu()
        torch.save({"x": x, "label": int(r["label"])}, out)
        if (i + 1) % 200 == 0:
            log.info("  cached %d/%d -> %s", i + 1, len(rows), out_dir.name)


def _load_split(out_dir: Path):
    files = sorted(out_dir.glob("sample_*.pt"))
    feats = [torch.load(f) for f in files]
    t_min = min(d["x"].shape[0] for d in feats)          # clips are fixed-length; guard anyway
    X = torch.stack([d["x"][:t_min] for d in feats]).float()
    y = torch.tensor([d["label"] for d in feats], dtype=torch.float32)
    return X, y


def _iter_batches(X, y, bs, shuffle, rng):
    idx = list(range(len(X)))
    if shuffle:
        rng.shuffle(idx)
    for k in range(0, len(idx), bs):
        j = idx[k:k + bs]
        yield X[j], y[j]


@torch.no_grad()
def _scores(model, X, device, bs=32):
    model.eval()
    out = []
    for k in range(0, len(X), bs):
        logits = model(X[k:k + bs].to(device))
        out.append(torch.sigmoid(logits).cpu())
    return torch.cat(out).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/mvp.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg.seed)
    fc = cfg.finetune
    device = select_device(cfg.train.device)
    import random
    rng = random.Random(int(cfg.get("seed", 42)))

    ex = WavLMExtractor(resolve(cfg.paths.wavlm_checkpoint), device=device)

    # 1) cache encoder-input features -------------------------------------
    mdir = resolve(cfg.paths.manifest_dir)
    feat_root = resolve(fc.feat_dir)
    log.info("Caching encoder-input features (device=%s)", device)
    for split in ("train", "val", "test"):
        _cache_split(ex, _read_manifest(mdir / f"{split}.csv"), feat_root / split)
    Xtr, ytr = _load_split(feat_root / "train")
    Xva, yva = _load_split(feat_root / "val")
    Xte, yte = _load_split(feat_root / "test")
    log.info("features: train=%s val=%s test=%s", tuple(Xtr.shape), tuple(Xva.shape), tuple(Xte.shape))

    # 2) build & fine-tune -------------------------------------------------
    model = WavLMPartialFinetune(
        ex.model,
        n_keep=int(fc.n_keep_layers),
        n_trainable=int(fc.n_trainable_layers),
        hidden_dim=int(fc.hidden_dim),
        dropout=float(fc.dropout),
        pooling=str(fc.pooling),
    ).to(device)
    n_train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("trainable params: %.2fM (layers %d-%d + head)", n_train_params / 1e6,
             model.n_frozen + 1, model.n_keep)

    opt = torch.optim.AdamW(
        [{"params": model.backbone_parameters(), "lr": float(fc.lr_backbone)},
         {"params": model.head.parameters(), "lr": float(fc.lr_head)}],
        weight_decay=float(fc.weight_decay),
    )
    crit = nn.BCEWithLogitsLoss()
    trainable_keys = [n for n, p in model.named_parameters() if p.requires_grad]

    def snapshot():
        sd = model.state_dict()
        return {k: sd[k].detach().cpu().clone() for k in trainable_keys}

    def restore(s):
        sd = model.state_dict(); sd.update(s); model.load_state_dict(sd)

    report_dir = resolve(cfg.paths.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    run_id = "ft-" + datetime.now().strftime("%Y%m%d-%H%M%S")
    status = RunStatus(resolve(cfg.monitoring.status_file), run_id)
    status.start_stage("train", total=int(fc.epochs), detail=f"finetune L{model.n_frozen + 1}-{model.n_keep} device={device}")

    hist_path = report_dir / "history.csv"
    cols = ["epoch", "lr", "train_loss", "val_loss", "accuracy", "precision",
            "recall", "f1", "roc_auc", "eer"]
    with open(hist_path, "w", newline="") as f:
        csv.writer(f).writerow(cols)

    history, best_val, best_epoch, bad = [], float("inf"), -1, 0
    patience = int(fc.early_stopping_patience)
    bs = int(fc.batch_size)
    for epoch in range(1, int(fc.epochs) + 1):
        model.train()
        tot, nb = 0.0, 0
        for xb, yb in _iter_batches(Xtr, ytr, bs, True, rng):
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
            tot += float(loss); nb += 1
        train_loss = tot / max(nb, 1)

        # validate
        model.eval()
        with torch.no_grad():
            vtot, vnb = 0.0, 0
            for xb, yb in _iter_batches(Xva, yva, 64, False, rng):
                vtot += float(crit(model(xb.to(device)), yb.to(device))); vnb += 1
            val_loss = vtot / max(vnb, 1)
        vm = compute_metrics(yva.numpy(), _scores(model, Xva, device))

        row = {"epoch": epoch, "lr": opt.param_groups[1]["lr"], "train_loss": train_loss,
               "val_loss": val_loss, **{k: vm[k] for k in
               ("accuracy", "precision", "recall", "f1", "roc_auc", "eer")}}
        with open(hist_path, "a", newline="") as f:
            csv.writer(f).writerow([row[c] for c in cols])
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, **vm})
        overfit = of.analyze(history, dict(cfg.monitoring.overfitting))
        status.update_stage("train", processed=epoch, epoch=epoch, epochs=int(fc.epochs),
                            train_loss=round(train_loss, 5), val_loss=round(val_loss, 5),
                            val_metrics={k: round(vm[k], 4) for k in
                                         ("accuracy", "precision", "recall", "f1", "roc_auc", "eer")},
                            overfitting=overfit)
        log.info("epoch %2d | train %.4f | val %.4f | val_AUC %.3f val_EER %.3f | %s",
                 epoch, train_loss, val_loss, vm["roc_auc"], vm["eer"], overfit["level"])

        if val_loss < best_val - float(fc.get("min_delta", 1e-4)):
            best_val, best_epoch, bad, best_state = val_loss, epoch, 0, snapshot()
        else:
            bad += 1
        status.set_early_stopping({"best_val_loss": round(best_val, 5),
                                   "current_val_loss": round(val_loss, 5),
                                   "best_epoch": best_epoch, "patience": patience,
                                   "patience_left": max(0, patience - bad)})
        if bad >= patience:
            log.info("Early stopping @ epoch %d (best epoch %d, val_loss %.4f)", epoch, best_epoch, best_val)
            break
    status.finish_stage("train", best_epoch=best_epoch, best_val_loss=round(best_val, 5))

    # 3) evaluate best on test --------------------------------------------
    restore(best_state)
    ckpt_dir = resolve(cfg.paths.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "best_finetune.pt"
    torch.save({"trainable": best_state, "config": dict(fc), "best_epoch": best_epoch}, ckpt_path)

    status.start_stage("evaluate", total=len(Xte))
    scores = _scores(model, Xte, device)
    thr = float(cfg.evaluate.decision_threshold)
    metrics = compute_metrics(yte.numpy(), scores, threshold=thr)
    (report_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    save_plots(yte.numpy(), scores, report_dir, threshold=thr)
    from sklearn.metrics import classification_report
    rep = classification_report(yte.numpy().astype(int), (scores >= thr).astype(int),
                                target_names=["real", "fake"], zero_division=0)
    (report_dir / "classification_report.txt").write_text(rep)

    # registry entry so the dashboard history shows this run
    reg = resolve(cfg.tracking.registry)
    reg.parent.mkdir(parents=True, exist_ok=True)
    with open(reg, "a") as f:
        f.write(json.dumps({
            "run_id": run_id, "timestamp": datetime.now().isoformat(timespec="seconds"),
            "dataset": cfg.dataset.hf_id, "params": {"mode": "finetune", **dict(fc)},
            "test_metrics": {k: v for k, v in metrics.items() if k != "confusion_matrix"},
            "checkpoint": str(ckpt_path),
        }, default=str) + "\n")
    status.finish_stage("evaluate", **{k: round(metrics[k], 4) for k in
                        ("accuracy", "precision", "recall", "f1", "roc_auc", "eer")})

    log.info("FINETUNE done. test: AUC %.3f EER %.3f acc %.3f f1 %.3f (best epoch %d)",
             metrics["roc_auc"], metrics["eer"], metrics["accuracy"], metrics["f1"], best_epoch)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
