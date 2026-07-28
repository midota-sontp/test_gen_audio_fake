"""Training loop: AdamW + BCEWithLogits + ReduceLROnPlateau + early stopping.

Checkpoints (model/optimizer/scheduler/epoch/metrics/config) support resume,
best and last. Per-epoch metrics are appended to reports/history.csv — the
substrate a dashboard/TensorBoard can consume later.
"""
from __future__ import annotations

import csv
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..dataset.embeddings_ds import EmbeddingDataset
from ..evaluator.metrics import compute_metrics
from ..models.mlp import MLPClassifier
from ..monitoring import overfitting as of
from ..monitoring.status import RunStatus, resolve_run_id
from ..monitoring.tracking import ExperimentTracker
from ..utils.config import Config, resolve
from ..utils.device import select_device
from ..utils.logging import get_logger

log = get_logger("train")


def _loader(split_dir, batch_size, shuffle, num_workers) -> DataLoader:
    return DataLoader(
        EmbeddingDataset(split_dir),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )


@torch.no_grad()
def _evaluate(model, loader, device, criterion) -> tuple[float, dict]:
    model.eval()
    losses, all_scores, all_labels = [], [], []
    for emb, y in loader:
        emb, y = emb.to(device), y.to(device)
        logits = model(emb)
        losses.append(criterion(logits, y).item())
        all_scores.append(torch.sigmoid(logits).cpu().numpy())
        all_labels.append(y.cpu().numpy())
    scores = np.concatenate(all_scores)
    labels = np.concatenate(all_labels)
    return float(np.mean(losses)), compute_metrics(labels, scores)


def train(cfg: Config) -> Path:
    tcfg = cfg.train
    device = select_device(tcfg.device)
    log.info("Training on %s", device)

    emb_dir = resolve(cfg.paths.embedding_dir)
    ckpt_dir = resolve(cfg.paths.checkpoint_dir)
    report_dir = resolve(cfg.paths.report_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    train_loader = _loader(emb_dir / "train", tcfg.batch_size, True, tcfg.num_workers)
    val_loader = _loader(emb_dir / "val", tcfg.batch_size, False, tcfg.num_workers)

    model = MLPClassifier(cfg.model.input_dim, cfg.model.hidden_dim, cfg.model.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=tcfg.lr, weight_decay=tcfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=tcfg.scheduler.factor, patience=tcfg.scheduler.patience
    )
    criterion = nn.BCEWithLogitsLoss()

    # monitoring / experiment tracking (fail-safe: never breaks training)
    run_id = resolve_run_id(cfg)
    status = RunStatus(resolve(cfg.monitoring.status_file), run_id)
    tracker = ExperimentTracker(cfg, run_id)
    tracker.log_params({**dict(cfg.model), **dict(cfg.train), "run_id": run_id})
    status.start_stage("train", total=int(tcfg.epochs), detail=f"device={device}")

    history_path = report_dir / "history.csv"
    hist_fields = ["epoch", "lr", "train_loss", "val_loss", "accuracy",
                   "precision", "recall", "f1", "roc_auc", "eer"]
    with open(history_path, "w", newline="") as f:
        csv.writer(f).writerow(hist_fields)
    history: list[dict] = []

    best_val = float("inf")
    best_epoch = -1
    patience = int(cfg.train.early_stopping.patience)
    min_delta = float(cfg.train.early_stopping.min_delta)
    bad_epochs = 0

    for epoch in range(1, int(tcfg.epochs) + 1):
        model.train()
        t0 = time.time()
        train_losses = []
        for emb, y in train_loader:
            emb, y = emb.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(emb), y)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        train_loss = float(np.mean(train_losses))

        val_loss, val_metrics = _evaluate(model, val_loader, device, criterion)
        scheduler.step(val_loss)
        lr_now = optimizer.param_groups[0]["lr"]

        log.info(
            "epoch %02d | lr %.2e | train %.4f | val %.4f | acc %.3f f1 %.3f auc %.3f eer %.3f | %.1fs",
            epoch, lr_now, train_loss, val_loss, val_metrics["accuracy"],
            val_metrics["f1"], val_metrics["roc_auc"], val_metrics["eer"], time.time() - t0,
        )
        with open(history_path, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch, lr_now, train_loss, val_loss, val_metrics["accuracy"],
                val_metrics["precision"], val_metrics["recall"], val_metrics["f1"],
                val_metrics["roc_auc"], val_metrics["eer"],
            ])

        # --- monitoring: history / tracker / status / overfitting ---
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                        **val_metrics})
        tracker.log_metrics({
            "train_loss": train_loss, "val_loss": val_loss, "lr": lr_now,
            "val_accuracy": val_metrics["accuracy"], "val_f1": val_metrics["f1"],
            "val_roc_auc": val_metrics["roc_auc"], "val_eer": val_metrics["eer"],
        }, step=epoch)
        overfit = of.analyze(history, dict(cfg.monitoring.overfitting))
        status.update_stage(
            "train", processed=epoch, epoch=epoch, epochs=int(tcfg.epochs),
            lr=lr_now, train_loss=round(train_loss, 5), val_loss=round(val_loss, 5),
            val_metrics={k: round(v, 4) for k, v in val_metrics.items()
                         if isinstance(v, float)},
            overfitting=overfit,
        )

        ckpt = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "val_loss": val_loss,
            "val_metrics": val_metrics,
            "config": {"model": dict(cfg.model), "train": dict(cfg.train)},
        }
        torch.save(ckpt, ckpt_dir / "last.pt")

        if val_loss < best_val - min_delta:
            best_val, best_epoch, bad_epochs = val_loss, epoch, 0
            torch.save(ckpt, ckpt_dir / "best.pt")
            log.info("  new best (val_loss=%.4f) -> best.pt", best_val)
        else:
            bad_epochs += 1

        status.set_early_stopping({
            "best_val_loss": round(best_val, 5), "current_val_loss": round(val_loss, 5),
            "best_epoch": best_epoch, "patience": patience,
            "patience_left": max(0, patience - bad_epochs),
        })

        if bad_epochs >= patience:
            log.info("Early stopping at epoch %d (best epoch %d, val_loss %.4f)",
                     epoch, best_epoch, best_val)
            break

    status.finish_stage("train", best_epoch=best_epoch, best_val_loss=round(best_val, 5))
    tracker.log_artifacts(ckpt_dir / "best.pt")
    tracker.close()
    log.info("Training done. Best val_loss=%.4f @ epoch %d", best_val, best_epoch)
    return ckpt_dir / "best.pt"
