"""Huấn luyện classifier trên embedding đã cache."""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .corpus.manifest import Manifest
from .features import FeatureStore
from .features.backbones import Backbone
from .metrics import compute_eer, decision_threshold
from .models import build_head, save_checkpoint
from .utils import ensure_dir, get_logger, human_time

log = get_logger("aidetector.train")


def _load_split(store: FeatureStore, manifest: Manifest, split: str):
    X, y, records = store.load_many(manifest.by_split(split))
    if len(records) == 0:
        raise RuntimeError(
            f"Split {split!r} không có embedding nào. Chạy `python -m aidetector features` trước."
        )
    log.info("  %-5s: %d mẫu (real=%d fake=%d)", split, len(y), int((y == 0).sum()), int((y == 1).sum()))
    return X, y, records


def train(
    manifest: Manifest,
    backbone: Backbone,
    config,
    cache_root: str | Path = "features",
    checkpoint_dir: str | Path = "checkpoints",
    report_dir: str | Path = "reports",
    device: str = "cpu",
) -> dict:
    store = FeatureStore(cache_root, backbone)
    log.info("Nạp embedding từ %s", store.dir)
    X_train, y_train, _ = _load_split(store, manifest, "train")
    X_val, y_val, _ = _load_split(store, manifest, "val")

    # Chuẩn hoá theo thống kê của TRAIN (không được nhìn val/test).
    mean = X_train.mean(axis=0, keepdims=True)
    std = X_train.std(axis=0, keepdims=True) + 1e-6
    X_train_n = (X_train - mean) / std
    X_val_n = (X_val - mean) / std

    head = build_head(config.section("model"), input_dim=X_train.shape[1]).to(device)

    epochs = int(config.get("train.epochs", 100))
    batch_size = int(config.get("train.batch_size", 64))
    lr = float(config.get("train.lr", 5e-4))
    weight_decay = float(config.get("train.weight_decay", 0.01))
    patience = int(config.get("train.early_stopping.patience", 20))
    min_delta = float(config.get("train.early_stopping.min_delta", 1e-4))
    monitor = str(config.get("train.early_stopping.monitor", "val_eer"))

    loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train_n), torch.from_numpy(y_train.astype(np.float32))),
        batch_size=batch_size, shuffle=True, drop_last=len(y_train) > batch_size,
        num_workers=int(config.get("train.num_workers", 0)),
    )
    val_x = torch.from_numpy(X_val_n).to(device)
    val_y = torch.from_numpy(y_val.astype(np.float32)).to(device)

    # Cân bằng lớp: nếu fake nhiều/ít hơn real thì bù bằng pos_weight.
    n_fake = max(int((y_train == 1).sum()), 1)
    n_real = max(int((y_train == 0).sum()), 1)
    pos_weight = torch.tensor([n_real / n_fake], device=device)
    if abs(n_real / n_fake - 1) > 0.05:
        log.info("Lệch lớp real/fake = %.2f ⇒ pos_weight=%.3f", n_real / n_fake, pos_weight.item())
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min",
        factor=float(config.get("train.scheduler.factor", 0.5)),
        patience=int(config.get("train.scheduler.patience", 5)),
    )

    report_dir = ensure_dir(report_dir)
    history_path = report_dir / "history.csv"
    history: list[dict] = []
    # So sánh theo cặp (chỉ số chính, val_loss): khi val_eer bằng nhau — rất hay xảy
    # ra vì EER nhận giá trị rời rạc — val_loss thấp hơn được coi là tốt hơn.
    best_score = (float("inf"), float("inf"))
    best_epoch = -1
    bad_epochs = 0
    started = time.time()

    log.info("Bắt đầu huấn luyện: %d epoch tối đa · theo dõi %s · patience %d", epochs, monitor, patience)
    for epoch in range(1, epochs + 1):
        head.train()
        total = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(head(xb), yb)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(xb)
        train_loss = total / max(len(loader.dataset), 1)

        head.eval()
        with torch.no_grad():
            val_logits = head(val_x)
            val_loss = criterion(val_logits, val_y).item()
            val_scores = torch.sigmoid(val_logits).cpu().numpy()
        val_eer, _ = compute_eer(y_val, val_scores)
        val_threshold = decision_threshold(y_val, val_scores)

        scheduler.step(val_loss)
        gap = val_loss - train_loss
        row = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_loss": round(val_loss, 6),
            "val_eer": round(float(val_eer), 6),
            "lr": optimizer.param_groups[0]["lr"],
            "gap": round(gap, 6),
        }
        history.append(row)

        primary = val_eer if monitor == "val_eer" else val_loss
        score = (primary, val_loss)
        improved = primary < best_score[0] - min_delta or (
            abs(primary - best_score[0]) <= min_delta and val_loss < best_score[1]
        )
        if improved:
            best_score, best_epoch, bad_epochs = score, epoch, 0
            save_checkpoint(
                Path(checkpoint_dir) / "best.pt",
                head,
                {
                    "model": config.section("model"),
                    "input_dim": int(X_train.shape[1]),
                    "norm_mean": mean.tolist(),
                    "norm_std": std.tolist(),
                    "backbone": {
                        "name": backbone.id,
                        "checkpoint": backbone.checkpoint,
                        "output_layer": backbone.output_layer,
                        "pooling": backbone.pooling,
                    },
                    "audio": config.section("audio"),
                    "epoch": epoch,
                    "val_eer": float(val_eer),
                    "val_loss": float(val_loss),
                    # Ngưỡng quyết định mặc định lúc suy luận, chốt trên tập val.
                    "threshold": float(val_threshold),
                },
            )
        else:
            bad_epochs += 1

        log.info(
            "epoch %3d/%d · train %.4f · val %.4f · EER %.2f%% · gap %+.3f%s",
            epoch, epochs, train_loss, val_loss, val_eer * 100, gap, "  ★" if improved else "",
        )
        if gap > float(config.get("monitoring.overfitting.warning", 0.30)):
            log.warning("  ⚠ khoảng cách val-train = %.3f — có dấu hiệu overfit", gap)
        if bad_epochs >= patience:
            log.info("Dừng sớm ở epoch %d (%s không cải thiện %d epoch)", epoch, monitor, patience)
            break

    with history_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)

    summary = {
        "epochs_run": len(history),
        "best_epoch": best_epoch,
        "best_val_eer": float(min(h["val_eer"] for h in history)),
        "best_val_loss": float(min(h["val_loss"] for h in history)),
        "monitor": monitor,
        "checkpoint": str(Path(checkpoint_dir) / "best.pt"),
        "history": str(history_path),
        "elapsed": human_time(time.time() - started),
    }
    (report_dir / "train_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    log.info(
        "Xong sau %s · best epoch %d · val EER %.2f%%",
        summary["elapsed"], best_epoch, summary["best_val_eer"] * 100,
    )
    return summary
