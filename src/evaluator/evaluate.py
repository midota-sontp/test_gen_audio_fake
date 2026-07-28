"""Evaluate the best checkpoint on the test set; write metrics, report, plots,
prediction samples, and append a row to the experiment registry."""
from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..dataset.embeddings_ds import EmbeddingDataset
from ..models.mlp import MLPClassifier
from ..monitoring.status import RunStatus, resolve_run_id
from ..utils.config import Config, resolve
from ..utils.device import select_device
from ..utils.logging import get_logger
from .metrics import compute_metrics, save_plots

log = get_logger("evaluate")


def evaluate(cfg: Config, checkpoint: str | Path | None = None) -> dict:
    from sklearn.metrics import classification_report

    device = select_device(cfg.train.device)
    run_id = resolve_run_id(cfg)
    status = RunStatus(resolve(cfg.monitoring.status_file), run_id)
    status.start_stage("evaluate")

    ckpt_path = resolve(checkpoint) if checkpoint else resolve(cfg.paths.checkpoint_dir) / "best.pt"
    ckpt = torch.load(ckpt_path, map_location=device)

    model = MLPClassifier(cfg.model.input_dim, cfg.model.hidden_dim, cfg.model.dropout).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    ds = EmbeddingDataset(resolve(cfg.paths.embedding_dir) / "test")
    loader = DataLoader(ds, batch_size=cfg.train.batch_size, shuffle=False)

    scores, labels = [], []
    with torch.no_grad():
        for emb, y in loader:
            logits = model(emb.to(device))
            scores.append(torch.sigmoid(logits).cpu().numpy())
            labels.append(y.numpy())
    scores = np.concatenate(scores)
    labels = np.concatenate(labels)
    paths = [torch.load(f, map_location="cpu").get("path", f.name) for f in ds.files]

    thr = float(cfg.evaluate.decision_threshold)
    metrics = compute_metrics(labels, scores, threshold=thr)
    log.info("TEST metrics: %s", {k: round(v, 4) if isinstance(v, float) else v
                                   for k, v in metrics.items() if k != "confusion_matrix"})
    log.info("Confusion matrix [[TN,FP],[FN,TP]]: %s", metrics["confusion_matrix"])

    report_dir = resolve(cfg.paths.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    preds = (scores >= thr).astype(int)
    rep = classification_report(labels.astype(int), preds, target_names=["real", "fake"],
                                zero_division=0)
    (report_dir / "classification_report.txt").write_text(rep)
    save_plots(labels, scores, report_dir, threshold=thr)

    _write_prediction_samples(report_dir, paths, scores, preds, labels,
                              int(cfg.evaluate.get_path("n_prediction_samples", 24)))
    _register_experiment(cfg, run_id, metrics, ckpt_path)

    status.finish_stage("evaluate", **{k: round(v, 4) for k, v in metrics.items()
                                       if isinstance(v, float)})
    log.info("Wrote metrics.json, classification_report.txt, predictions.csv, plots to %s",
             report_dir)
    return metrics


def _write_prediction_samples(report_dir, paths, scores, preds, labels, n) -> None:
    out = report_dir / "predictions.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "probability", "prediction", "ground_truth"])
        for p, s, pr, y in list(zip(paths, scores, preds, labels))[:n]:
            w.writerow([p, round(float(s), 4),
                        "fake" if pr == 1 else "real", "fake" if int(y) == 1 else "real"])


def _register_experiment(cfg, run_id, metrics, ckpt_path) -> None:
    reg = resolve(cfg.tracking.registry)
    reg.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "dataset": cfg.dataset.hf_id,
        "params": {**dict(cfg.model), **dict(cfg.train)},
        "test_metrics": {k: v for k, v in metrics.items() if k != "confusion_matrix"},
        "checkpoint": str(ckpt_path),
    }
    with open(reg, "a") as f:
        f.write(json.dumps(entry, default=str) + "\n")
