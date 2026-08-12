"""Đánh giá trên tập test + phân tích chi tiết.

Ngoài số tổng, phần quan trọng nhất là **breakdown theo generator**: mô hình bắt
tốt engine nào, thua engine nào; và nếu có engine bị giữ riêng cho test
(`splits.holdout_generators`) thì đó chính là thước đo khả năng tổng quát hoá sang
engine chưa từng thấy.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from .corpus.manifest import Manifest
from .corpus.schema import LABEL_FAKE, LABEL_REAL
from .features import FeatureStore
from .features.backbones import Backbone
from .metrics import compute_eer, full_report
from .models import load_checkpoint
from .utils import ensure_dir, get_logger

log = get_logger("aidetector.evaluate")


def _predict(head, X: np.ndarray, meta: dict, device: str) -> np.ndarray:
    mean = np.asarray(meta["norm_mean"], dtype=np.float32)
    std = np.asarray(meta["norm_std"], dtype=np.float32)
    with torch.no_grad():
        logits = head(torch.from_numpy((X - mean) / std).to(device))
        return torch.sigmoid(logits).cpu().numpy()


def evaluate(
    manifest: Manifest,
    backbone: Backbone,
    checkpoint: str | Path = "checkpoints/best.pt",
    cache_root: str | Path = "features",
    report_dir: str | Path = "reports",
    split: str = "test",
    device: str = "cpu",
    make_plots: bool = True,
) -> dict:
    head, meta = load_checkpoint(checkpoint, device)
    store = FeatureStore(cache_root, backbone)
    X, y, records = store.load_many(manifest.by_split(split))
    if len(records) == 0:
        raise RuntimeError(f"Split {split!r} rỗng hoặc chưa trích đặc trưng.")

    scores = _predict(head, X, meta, device)
    # Ngưỡng lấy từ checkpoint (đã chốt trên tập val) — đúng như lúc triển khai
    # thật; nếu checkpoint không có thì mới suy ra từ chính tập đang đánh giá.
    threshold = meta.get("threshold")
    report = full_report(y, scores, threshold)
    threshold = report["threshold"]

    log.info(
        "[%s] EER %.2f%% · AUC %.4f · ACC %.2f%% (ngưỡng %.3f) · %d mẫu",
        split, report["eer"] * 100, report["roc_auc"], report["accuracy"] * 100,
        threshold, report["n_samples"],
    )

    result = {
        "split": split,
        "checkpoint": str(checkpoint),
        "backbone": meta.get("backbone", {}),
        "overall": report,
        "by_generator": _breakdown(records, y, scores, threshold, key=lambda r: r.generator or "(real)"),
        "by_engine": _breakdown(records, y, scores, threshold, key=lambda r: r.engine or "(real)"),
        "by_source": _breakdown(records, y, scores, threshold, key=lambda r: r.source or "(không rõ)"),
        "by_condition": _breakdown(
            records, y, scores, threshold,
            key=lambda r: "augmented" if r.augment else "clean",
        ),
    }

    report_dir = ensure_dir(report_dir)
    (report_dir / "metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_predictions(report_dir / "predictions.csv", records, y, scores, threshold)
    _log_breakdown(result)

    if make_plots:
        try:
            _plots(y, scores, threshold, report_dir)
        except Exception as exc:  # noqa: BLE001 — thiếu matplotlib không nên chặn pipeline
            log.warning("Không vẽ được biểu đồ: %s", exc)

    return result


# ------------------------------------------------------------------ breakdown
def _breakdown(records, y, scores, threshold, key) -> dict:
    """Nhóm theo một thuộc tính; mỗi nhóm fake được ghép với TOÀN BỘ real để tính EER.

    (EER cần cả hai lớp; một nhóm chỉ chứa fake của một engine sẽ không tính được
    nếu không có real làm đối chứng.)
    """
    real_mask = y == 0
    real_scores = scores[real_mask]

    groups: dict[str, list[int]] = defaultdict(list)
    for i, rec in enumerate(records):
        groups[key(rec)].append(i)

    out: dict[str, dict] = {}
    for name, indices in sorted(groups.items()):
        idx = np.asarray(indices)
        group_y, group_scores = y[idx], scores[idx]
        entry = {
            "n": int(len(idx)),
            "n_real": int((group_y == 0).sum()),
            "n_fake": int((group_y == 1).sum()),
            "mean_score": round(float(group_scores.mean()), 4),
        }
        if entry["n_fake"] and entry["n_real"] == 0:
            # Chỉ có fake → ghép với real toàn cục để có EER so sánh được.
            paired_y = np.concatenate([np.zeros(len(real_scores), int), np.ones(len(idx), int)])
            paired_scores = np.concatenate([real_scores, group_scores])
            eer, _ = compute_eer(paired_y, paired_scores)
            entry["eer_vs_all_real"] = round(float(eer), 4)
            entry["detection_rate"] = round(float((group_scores >= threshold).mean()), 4)
        elif entry["n_real"] and entry["n_fake"] == 0:
            entry["false_alarm_rate"] = round(float((group_scores >= threshold).mean()), 4)
        else:
            eer, _ = compute_eer(group_y, group_scores)
            entry["eer"] = round(float(eer), 4)
        out[name] = entry
    return out


def _log_breakdown(result: dict) -> None:
    log.info("Chi tiết theo generator:")
    for name, entry in result["by_generator"].items():
        if "eer_vs_all_real" in entry:
            log.info(
                "  %-40s n=%4d · EER %.2f%% · bắt được %.1f%%",
                name, entry["n"], entry["eer_vs_all_real"] * 100, entry["detection_rate"] * 100,
            )
        elif "false_alarm_rate" in entry:
            log.info("  %-40s n=%4d · báo nhầm %.1f%%", name, entry["n"], entry["false_alarm_rate"] * 100)
    for name, entry in result["by_condition"].items():
        log.info("  điều kiện %-10s n=%4d · mean score %.3f", name, entry["n"], entry["mean_score"])


def _write_predictions(path: Path, records, y, scores, threshold) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["utt_id", "label", "score", "predicted", "correct",
                         "generator", "source", "speaker", "augment", "duration"])
        for rec, truth, score in zip(records, y, scores):
            predicted = LABEL_FAKE if score >= threshold else LABEL_REAL
            writer.writerow([
                rec.utt_id, rec.label, round(float(score), 6), predicted,
                int((score >= threshold) == bool(truth)),
                rec.generator, rec.source, rec.speaker, rec.augment, rec.duration,
            ])


# -------------------------------------------------------------------- biểu đồ
def _plots(y, scores, threshold, report_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay, precision_recall_curve, roc_curve

    fpr, tpr, _ = roc_curve(y, scores)
    fnr = 1 - tpr
    eer, _ = compute_eer(y, scores)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    axes[0, 0].plot(fpr, tpr, lw=2)
    axes[0, 0].plot([0, 1], [0, 1], "--", lw=1, color="grey")
    axes[0, 0].set(title="ROC", xlabel="False positive rate", ylabel="True positive rate")

    precision, recall, _ = precision_recall_curve(y, scores)
    axes[0, 1].plot(recall, precision, lw=2)
    axes[0, 1].set(title="Precision–Recall", xlabel="Recall", ylabel="Precision")

    axes[1, 0].plot(fpr, fnr, lw=2)
    axes[1, 0].plot([0, 1], [0, 1], "--", lw=1, color="grey")
    axes[1, 0].scatter([eer], [eer], color="red", zorder=5, label=f"EER = {eer * 100:.2f}%")
    axes[1, 0].set(title="DET", xlabel="False alarm", ylabel="Miss", xscale="log", yscale="log")
    axes[1, 0].legend()

    axes[1, 1].hist(scores[y == 0], bins=40, alpha=0.6, label="real")
    axes[1, 1].hist(scores[y == 1], bins=40, alpha=0.6, label="fake")
    axes[1, 1].axvline(threshold, color="red", ls="--", label=f"ngưỡng {threshold:.3f}")
    axes[1, 1].set(title="Phân bố điểm", xlabel="P(fake)", ylabel="Số mẫu")
    axes[1, 1].legend()

    fig.tight_layout()
    fig.savefig(report_dir / "curves.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4.5))
    ConfusionMatrixDisplay.from_predictions(
        y, (scores >= threshold).astype(int), display_labels=["real", "fake"], ax=ax, colorbar=False
    )
    ax.set_title("Confusion matrix @ ngưỡng EER")
    fig.tight_layout()
    fig.savefig(report_dir / "confusion_matrix.png", dpi=130)
    plt.close(fig)
    log.info("Đã lưu biểu đồ vào %s", report_dir)
