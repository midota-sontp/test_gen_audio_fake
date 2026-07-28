"""Classification metrics for the detector, including Equal Error Rate (EER)."""
from __future__ import annotations

from pathlib import Path

import numpy as np


def compute_eer(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    """EER + its threshold, where the fake/positive class score is `scores`.

    Returns (eer, threshold). EER is the point where FPR == FNR.
    """
    from sklearn.metrics import roc_curve

    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores, dtype=float)
    fpr, tpr, thr = roc_curve(labels, scores, pos_label=1)
    fnr = 1.0 - tpr
    # first index where fnr <= fpr (curves cross); interpolate for a smooth value
    idx = int(np.nanargmin(np.abs(fnr - fpr)))
    eer = float((fpr[idx] + fnr[idx]) / 2.0)
    return eer, float(thr[idx])


def compute_metrics(labels, scores, threshold: float = 0.5) -> dict:
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores, dtype=float)
    preds = (scores >= threshold).astype(int)

    # roc_auc / eer need both classes present
    both = len(np.unique(labels)) == 2
    eer, eer_thr = compute_eer(labels, scores) if both else (float("nan"), float("nan"))
    cm = confusion_matrix(labels, preds, labels=[0, 1]).tolist()
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, scores)) if both else float("nan"),
        "eer": eer,
        "eer_threshold": eer_thr,
        "confusion_matrix": cm,  # [[TN, FP], [FN, TP]]
        "n": int(len(labels)),
    }


def save_plots(labels, scores, out_dir: str | Path, threshold: float = 0.5) -> None:
    """ROC, PR and confusion-matrix figures. No-op-safe if only one class present."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import (
        ConfusionMatrixDisplay,
        confusion_matrix,
        precision_recall_curve,
        roc_curve,
    )

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores, dtype=float)
    preds = (scores >= threshold).astype(int)

    if len(np.unique(labels)) == 2:
        fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
        plt.figure()
        plt.plot(fpr, tpr, label="ROC")
        plt.plot([0, 1], [0, 1], "--", color="grey")
        plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC Curve"); plt.legend()
        plt.savefig(out / "roc.png", bbox_inches="tight", dpi=120); plt.close()

        prec, rec, _ = precision_recall_curve(labels, scores, pos_label=1)
        plt.figure()
        plt.plot(rec, prec, label="PR")
        plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Precision-Recall"); plt.legend()
        plt.savefig(out / "pr.png", bbox_inches="tight", dpi=120); plt.close()

    cm = confusion_matrix(labels, preds, labels=[0, 1])
    disp = ConfusionMatrixDisplay(cm, display_labels=["real", "fake"])
    disp.plot(cmap="Blues", values_format="d")
    plt.title("Confusion Matrix")
    plt.savefig(out / "confusion_matrix.png", bbox_inches="tight", dpi=120); plt.close()
