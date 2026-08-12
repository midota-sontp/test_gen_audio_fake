"""Số đo cho bài toán phát hiện giọng giả.

**EER** (Equal Error Rate) là số đo chính: điểm mà tỉ lệ báo nhầm real thành fake
bằng tỉ lệ bỏ sót fake. Không phụ thuộc ngưỡng nên so sánh giữa các lần chạy được.
"""

from __future__ import annotations

import numpy as np


def compute_eer(y_true: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    """Trả (EER, ngưỡng tại EER). `y_true`: 1 = fake, `scores`: điểm càng cao càng fake."""
    from sklearn.metrics import roc_curve

    if len(np.unique(y_true)) < 2:
        return float("nan"), float("nan")
    fpr, tpr, thresholds = roc_curve(y_true, scores, pos_label=1)
    fnr = 1 - tpr
    idx = int(np.nanargmin(np.abs(fnr - fpr)))
    return float((fpr[idx] + fnr[idx]) / 2), float(thresholds[idx])


def decision_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Ngưỡng dùng để RA QUYẾT ĐỊNH lúc suy luận.

    Ngưỡng tại EER đúng về mặt toán học nhưng **không có biên an toàn** khi hai lớp
    tách hoàn hảo: nó rơi đúng vào điểm của mẫu fake "ít fake nhất" (vd 1.000 khi
    mọi fake đều ~1.0). Mọi audio chấm thấp hơn mẫu đó — kể cả 0.96 — sẽ bị xếp là
    REAL. Trường hợp này lấy trung điểm giữa hai lớp để có biên đối xứng; các đầu
    vào bất thường khác quay về 0.5.
    """
    real, fake = scores[y_true == 0], scores[y_true == 1]
    if len(real) and len(fake) and real.max() < fake.min():
        return float((real.max() + fake.min()) / 2)
    _, threshold = compute_eer(y_true, scores)
    if not np.isfinite(threshold) or not scores.min() < threshold < scores.max():
        return 0.5
    return float(threshold)


def min_dcf(
    y_true: np.ndarray, scores: np.ndarray,
    p_target: float = 0.05, c_miss: float = 1.0, c_fa: float = 1.0,
) -> float:
    """min-DCF chuẩn hoá — số đo phụ dùng trong các thử thách ASVspoof."""
    from sklearn.metrics import roc_curve

    if len(np.unique(y_true)) < 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(y_true, scores, pos_label=1)
    fnr = 1 - tpr
    dcf = c_miss * fnr * p_target + c_fa * fpr * (1 - p_target)
    return float(np.min(dcf) / min(c_miss * p_target, c_fa * (1 - p_target)))


def classification_metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    y_pred = (scores >= threshold).astype(int)
    both_classes = len(np.unique(y_true)) > 1
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)) if both_classes else float("nan"),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, scores)) if both_classes else float("nan"),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        # false_alarm: real bị gán nhãn fake · miss: fake lọt qua
        "false_alarm_rate": float(fp / max(tn + fp, 1)),
        "miss_rate": float(fn / max(fn + tp, 1)),
    }


def full_report(y_true: np.ndarray, scores: np.ndarray, threshold: float | None = None) -> dict:
    """Bộ số đo đầy đủ. `threshold=None` ⇒ dùng ngưỡng quyết định robust."""
    eer, eer_threshold = compute_eer(y_true, scores)
    used = decision_threshold(y_true, scores) if threshold is None else threshold
    return {
        "eer": eer,
        "eer_threshold": eer_threshold,
        "min_dcf": min_dcf(y_true, scores),
        "n_samples": int(len(y_true)),
        "n_real": int(np.sum(y_true == 0)),
        "n_fake": int(np.sum(y_true == 1)),
        **classification_metrics(y_true, scores, used),
    }
