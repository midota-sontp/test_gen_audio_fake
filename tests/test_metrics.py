"""Số đo + ngưỡng quyết định."""

from __future__ import annotations

import numpy as np
import pytest

from aidetector.metrics import compute_eer, decision_threshold, full_report


def test_eer_is_zero_when_perfectly_separable():
    y = np.array([0, 0, 0, 1, 1, 1])
    scores = np.array([0.01, 0.02, 0.03, 0.97, 0.98, 0.99])
    eer, _ = compute_eer(y, scores)
    assert eer == 0.0


def test_eer_is_half_for_random_scores():
    rng = np.random.default_rng(0)
    y = np.repeat([0, 1], 500)
    eer, _ = compute_eer(y, rng.random(1000))
    assert 0.4 < eer < 0.6


def test_decision_threshold_sits_between_classes_when_separable():
    """Khi hai lớp tách hoàn hảo, ngưỡng tại EER dính sát mẫu fake thấp nhất."""
    y = np.array([0, 0, 1, 1])
    scores = np.array([0.10, 0.20, 0.95, 0.99])
    _, eer_threshold = compute_eer(y, scores)
    threshold = decision_threshold(y, scores)

    # Không còn biên an toàn: chỉ cần thấp hơn 0.95 một chút là bị coi là REAL.
    assert eer_threshold == pytest.approx(0.95)
    assert (0.949 >= eer_threshold) is False
    # Ngưỡng robust nằm giữa hai lớp nên có biên về cả hai phía.
    assert 0.20 < threshold < 0.95
    assert 0.949 >= threshold
    assert list((scores >= threshold).astype(int)) == list(y)


def test_decision_threshold_falls_back_to_half_on_degenerate_input():
    y = np.array([0, 0, 1, 1])
    assert decision_threshold(y, np.full(4, 0.7)) == 0.5


def test_full_report_uses_robust_threshold_by_default():
    y = np.array([0, 0, 0, 1, 1, 1])
    scores = np.array([0.05, 0.06, 0.07, 0.90, 0.95, 0.99])
    report = full_report(y, scores)
    assert report["eer"] == 0.0
    assert report["accuracy"] == 1.0
    assert scores[y == 0].max() < report["threshold"] < scores[y == 1].min()
    assert report["confusion_matrix"] == {"tn": 3, "fp": 0, "fn": 0, "tp": 3}


def test_full_report_respects_explicit_threshold():
    y = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.6, 0.7, 0.8])
    report = full_report(y, scores, threshold=0.65)
    assert report["threshold"] == 0.65
    assert report["confusion_matrix"]["fp"] == 0
    assert report["false_alarm_rate"] == 0.0
