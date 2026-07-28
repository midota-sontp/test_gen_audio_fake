"""Unit tests for the EER / metrics computation."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluator.metrics import compute_eer, compute_metrics  # noqa: E402


def test_eer_perfectly_separable():
    labels = np.array([0, 0, 0, 1, 1, 1])
    scores = np.array([0.01, 0.02, 0.10, 0.90, 0.95, 0.99])  # fake scores clearly higher
    eer, _ = compute_eer(labels, scores)
    assert eer < 0.02, f"expected ~0 EER, got {eer}"


def test_eer_random_is_near_half():
    rng = np.random.RandomState(0)
    labels = np.array([0] * 500 + [1] * 500)
    scores = rng.rand(1000)  # no signal
    eer, _ = compute_eer(labels, scores)
    assert 0.4 < eer < 0.6, f"expected ~0.5 EER, got {eer}"


def test_metrics_keys_present():
    labels = np.array([0, 1, 0, 1])
    scores = np.array([0.2, 0.8, 0.3, 0.7])
    m = compute_metrics(labels, scores, threshold=0.5)
    for k in ["accuracy", "precision", "recall", "f1", "roc_auc", "eer",
              "confusion_matrix", "n"]:
        assert k in m
    assert m["accuracy"] == 1.0


if __name__ == "__main__":
    test_eer_perfectly_separable()
    test_eer_random_is_near_half()
    test_metrics_keys_present()
    print("All EER/metrics tests passed.")
