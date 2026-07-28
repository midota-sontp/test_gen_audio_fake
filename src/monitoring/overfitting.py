"""Overfitting analysis from the per-epoch training history."""
from __future__ import annotations

from typing import Any


def analyze(history: list[dict], cfg: dict | None = None) -> dict[str, Any]:
    """history: list of per-epoch dicts with train_loss / val_loss.

    Returns {level, gap, messages}. Levels: EXCELLENT/GOOD/WARNING/CRITICAL/UNKNOWN.
    """
    cfg = cfg or {}
    exc = float(cfg.get("excellent", 0.05))
    good = float(cfg.get("good", 0.15))
    warn = float(cfg.get("warning", 0.30))
    win = int(cfg.get("trend_window", 3))

    rows = [h for h in history if h.get("train_loss") is not None and h.get("val_loss") is not None]
    if not rows:
        return {"level": "UNKNOWN", "gap": None, "messages": ["No history yet."]}

    last = rows[-1]
    gap = float(last["val_loss"]) - float(last["train_loss"])
    messages: list[str] = []

    if gap < exc:
        level = "EXCELLENT"
    elif gap < good:
        level = "GOOD"
    elif gap < warn:
        level = "WARNING"
        messages.append("Generalization gap elevated — consider more dropout / weight_decay.")
    else:
        level = "CRITICAL"
        messages.append("Large generalization gap — regularize harder or stop training.")

    # trend: val loss rising while train loss keeps falling over the last `win` epochs
    if len(rows) >= win + 1:
        window = rows[-(win + 1):]
        val_rising = all(
            window[i]["val_loss"] > window[i - 1]["val_loss"] for i in range(1, len(window))
        )
        train_falling = all(
            window[i]["train_loss"] < window[i - 1]["train_loss"] for i in range(1, len(window))
        )
        if val_rising and train_falling:
            messages.append(
                f"Val loss rising while train loss falls over last {win} epochs — overfitting."
            )
            if level in ("EXCELLENT", "GOOD"):
                level = "WARNING"

    return {"level": level, "gap": round(gap, 4), "messages": messages}
