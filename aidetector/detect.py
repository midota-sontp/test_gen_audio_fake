"""Suy luận trên file audio bất kỳ.

File đầu vào đi qua ĐÚNG chuỗi chuẩn hoá như lúc huấn luyện (16 kHz, mono, trim,
chuẩn mức, cắt 3–10 giây). File dài hơn 10 giây được chấm điểm từng đoạn rồi lấy
trung bình, kèm điểm chi tiết từng đoạn.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from .corpus.spec import AudioSpec, load_audio, normalize_file
from .features.backbones import Backbone, build_backbone
from .models import load_checkpoint
from .utils import get_logger, resolve_device

log = get_logger("aidetector.detect")


class Detector:
    """Bọc backbone + head thành một API gọn để dùng lại nhiều lần."""

    def __init__(
        self,
        checkpoint: str | Path = "checkpoints/best.pt",
        device: str = "auto",
        spec: AudioSpec | None = None,
        threshold: float | None = None,
        allow_short: bool = True,
    ) -> None:
        self.device = resolve_device(device)
        self.head, self.meta = load_checkpoint(checkpoint, self.device)
        self.backbone: Backbone = build_backbone(dict(self.meta.get("backbone", {})), self.device)
        self.spec = spec or AudioSpec.from_config(self.meta.get("audio"))
        # Lúc suy luận, file ngắn hơn chuẩn được đệm im lặng thay vì bị từ chối —
        # audio thật ngoài đời hay ngắn, và một dự đoán kèm cảnh báo vẫn hữu ích
        # hơn là không có gì.
        self.allow_short = allow_short
        if allow_short and self.spec.short_policy != "pad":
            self.spec = replace(self.spec, short_policy="pad")
        self.mean = np.asarray(self.meta["norm_mean"], dtype=np.float32)
        self.std = np.asarray(self.meta["norm_std"], dtype=np.float32)
        # Ngưỡng mặc định lấy từ EER trên tập val lúc huấn luyện.
        self.threshold = float(threshold if threshold is not None else self.meta.get("threshold", 0.5))
        log.info(
            "Detector sẵn sàng · backbone=%s · device=%s · ngưỡng=%.3f",
            self.backbone.checkpoint, self.device, self.threshold,
        )

    def score_chunks(self, chunks: list[np.ndarray]) -> np.ndarray:
        if not chunks:
            return np.zeros((0,), dtype=np.float32)
        embeddings = self.backbone.embed(chunks)
        with torch.no_grad():
            logits = self.head(torch.from_numpy((embeddings - self.mean) / self.std).to(self.device))
            return torch.sigmoid(logits).cpu().numpy().reshape(-1)

    def predict(self, path: str | Path) -> dict:
        try:
            raw_seconds = len(load_audio(path, self.spec.sample_rate)) / self.spec.sample_rate
        except Exception as exc:  # noqa: BLE001
            return {"path": str(path), "error": f"Không đọc được audio: {exc}"}

        chunks = normalize_file(path, self.spec)
        if not chunks:
            return {
                "path": str(path),
                "error": f"Audio dài {raw_seconds:.2f}s — không còn nội dung hợp lệ "
                         "sau khi cắt im lặng",
            }
        scores = self.score_chunks(chunks)
        mean_score = float(scores.mean())
        result = {
            "path": str(path),
            "score_fake": round(mean_score, 4),
            "label": "FAKE" if mean_score >= self.threshold else "REAL",
            "confidence": round(float(abs(mean_score - self.threshold) / max(self.threshold, 1 - self.threshold)), 4),
            "threshold": self.threshold,
            "n_chunks": len(chunks),
            "chunk_scores": [round(float(s), 4) for s in scores],
            "duration": round(raw_seconds, 2),
        }
        if raw_seconds < self.spec.min_seconds:
            result["warning"] = (
                f"Audio chỉ {raw_seconds:.2f}s, ngắn hơn chuẩn huấn luyện "
                f"{self.spec.min_seconds:g}s — đã đệm im lặng, kết quả kém tin cậy."
            )
        return result

    def predict_many(self, paths: list[str | Path]) -> list[dict]:
        return [self.predict(p) for p in paths]
