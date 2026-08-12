"""Tầng sinh audio giả — engine TTS / voice-cloning cắm rời.

Thêm engine mới = thêm một file trong `aidetector/generate/` với lớp con của
`Generator` được đánh dấu `@register`, rồi khai báo tên engine trong
`configs/default.yaml` mục `generate.engines`. Không cần sửa chỗ nào khác.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ..utils import get_logger

log = get_logger("aidetector.generate")

KIND_TTS = "tts"        # giọng cố định, chỉ cần text
KIND_CLONE = "clone"    # zero-shot voice cloning, cần audio tham chiếu


@dataclass
class Availability:
    ok: bool
    reason: str = ""
    hint: str = ""

    def __bool__(self) -> bool:
        return self.ok


class Generator:
    """Lớp cơ sở cho mọi engine sinh audio giả."""

    id: str = "base"
    kind: str = KIND_TTS
    description: str = ""
    #: engine có bắt buộc GPU không (CPU vẫn chạy được nhưng rất chậm)
    prefers_gpu: bool = False
    #: sample rate gốc của engine, dùng để resample về chuẩn corpus
    native_sample_rate: int = 22_050

    def __init__(self, device: str = "cpu", **options) -> None:
        self.device = device
        self.options = options
        self._loaded = False

    # ------------------------------------------------------------- vòng đời
    @classmethod
    def availability(cls) -> Availability:
        """Kiểm tra thư viện/checkpoint đã sẵn sàng chưa (không tải model)."""
        return Availability(True)

    def load(self) -> None:
        """Nạp model vào bộ nhớ. Được gọi một lần trước lô sinh đầu tiên."""
        self._loaded = True

    def unload(self) -> None:
        self._loaded = False

    def ensure_loaded(self) -> None:
        if not self._loaded:
            self.load()

    # -------------------------------------------------------------- sinh audio
    def voices(self) -> Sequence[str]:
        """Danh sách giọng khả dụng. Engine cloning trả về [] (giọng do reference quyết định)."""
        return []

    def synthesize(
        self,
        text: str,
        voice: str | None = None,
        ref_audio: str | None = None,
        ref_text: str | None = None,
    ) -> tuple[np.ndarray, int]:
        """Sinh audio → (mảng float32 mono, sample_rate). Ném lỗi nếu thất bại."""
        raise NotImplementedError

    # ------------------------------------------------------------------ tiện ích
    def tag(self, voice: str | None) -> str:
        """Chuỗi ghi vào cột `generator` của manifest: `piper:vi_VN-vais1000-medium`."""
        return f"{self.id}:{voice}" if voice else self.id

    def __repr__(self) -> str:  # pragma: no cover
        return f"<{type(self).__name__} id={self.id} kind={self.kind} device={self.device}>"


_REGISTRY: dict[str, type[Generator]] = {}


def register(cls: type[Generator]) -> type[Generator]:
    if cls.id in _REGISTRY:
        raise ValueError(f"Generator trùng id: {cls.id}")
    _REGISTRY[cls.id] = cls
    return cls


def get_generator(engine_id: str) -> type[Generator]:
    if engine_id not in _REGISTRY:
        raise KeyError(
            f"Không có engine {engine_id!r}. Hiện có: {', '.join(sorted(_REGISTRY))}"
        )
    return _REGISTRY[engine_id]


def available_generators() -> dict[str, type[Generator]]:
    return dict(_REGISTRY)
