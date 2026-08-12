"""OmniVoice — zero-shot voice cloning (k2-fsa/OmniVoice, 600+ ngôn ngữ).

Đây là engine CLONING: cần một đoạn audio tham chiếu 3–25 giây của người nói thật
(+ transcript của đoạn đó), rồi đọc text mới bằng đúng giọng đó. Trong pipeline này
reference được lấy thẳng từ corpus REAL nên fake và real chia sẻ cùng danh tính
người nói — mô hình buộc phải học dấu vết tổng hợp thay vì học giọng ai.

Cài:  pip install omnivoice
Backend hỗ trợ CUDA, Apple Silicon (MPS) và Intel XPU; CPU thuần rất chậm.
Bản fine-tune riêng cho tiếng Việt: `g-group-ai-lab/g-omnivoice`.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from ..utils import get_logger
from .base import KIND_CLONE, Availability, Generator, register

log = get_logger("aidetector.generate.omnivoice")

DEFAULT_CHECKPOINT = "k2-fsa/OmniVoice"


@register
class OmniVoiceGenerator(Generator):
    id = "omnivoice"
    kind = KIND_CLONE
    description = "OmniVoice zero-shot voice cloning (cần audio tham chiếu; nên có GPU/MPS)"
    prefers_gpu = True
    native_sample_rate = 24_000

    def __init__(self, device: str = "cpu", **options) -> None:
        super().__init__(device, **options)
        self.checkpoint = options.get("checkpoint", DEFAULT_CHECKPOINT)
        self.dtype = options.get("dtype", "auto")
        self._model = None

    @classmethod
    def availability(cls) -> Availability:
        try:
            import omnivoice  # noqa: F401
        except ImportError:
            return Availability(False, "chưa cài omnivoice", "pip install omnivoice")
        return Availability(True)

    def voices(self) -> Sequence[str]:
        return []  # giọng do reference quyết định

    def load(self) -> None:
        import torch
        from omnivoice import OmniVoice

        if self.device == "cpu":
            log.warning("OmniVoice chạy trên CPU sẽ rất chậm — cân nhắc GPU/MPS.")
        dtype = (
            torch.float16 if self.dtype == "float16"
            else torch.float32 if self.dtype == "float32"
            # MPS/CPU ổn định hơn với float32; CUDA dùng float16 cho nhanh.
            else (torch.float16 if self.device.startswith("cuda") else torch.float32)
        )
        log.info("Nạp OmniVoice %s (device=%s, dtype=%s)", self.checkpoint, self.device, dtype)
        self._model = OmniVoice.from_pretrained(
            self.checkpoint, device_map=self.device, dtype=dtype
        )
        self._loaded = True

    def synthesize(
        self,
        text: str,
        voice: str | None = None,
        ref_audio: str | None = None,
        ref_text: str | None = None,
    ) -> tuple[np.ndarray, int]:
        if not ref_audio:
            raise ValueError("OmniVoice cần `ref_audio` (đoạn giọng thật để clone)")
        self.ensure_loaded()
        assert self._model is not None
        output = self._model.generate(text=text, ref_audio=str(ref_audio), ref_text=ref_text or "")
        audio = np.asarray(output[0] if isinstance(output, (list, tuple)) else output, dtype=np.float32)
        return audio.reshape(-1), self.native_sample_rate
