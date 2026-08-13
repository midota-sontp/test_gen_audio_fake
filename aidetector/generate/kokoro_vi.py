"""Kokoro tiếng Việt — bản fine-tune cộng đồng của Kokoro-82M cho tiếng Việt.

Kokoro gốc (hexgrad/kokoro) CHƯA hỗ trợ tiếng Việt; bản dùng ở đây là
`iamdinhthuan/Kokoro-Vietnamese` (G2P tiếng Việt bằng vig2p, 13 giọng),
checkpoint trên HF: `contextboxai/Kokoro-Vietnamese`.

Cài:
    pip install git+https://github.com/iamdinhthuan/Kokoro-Vietnamese.git
    # tuỳ chọn, chạy ONNX cho nhanh trên CPU:
    pip install "kokoro-vietnamese[onnx]"
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from ..utils import get_logger
from .base import (
    KIND_TTS,
    Availability,
    Generator,
    check_transformers_range,
    is_installed,
    register,
)

log = get_logger("aidetector.generate.kokoro_vi")

DEFAULT_VOICES = (
    "diem_trinh", "hung_thinh", "mai_linh", "mai_loan", "manh_dung",
    "my_yen", "ngoc_huyen", "phat_tai", "thanh_dat", "thuc_trinh",
    "tuan_ngoc", "duc_an", "duc_duy",
)


@register
class KokoroVietnameseGenerator(Generator):
    id = "kokoro"
    kind = KIND_TTS
    description = "Kokoro-Vietnamese (82M, 13 giọng vi) — chạy được trên CPU"
    prefers_gpu = False
    native_sample_rate = 24_000

    def __init__(self, device: str = "cpu", **options) -> None:
        super().__init__(device, **options)
        self._voice_names = list(options.get("voices") or DEFAULT_VOICES)
        self._engines: dict[str, object] = {}

    @classmethod
    def availability(cls) -> Availability:
        if not is_installed("kokoro_vietnamese"):
            return Availability(
                False,
                "chưa cài kokoro-vietnamese",
                "pip install git+https://github.com/iamdinhthuan/Kokoro-Vietnamese.git",
            )
        # Kokoro chạy trên transformers 4.x, OmniVoice cần >=5.3 — hai engine này
        # không sống chung được, phải sinh fake thành hai lượt.
        conflict = check_transformers_range("kokoro-vietnamese", below_major=5)
        if conflict is not None:
            return conflict
        try:
            import kokoro_vietnamese  # noqa: F401
        except Exception as exc:  # noqa: BLE001
            return Availability(False, f"kokoro-vietnamese lỗi khi import: {exc}", "")
        return Availability(True)

    def voices(self) -> Sequence[str]:
        return list(self._voice_names)

    def _engine(self, voice: str):
        if voice not in self._engines:
            from kokoro_vietnamese import KokoroVietnamese

            # Thư viện chỉ hỗ trợ cuda/cpu; MPS chưa được nhận ⇒ hạ về cpu.
            device = self.device if self.device in ("cuda", "cpu") else "cpu"
            log.info("Nạp Kokoro-VI voice=%s device=%s", voice, device)
            self._engines[voice] = KokoroVietnamese(device=device, voice=voice)
        return self._engines[voice]

    def load(self) -> None:
        self._engine(self._voice_names[0])
        self._loaded = True

    def synthesize(
        self,
        text: str,
        voice: str | None = None,
        ref_audio: str | None = None,
        ref_text: str | None = None,
        language: str | None = None,   # bỏ qua: model chỉ có tiếng Việt
    ) -> tuple[np.ndarray, int]:
        name = voice or self._voice_names[0]
        result = self._engine(name).synthesize(text)
        # API trả (audio, phonemes); một số bản chỉ trả audio.
        audio = result[0] if isinstance(result, tuple) else result
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        return audio, self.native_sample_rate
