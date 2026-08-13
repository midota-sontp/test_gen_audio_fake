"""OmniVoice — zero-shot voice cloning (k2-fsa/OmniVoice, 600+ ngôn ngữ).

Đây là engine CLONING: cần một đoạn audio tham chiếu 3–25 giây của người nói thật
(+ transcript của đoạn đó), rồi đọc text mới bằng đúng giọng đó. Trong pipeline này
reference được lấy thẳng từ corpus REAL nên fake và real chia sẻ cùng danh tính
người nói — mô hình buộc phải học dấu vết tổng hợp thay vì học giọng ai.

Cài:  pip install omnivoice
Backend hỗ trợ CUDA, Apple Silicon (MPS) và Intel XPU; CPU thuần rất chậm.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from ..utils import get_logger
from .base import (
    KIND_CLONE,
    Availability,
    Generator,
    check_transformers_range,
    is_installed,
    register,
)

log = get_logger("aidetector.generate.omnivoice")

#: Bản fine-tune tiếng Việt, repo CÔNG KHAI nên tải được ngay không cần token.
#: Nó không kèm thư mục `audio_tokenizer/`, nhưng thư viện tự bù bằng
#: `eustlb/higgs-audio-v2-tokenizer` (cũng công khai) nên vẫn nạp được bình thường.
#: Bản gốc đa ngữ `k2-fsa/OmniVoice` đọc tiếng Việt kém hơn rõ rệt.
DEFAULT_CHECKPOINT = "splendor1811/omnivoice-vietnamese"
#: Bản gốc đa ngữ 600+ ngôn ngữ — dùng khi cần ngôn ngữ khác tiếng Việt.
MULTILINGUAL_CHECKPOINT = "k2-fsa/OmniVoice"


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
        # Ngôn ngữ mặc định khi bản ghi không ghi rõ. Model phủ 600+ ngôn ngữ nên
        # để nó tự đoán là cách nhanh nhất để ra audio "không giống tiếng Việt".
        self.language = options.get("language", "vi")
        # VIVOS và nhiều corpus khác lưu transcript TOÀN CHỮ HOA; bật chuẩn hoá để
        # model không đọc chúng như chuỗi chữ cái rời rạc.
        self.normalize_text = bool(options.get("normalize_text", True))
        self._model = None

    #: OmniVoice 0.2.x dùng HiggsAudioV2TokenizerModel, chỉ có từ transformers 5.3.
    MIN_TRANSFORMERS = (5, 3)

    @classmethod
    def availability(cls) -> Availability:
        if not is_installed("omnivoice"):
            return Availability(False, "chưa cài omnivoice", "pip install omnivoice")
        # Ngược chiều với Kokoro: engine này cần transformers 5.x. Báo rõ ở đây để
        # người dùng biết phải sinh fake thành hai lượt chứ không phải engine hỏng.
        conflict = check_transformers_range("omnivoice", minimum=cls.MIN_TRANSFORMERS)
        if conflict is not None:
            return conflict
        try:
            import omnivoice  # noqa: F401
        except Exception as exc:  # noqa: BLE001
            return Availability(False, f"omnivoice lỗi khi import: {exc}", "")
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
        try:
            self._model = OmniVoice.from_pretrained(
                self.checkpoint, device_map=self.device, dtype=dtype
            )
        except Exception as exc:  # noqa: BLE001 — cần dịch lỗi HF sang việc phải làm
            raise RuntimeError(self._explain_load_failure(exc)) from exc
        self._loaded = True

    def _explain_load_failure(self, exc: Exception) -> str:
        """Biến lỗi tải checkpoint thành việc người dùng có thể làm ngay."""
        message = str(exc)
        if "gated" in message or "401" in message or "restricted" in message:
            return (
                f"Checkpoint {self.checkpoint!r} là repo GATED trên HuggingFace — phải xin "
                f"quyền rồi đăng nhập mới tải được.\n"
                f"  → Cách nhanh nhất: dùng bản công khai bằng cách thêm\n"
                f"      --set generate.options.omnivoice.checkpoint={DEFAULT_CHECKPOINT}\n"
                f"  → Hoặc: mở https://huggingface.co/{self.checkpoint} bấm xin quyền, tạo\n"
                f"    token ở https://huggingface.co/settings/tokens rồi đặt biến môi trường\n"
                f"    HF_TOKEN (trên Kaggle: Add-ons → Secrets).\n"
                f"Lỗi gốc: {message.splitlines()[0]}"
            )
        if "out of memory" in message.lower() or "CUDA" in message:
            return (
                f"Không đủ bộ nhớ GPU để nạp {self.checkpoint!r}. Thử dtype float16 "
                f"(--set generate.options.omnivoice.dtype=float16) hoặc giảm --count.\n"
                f"Lỗi gốc: {message.splitlines()[0]}"
            )
        return f"Không nạp được checkpoint {self.checkpoint!r}: {message}"

    def synthesize(
        self,
        text: str,
        voice: str | None = None,
        ref_audio: str | None = None,
        ref_text: str | None = None,
        language: str | None = None,
    ) -> tuple[np.ndarray, int]:
        if not ref_audio:
            raise ValueError("OmniVoice cần `ref_audio` (đoạn giọng thật để clone)")
        self.ensure_loaded()
        assert self._model is not None
        output = self._model.generate(
            text=text,
            language=language or self.language,
            ref_audio=str(ref_audio),
            ref_text=ref_text or "",
            normalize_text=self.normalize_text,
        )
        audio = np.asarray(output[0] if isinstance(output, (list, tuple)) else output, dtype=np.float32)
        return audio.reshape(-1), self.native_sample_rate
