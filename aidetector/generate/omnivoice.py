"""OmniVoice — zero-shot voice cloning (k2-fsa/OmniVoice, 600+ ngôn ngữ).

Đây là engine CLONING: cần một đoạn audio tham chiếu 3–25 giây của người nói thật
(+ transcript của đoạn đó), rồi đọc text mới bằng đúng giọng đó. Trong pipeline này
reference được lấy thẳng từ corpus REAL nên fake và real chia sẻ cùng danh tính
người nói — mô hình buộc phải học dấu vết tổng hợp thay vì học giọng ai.

Độ giống giọng phụ thuộc trước hết vào reference, không vào knob của engine: 3 giây
là mức chạy được chứ không phải mức đủ. `aidetector.generate` ghép nhiều utterance
cùng speaker để đạt ~12 giây (xem `TARGET_REF_SECONDS`).

Cài:  pip install omnivoice
Backend hỗ trợ CUDA, Apple Silicon (MPS) và Intel XPU; CPU thuần rất chậm.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from ..utils import get_logger, slugify
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
#: Lưu ý khi giọng clone nghe "lệch người": fine-tune một-ngôn-ngữ đọc tiếng Việt
#: chuẩn hơn nhưng thường clone danh tính KÉM hơn bản gốc (nó chỉ thấy lại tập
#: speaker hẹp khi fine-tune). Nếu độ giống giọng quan trọng hơn phát âm, thử
#: `--set generate.options.omnivoice.checkpoint=k2-fsa/OmniVoice` rồi nghe so sánh.
MULTILINGUAL_CHECKPOINT = "k2-fsa/OmniVoice"

#: Knob của bộ giải mã diffusion (OmniVoiceGenerationConfig). Không đặt thì để
#: nguyên mặc định của thư viện: num_step=32, guidance_scale=2.0, t_shift=0.1.
#: `guidance_scale` cao hơn ⇒ bám sát prompt hơn (gồm cả danh tính người nói),
#: `num_step` cao hơn ⇒ audio mịn hơn nhưng chậm hơn.
GENERATION_KEYS = ("num_step", "guidance_scale", "t_shift")


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
        self.generation = {k: options[k] for k in GENERATION_KEYS if k in options}
        # Sample rate THẬT do audio tokenizer của checkpoint quyết định; chỉ biết
        # được sau khi nạp model. Trước đó dùng giá trị mặc định làm chỗ dựa.
        self._sample_rate = self.native_sample_rate
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

    @property
    def variant(self) -> str:
        """Checkpoint khác mặc định thì phải lộ ra trong manifest, để A/B được."""
        if self.checkpoint == DEFAULT_CHECKPOINT:
            return ""
        return slugify(self.checkpoint, 32)

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
        # Sample rate do audio tokenizer của checkpoint quyết định, KHÔNG phải hằng số
        # của engine. Hardcode 24 kHz mà tokenizer chạy ở tần số khác thì bước resample
        # về chuẩn corpus sẽ dịch cả cao độ lẫn tốc độ — giọng clone nghe "lệch người"
        # dù model chẳng làm gì sai.
        rate = getattr(self._model, "sampling_rate", None)
        self._sample_rate = int(rate) if rate else self.native_sample_rate
        if self._sample_rate != self.native_sample_rate:
            log.info(
                "Audio tokenizer của %s chạy ở %d Hz (mặc định của engine là %d Hz)",
                self.checkpoint, self._sample_rate, self.native_sample_rate,
            )
        self._loaded = True

    @staticmethod
    def _lower_if_shouting(text: str) -> str:
        """Hạ TOÀN CHỮ HOA về chữ thường. Không đổi nội dung, chỉ đổi dạng.

        `normalize_text` của OmniVoice KHÔNG làm việc này. Đọc mã thư viện: với ngôn
        ngữ không phải zh/en, nó chỉ chạy `num2words` trên số nguyên rồi trả text về
        nguyên vẹn — chữ hoa đi thẳng vào model. Mà VIVOS lưu transcript TOÀN CHỮ HOA,
        và chuỗi chữ hoa là dạng text tokenizer gần như không thấy khi huấn luyện.
        """
        text = text.strip()
        if any(c.isalpha() for c in text) and not any(c.islower() for c in text):
            return text.lower()
        return text

    def _clean_text(self, text: str) -> str:
        """Text ĐÍCH: hạ chữ hoa và đảm bảo có dấu kết câu.

        Thư viện tự thêm dấu kết cho ref_text (`add_punctuation`) nhưng KHÔNG làm thế
        với text đích. Transcript VIVOS không có dấu chấm nào, nên model phải tự đoán
        chỗ kết thúc — cộng thêm chữ hoa nữa thì nó đọc đúng vài từ đầu rồi vỡ.
        """
        text = self._lower_if_shouting(text)
        if text and text[-1] not in ".!?…":
            text += "."
        return text

    def _clean_ref_text(self, ref_text: str | None) -> str | None:
        """Transcript của REFERENCE — thư viện cũng không chuẩn hoá cái này.

        `normalize_text` chỉ áp lên text đích, có chú thích rõ trong mã nguồn:
        *"not ref_text, which must stay aligned with the reference audio"*. Chữ hoa ở
        đây làm cặp (audio, text) của reference gióng hàng kém ⇒ danh tính giọng lấy
        ra được cũng nhoè theo.

        Trả `None` thay vì `""` khi không có transcript: `None` là tín hiệu để thư
        viện tự nhận dạng bằng ASR, còn `""` bị hiểu là "reference không nói gì" —
        đúng cách nhanh nhất để phá gióng hàng và mất luôn giọng cần clone.
        Dấu kết câu để thư viện tự thêm, nó đã có `add_punctuation` cho ref_text.
        """
        if not ref_text or not ref_text.strip():
            return None
        return self._lower_if_shouting(ref_text)

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
            text=self._clean_text(text),
            language=language or self.language,
            ref_audio=str(ref_audio),
            ref_text=self._clean_ref_text(ref_text),
            normalize_text=self.normalize_text,
            **self.generation,
        )
        audio = np.asarray(output[0] if isinstance(output, (list, tuple)) else output, dtype=np.float32)
        return audio.reshape(-1), self._sample_rate
