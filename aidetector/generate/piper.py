"""Piper — TTS neural chạy ONNX, rất nhanh trên CPU (hợp máy Mac/Apple Silicon).

Cài:  pip install piper-tts
Tải giọng tiếng Việt:
    python -m piper.download_voices vi_VN-vais1000-medium --data-dir models/piper
Các giọng vi_VN có sẵn trên `rhasspy/piper-voices`:
    vi_VN-vais1000-medium, vi_VN-25hours_single-low, vi_VN-vivos-x_low
"""

from __future__ import annotations

import io
import wave
from pathlib import Path
from typing import Sequence

import numpy as np

from ..utils import ensure_dir, get_logger
from .base import KIND_TTS, Availability, Generator, register

log = get_logger("aidetector.generate.piper")

#: Chỉ liệt kê giọng mã hoá được thanh điệu. Hai giọng vi_VN còn lại trên
#: `rhasspy/piper-voices` (`vi_VN-25hours_single-low`, `vi_VN-vivos-x_low`) có bảng
#: phoneme KHÔNG chứa ký hiệu thanh điệu nên đọc tiếng Việt mất dấu — xem
#: `_drop_toneless_voices`. Muốn dùng thì phải tắt check_tones một cách tường minh.
DEFAULT_VOICES = ("vi_VN-vais1000-medium",)
DEFAULT_DATA_DIR = Path("models/piper")

#: Ngôn ngữ có thanh điệu — thiếu ký hiệu thanh là lỗi nghiêm trọng, không phải tiểu tiết.
TONAL_LANGUAGES = {"vi", "zh", "th", "yue", "cmn", "lo", "my"}
#: eSpeak biểu diễn thanh điệu bằng chữ số trong chuỗi phoneme.
TONE_SYMBOLS = tuple("12345678")


@register
class PiperGenerator(Generator):
    id = "piper"
    kind = KIND_TTS
    description = "Piper TTS (ONNX, CPU nhanh) — giọng vi_VN cố định"
    prefers_gpu = False
    native_sample_rate = 22_050

    def __init__(self, device: str = "cpu", **options) -> None:
        super().__init__(device, **options)
        self.data_dir = Path(options.get("data_dir", DEFAULT_DATA_DIR))
        self._voice_names: list[str] = list(options.get("voices") or DEFAULT_VOICES)
        # Mặc định BẬT: audio mất thanh điệu làm hỏng dataset một cách âm thầm.
        self.check_tones = bool(options.get("check_tones", True))
        self._models: dict[str, object] = {}

    @classmethod
    def availability(cls) -> Availability:
        try:
            import piper  # noqa: F401
        except ImportError:
            return Availability(False, "chưa cài piper-tts", "pip install piper-tts")
        return Availability(True)

    def voices(self) -> Sequence[str]:
        return list(self._voice_names)

    # ------------------------------------------------------------------ nạp model
    @staticmethod
    def _hub_paths(name: str) -> tuple[str, str]:
        """`vi_VN-vais1000-medium` → đường dẫn .onnx và .onnx.json trong repo HF.

        Cấu trúc repo `rhasspy/piper-voices`: <nhóm-ngữ>/<mã-ngôn-ngữ>/<tên>/<chất-lượng>/
        """
        try:
            lang, voice_name, quality = name.split("-", 2)
        except ValueError as exc:
            raise ValueError(
                f"Tên giọng Piper phải có dạng <lang>-<name>-<quality>, nhận {name!r}"
            ) from exc
        base = f"{lang.split('_')[0]}/{lang}/{voice_name}/{quality}/{name}.onnx"
        return base, base + ".json"

    def _voice_path(self, name: str) -> Path:
        """Tìm file .onnx của giọng, tải về nếu chưa có."""
        for path in [self.data_dir / f"{name}.onnx", *self.data_dir.rglob(f"{name}.onnx")]:
            if path.exists():
                return path

        ensure_dir(self.data_dir)
        log.info("Tải giọng Piper %s về %s ...", name, self.data_dir)
        # Ưu tiên huggingface_hub: có retry, cache, và dùng chứng chỉ của certifi
        # (bộ tải bằng urllib của piper hay lỗi SSL trên macOS).
        try:
            from huggingface_hub import hf_hub_download

            onnx_rel, config_rel = self._hub_paths(name)
            for rel in (config_rel, onnx_rel):
                src = Path(hf_hub_download("rhasspy/piper-voices", rel))
                dst = self.data_dir / src.name
                if not dst.exists():
                    dst.write_bytes(src.read_bytes())
            return self.data_dir / f"{name}.onnx"
        except Exception as exc:  # noqa: BLE001
            log.warning("Tải qua HuggingFace Hub thất bại (%s) — thử bộ tải của piper", exc)

        from piper.download_voices import download_voice

        download_voice(name, self.data_dir)
        found = list(self.data_dir.rglob(f"{name}.onnx"))
        if not found:
            raise FileNotFoundError(f"Tải xong nhưng không thấy {name}.onnx trong {self.data_dir}")
        return found[0]

    def _model(self, name: str):
        if name not in self._models:
            from piper import PiperVoice

            path = self._voice_path(name)
            log.info("Nạp Piper voice %s", path.name)
            self._models[name] = PiperVoice.load(str(path))
        return self._models[name]

    def load(self) -> None:
        if self.check_tones:
            self._voice_names = self._drop_toneless_voices(self._voice_names)
        # Chỉ nạp giọng đầu tiên; các giọng còn lại nạp khi thực sự được dùng
        # (mỗi giọng là một file ONNX 20–60 MB, không nên tải sẵn hết).
        self._model(self._voice_names[0])
        self._loaded = True

    # -------------------------------------------------- kiểm tra thanh điệu
    def _drop_toneless_voices(self, names: list[str]) -> list[str]:
        """Loại các giọng không mã hoá được thanh điệu của ngôn ngữ có thanh.

        eSpeak biểu diễn thanh điệu bằng chữ số (`1`–`8`) trong chuỗi phoneme. Một
        số giọng Piper được huấn luyện với bảng phoneme thiếu hẳn các ký hiệu đó,
        nên thanh điệu bị **bỏ lặng lẽ** — chỉ hiện ra dưới dạng cảnh báo
        `Missing phoneme from id map: 2` của thư viện piper. Kết quả là tiếng Việt
        mất dấu, nghe như một thứ tiếng khác.

        Với dự án này thì còn tệ hơn "audio xấu": mô hình sẽ học lối tắt "mất thanh
        điệu ⇒ fake", một manh mối không hề tồn tại ở các engine TTS tử tế.
        """
        keep, dropped = [], []
        for name in names:
            problem = self._tone_problem(name)
            (dropped if problem else keep).append((name, problem))
        for name, problem in dropped:
            log.warning("Bỏ giọng Piper %s — %s", name, problem)
        if dropped and keep:
            log.info("Còn dùng %d/%d giọng: %s",
                     len(keep), len(names), ", ".join(n for n, _ in keep))
        if not keep:
            raise RuntimeError(
                "Không giọng Piper nào mã hoá được thanh điệu: "
                + "; ".join(f"{n} ({p})" for n, p in dropped)
                + ". Chọn giọng khác (vd vi_VN-vais1000-medium), hoặc đặt "
                  "generate.options.piper.check_tones=false nếu cố tình muốn dùng."
            )
        return [name for name, _ in keep]

    def _tone_problem(self, name: str) -> str | None:
        """Mô tả vấn đề thanh điệu của giọng, hoặc None nếu ổn."""
        import json

        config = self._voice_path(name).with_suffix(".onnx.json")
        if not config.exists():
            return None                      # không đọc được thì để engine tự thử
        try:
            data = json.loads(config.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        family = (data.get("language") or {}).get("family") or data.get("espeak", {}).get("voice", "")
        if family not in TONAL_LANGUAGES:
            return None
        id_map = data.get("phoneme_id_map") or {}
        missing = [t for t in TONE_SYMBOLS if t not in id_map]
        if len(missing) == len(TONE_SYMBOLS):
            return (f"bảng phoneme không có ký hiệu thanh điệu nào ⇒ audio sẽ mất dấu "
                    f"(ngôn ngữ {family!r}, {len(id_map)} phoneme)")
        if missing:
            return f"thiếu ký hiệu thanh điệu {', '.join(missing)} ⇒ một số dấu bị bỏ"
        return None

    # -------------------------------------------------------------- sinh audio
    def synthesize(
        self,
        text: str,
        voice: str | None = None,
        ref_audio: str | None = None,
        ref_text: str | None = None,
        language: str | None = None,   # bỏ qua: giọng vi_VN đã cố định ngôn ngữ
    ) -> tuple[np.ndarray, int]:
        name = voice or self._voice_names[0]
        model = self._model(name)

        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav_file:
            model.synthesize_wav(text, wav_file)
        buffer.seek(0)
        with wave.open(buffer, "rb") as wav_file:
            sample_rate = wav_file.getframerate()
            frames = wav_file.readframes(wav_file.getnframes())
            channels = wav_file.getnchannels()

        audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        if channels > 1:
            audio = audio.reshape(-1, channels).mean(axis=1)
        return audio, sample_rate
