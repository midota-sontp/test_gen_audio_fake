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

DEFAULT_VOICES = ("vi_VN-vais1000-medium", "vi_VN-25hours_single-low", "vi_VN-vivos-x_low")
DEFAULT_DATA_DIR = Path("models/piper")


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
        # Chỉ nạp giọng đầu tiên; các giọng còn lại nạp khi thực sự được dùng
        # (mỗi giọng là một file ONNX 20–60 MB, không nên tải sẵn hết).
        self._model(self._voice_names[0])
        self._loaded = True

    # -------------------------------------------------------------- sinh audio
    def synthesize(
        self,
        text: str,
        voice: str | None = None,
        ref_audio: str | None = None,
        ref_text: str | None = None,
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
