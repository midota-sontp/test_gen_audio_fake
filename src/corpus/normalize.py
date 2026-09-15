"""Đưa audio về chuẩn corpus: 16 kHz, mono, WAV PCM 16-bit."""
from __future__ import annotations

import io

import numpy as np
import soundfile as sf
import soxr

TARGET_SR = 16000
TARGET_CHANNELS = 1
TARGET_SUBTYPE = "PCM_16"


def to_mono(x: np.ndarray) -> np.ndarray:
    if x.ndim == 1:
        return x.astype(np.float32, copy=False)
    if x.shape[1] == 1:
        return x[:, 0].astype(np.float32, copy=False)
    return x.mean(axis=1, dtype=np.float32)


def resample(x: np.ndarray, sr: int, target_sr: int = TARGET_SR) -> np.ndarray:
    if sr == target_sr:
        return x.astype(np.float32, copy=False)
    return soxr.resample(x, sr, target_sr, quality="VHQ").astype(np.float32, copy=False)


def to_mono_16k(x: np.ndarray, sr: int, target_sr: int = TARGET_SR) -> np.ndarray:
    return resample(to_mono(x), sr, target_sr)


def apply_rms(x: np.ndarray, target_dbfs: float) -> tuple[np.ndarray, float]:
    """Chuẩn hoá RMS (tuỳ chọn, mặc định TẮT vì spec Demo v1 không yêu cầu)."""
    rms = float(np.sqrt(np.mean(np.square(x, dtype=np.float64)))) if x.size else 0.0
    if rms <= 1e-9:
        return x, 1.0
    gain = (10.0 ** (target_dbfs / 20.0)) / rms
    peak = float(np.max(np.abs(x))) * gain
    if peak > 0.999:                       # tránh tự tạo clipping khi kéo gain lên
        gain *= 0.999 / peak
    return (x * gain).astype(np.float32), gain


def to_pcm16(x: np.ndarray) -> tuple[np.ndarray, float]:
    """float32 -> int16. Trả thêm peak trước khi cắt để ghi vào metadata."""
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    y = np.clip(x, -1.0, 32767.0 / 32768.0)
    return np.round(y * 32768.0).astype(np.int16), peak


def encode_wav(pcm16: np.ndarray, sr: int = TARGET_SR) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, pcm16, sr, subtype=TARGET_SUBTYPE, format="WAV")
    return buf.getvalue()
