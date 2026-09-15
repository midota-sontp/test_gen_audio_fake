"""Kiểm định chất lượng audio theo đúng thứ tự trong flowchart Demo v1.

Decode -> Duration -> Speech ratio -> Clipping -> SHA256 (dedupe) -> Normalize
"""
from __future__ import annotations

import hashlib
import io
import subprocess
from dataclasses import dataclass, asdict

import numpy as np
import soundfile as sf
import webrtcvad

CLIP_LEVEL = 0.999          # |x| >= ngưỡng này coi như clipped
VAD_FRAME_MS = 30
VAD_RATE = 16000


class RejectReason:
    DECODE = "decode_failed"
    EMPTY = "empty_audio"
    SR_UNKNOWN = "sample_rate_invalid"
    DURATION = "duration_out_of_range"
    SPEECH = "speech_ratio_too_low"
    CLIPPING = "clipping_too_high"
    DUP_RAW = "duplicate_raw_sha256"
    DUP_NORM = "duplicate_normalized_sha256"
    NORMALIZE = "normalize_failed"


@dataclass
class Probe:
    """Kết quả đo trên audio gốc (trước khi normalize)."""
    duration: float
    sample_rate: int
    channels: int
    speech_ratio: float
    clipping_ratio: float
    rms_db: float

    def as_dict(self) -> dict:
        return asdict(self)


def decode(data: bytes, fallback_ffmpeg: bool = True) -> tuple[np.ndarray, int]:
    """Giải mã bytes -> (float32 [n, ch], sample_rate). Ném ValueError nếu hỏng."""
    try:
        x, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
        if x.size:
            return x, int(sr)
    except Exception:
        if not fallback_ffmpeg:
            raise
    if not fallback_ffmpeg:
        raise ValueError("empty audio")
    return _decode_ffmpeg(data)


def _decode_ffmpeg(data: bytes) -> tuple[np.ndarray, int]:
    """Đường lui cho container mà libsndfile không đọc được (m4a, opus lạ...)."""
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", "pipe:0",
           "-f", "f32le", "-acodec", "pcm_f32le", "pipe:1"]
    proc = subprocess.run(cmd, input=data, capture_output=True)
    if proc.returncode != 0 or not proc.stdout:
        raise ValueError(f"ffmpeg decode failed: {proc.stderr[:200].decode('utf-8', 'replace')}")
    probe = subprocess.run(
        ["ffprobe", "-hide_banner", "-loglevel", "error", "-of", "csv=p=0",
         "-show_entries", "stream=sample_rate,channels", "-select_streams", "a:0", "pipe:0"],
        input=data, capture_output=True)
    sr, ch = probe.stdout.decode().strip().split(",")[:2]
    x = np.frombuffer(proc.stdout, dtype=np.float32).reshape(-1, int(ch))
    return x.copy(), int(sr)


def clipping_ratio(x: np.ndarray) -> float:
    """Tỷ lệ mẫu chạm trần, đo trên audio GỐC (resample có thể tạo overshoot giả)."""
    peak = np.max(np.abs(x), axis=1) if x.ndim == 2 else np.abs(x)
    return float(np.count_nonzero(peak >= CLIP_LEVEL) / max(peak.size, 1))


def rms_dbfs(x: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(np.square(x, dtype=np.float64)))) if x.size else 0.0
    return 20.0 * float(np.log10(max(rms, 1e-12)))


def speech_ratio(mono16k: np.ndarray, aggressiveness: int = 2) -> float:
    """Tỷ lệ frame có speech theo WebRTC VAD, tính trên bản 16 kHz mono."""
    vad = webrtcvad.Vad(aggressiveness)
    frame_len = VAD_RATE * VAD_FRAME_MS // 1000
    pcm = _to_int16(mono16k)
    n_frames = len(pcm) // frame_len
    if n_frames == 0:
        return 0.0
    voiced = 0
    buf = pcm[: n_frames * frame_len].tobytes()
    step = frame_len * 2
    for i in range(n_frames):
        if vad.is_speech(buf[i * step:(i + 1) * step], VAD_RATE):
            voiced += 1
    return voiced / n_frames


def _to_int16(x: np.ndarray) -> np.ndarray:
    y = np.clip(x, -1.0, 32767.0 / 32768.0)
    return np.round(y * 32768.0).astype(np.int16)


def sha256_samples(x: np.ndarray, sr: int) -> str:
    """Hash nội dung audio đã giải mã (không phụ thuộc container/metadata)."""
    h = hashlib.sha256()
    h.update(f"{sr}:{x.shape[1] if x.ndim == 2 else 1}:".encode())
    h.update(np.ascontiguousarray(x, dtype=np.float32).tobytes())
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
