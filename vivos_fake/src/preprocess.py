"""Audio quality layer: normalize to 16 kHz / mono / PCM-16, validate, skip corrupt.

Used for both real clips (copied into dataset/real) and generated fakes (re-encoded
after Fish Speech so both classes share the same format).
"""
from __future__ import annotations

import logging
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

log = logging.getLogger(__name__)


class CorruptAudioError(Exception):
    """Raised when an audio file cannot be decoded."""


def load_audio(path: str | Path, sample_rate: int, mono: bool = True) -> np.ndarray:
    """Decode + resample + downmix in one shot. Raises CorruptAudioError on failure."""
    try:
        y, _ = librosa.load(str(path), sr=sample_rate, mono=mono)
    except Exception as e:  # unreadable / truncated / not audio
        raise CorruptAudioError(f"cannot decode {path}: {e}") from e
    if y is None or y.size == 0:
        raise CorruptAudioError(f"empty audio: {path}")
    return y.astype(np.float32)


def peak_normalize(y: np.ndarray, level: float = 0.98) -> np.ndarray:
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    return (y / peak * level).astype(np.float32) if peak > 1e-6 else y


def write_pcm16(path: str | Path, y: np.ndarray, sample_rate: int) -> None:
    """Write mono PCM-16 wav, creating parent dirs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), y, sample_rate, subtype="PCM_16")


def normalize_file(src: str | Path, dst: str | Path, sample_rate: int,
                   mono: bool = True, do_peak: bool = True) -> bool:
    """Load `src`, normalize to 16k/mono/PCM-16, write `dst`.
    Returns True on success, False if the source is corrupt (logged, skipped)."""
    try:
        y = load_audio(src, sample_rate, mono)
    except CorruptAudioError as e:
        log.warning("skip corrupt source: %s", e)
        return False
    if do_peak:
        y = peak_normalize(y)
    write_pcm16(dst, y, sample_rate)
    return True


def validate_audio(path: str | Path) -> bool:
    """True iff `path` exists and decodes to at least one frame (used before
    writing a metadata row for a generated fake)."""
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return False
    try:
        info = sf.info(str(p))
        return info.frames > 0
    except Exception:
        return False
