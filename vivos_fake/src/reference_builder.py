"""Step 4: build a 10-20s reference voice per speaker for Fish Speech cloning.

Concatenates the speaker's longest utterances until ~reference_seconds is reached
(one long recording is often enough), normalized to 16k/mono/PCM-16. Also returns
the concatenated transcript, which Fish Speech S2 uses as the reference/prompt text.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from .parser import Utterance
from .preprocess import load_audio, peak_normalize, write_pcm16

log = logging.getLogger(__name__)

_MIN_SECONDS = 10.0
_MAX_SECONDS = 20.0


@dataclass(frozen=True)
class Reference:
    speaker: str
    wav_path: Path
    text: str
    seconds: float


def _duration(u: Utterance) -> float:
    try:
        info = sf.info(str(u.wav_path))
        return info.frames / float(info.samplerate)
    except Exception:
        return 0.0


def build_reference(speaker: str, utts: list[Utterance], reference_seconds: float,
                    sample_rate: int, out_dir: str | Path, overwrite: bool = False) -> Reference:
    """Build (or reuse) reference/<speaker>.wav. Cached unless overwrite=True."""
    out_dir = Path(out_dir)
    ref_wav = out_dir / f"{speaker}.wav"

    target = max(_MIN_SECONDS, min(float(reference_seconds), _MAX_SECONDS))
    ranked = sorted(utts, key=_duration, reverse=True)
    chosen: list[Utterance] = []
    total = 0.0
    for u in ranked:
        d = _duration(u)
        if d <= 0:
            continue
        chosen.append(u)
        total += d
        if total >= target:
            break
    if not chosen:
        raise RuntimeError(f"speaker {speaker}: no readable audio to build a reference")

    text = " ".join(u.text for u in chosen).strip()

    if ref_wav.exists() and not overwrite:
        return Reference(speaker, ref_wav, text, total)

    chunks = []
    for u in chosen:
        try:
            chunks.append(load_audio(u.wav_path, sample_rate, mono=True))
        except Exception as e:
            log.warning("reference %s: skip chunk %s (%s)", speaker, u.audio_id, e)
    if not chunks:
        raise RuntimeError(f"speaker {speaker}: all reference chunks failed to decode")
    y = peak_normalize(np.concatenate(chunks))
    write_pcm16(ref_wav, y, sample_rate)
    log.info("reference %s: %.1fs from %d clip(s) -> %s", speaker, total, len(chunks), ref_wav.name)
    return Reference(speaker, ref_wav, text, total)
