"""Chuẩn audio phải được thực thi đúng như bảng đặc tả."""

from __future__ import annotations

import numpy as np
import soundfile as sf

from aidetector.corpus.spec import (
    AudioSpec,
    check_quality,
    load_audio,
    normalize,
    normalize_file,
    normalize_level,
    save_audio,
)
from tests.conftest import SR, speech_like

SPEC = AudioSpec()


def test_spec_defaults_match_standard():
    assert SPEC.sample_rate == 16_000
    assert SPEC.channels == 1
    assert SPEC.subtype == "PCM_16"
    assert (SPEC.min_seconds, SPEC.max_seconds) == (3.0, 10.0)


def test_normalize_drops_too_short():
    assert normalize(speech_like(1.0, seed=1), SPEC) == []


def test_normalize_keeps_in_range_clip():
    chunks = normalize(speech_like(5.0, seed=2), SPEC)
    assert len(chunks) == 1
    duration = len(chunks[0]) / SR
    assert SPEC.min_seconds <= duration <= SPEC.max_seconds


def test_long_audio_is_split_into_valid_chunks():
    chunks = normalize(speech_like(26.0, seed=3), SPEC)
    assert len(chunks) >= 2
    for chunk in chunks:
        assert SPEC.min_seconds <= len(chunk) / SR <= SPEC.max_seconds


def test_pad_policy_lengthens_short_audio():
    spec = AudioSpec(short_policy="pad")
    chunks = normalize(speech_like(1.5, seed=4), spec)
    assert len(chunks) == 1
    assert len(chunks[0]) >= spec.min_samples


def test_nan_and_inf_are_removed():
    audio = speech_like(4.0, seed=5).copy()
    audio[100] = np.nan
    audio[200] = np.inf
    chunk = normalize(audio, SPEC)[0]
    assert np.all(np.isfinite(chunk))


def test_level_normalisation_hits_target_rms_and_avoids_clipping():
    loud = speech_like(4.0, seed=6) * 40          # cố tình làm vỡ tiếng
    out = normalize_level(loud, SPEC)
    rms_dbfs = 20 * np.log10(np.sqrt(np.mean(out**2)))
    peak_dbfs = 20 * np.log10(np.max(np.abs(out)))
    assert abs(rms_dbfs - SPEC.target_rms_dbfs) < 1.5
    assert peak_dbfs <= SPEC.peak_ceiling_dbfs + 0.1
    assert not check_quality(out, SPEC)


def test_silence_is_trimmed():
    audio = np.concatenate([np.zeros(SR * 2, np.float32), speech_like(5.0, seed=7),
                            np.zeros(SR * 2, np.float32)])
    chunk = normalize(audio, SPEC)[0]
    assert len(chunk) / SR < 7.0                   # đã cắt bớt phần im lặng


def test_quality_check_flags_problems():
    codes = {i.code for i in check_quality(np.ones(SR * 4, np.float32), SPEC)}
    assert "clipping" in codes
    assert {i.code for i in check_quality(np.zeros(SR * 4, np.float32), SPEC)} & {"silent"}
    assert {i.code for i in check_quality(speech_like(1.0, seed=8), SPEC)} == {"too_short"}


def test_roundtrip_through_disk_keeps_spec(tmp_path):
    path = tmp_path / "out.wav"
    chunk = normalize(speech_like(4.0, seed=9), SPEC)[0]
    save_audio(path, chunk, SPEC)
    info = sf.info(str(path))
    assert info.samplerate == 16_000
    assert info.channels == 1
    assert info.subtype == "PCM_16"
    assert np.allclose(load_audio(path, SR), chunk, atol=1e-4)


def test_resamples_and_downmixes_any_input(tmp_path):
    path = tmp_path / "stereo48k.wav"
    stereo = np.stack([speech_like(4.0, seed=10, sr=48_000)] * 2, axis=-1)
    sf.write(str(path), stereo, 48_000)
    chunk = normalize_file(path, SPEC)[0]
    assert chunk.ndim == 1
    assert 3.0 <= len(chunk) / SR <= 10.0
