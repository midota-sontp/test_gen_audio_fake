"""Các phép augment phải giữ đúng chuẩn audio và tất định theo seed."""

from __future__ import annotations

import numpy as np
import pytest

from aidetector.augment import AugmentChain
from aidetector.augment.ops import OPS, has_ffmpeg
from aidetector.corpus.spec import AudioSpec, check_quality, normalize_level
from aidetector.utils import stable_rand
from tests.conftest import SR, speech_like

SPEC = AudioSpec()


@pytest.mark.parametrize("name", sorted(OPS))
def test_every_op_returns_valid_audio(name):
    if name == "codec" and not has_ffmpeg():
        pytest.skip("cần ffmpeg")
    audio = speech_like(4.0, seed=1)
    out, tag = OPS[name](audio.copy(), SR, stable_rand("t", name))
    assert out.dtype == np.float32
    assert np.all(np.isfinite(out))
    assert out.size > 0
    assert tag                                   # mỗi phép phải tự khai tên
    assert not check_quality(normalize_level(out, SPEC), SPEC)


def test_chain_is_deterministic_for_the_same_key():
    audio = speech_like(4.0, seed=2)
    chain = AugmentChain({"gaussian_noise": {"p": 1.0}, "gain": {"p": 0.7}}, max_ops=2)
    a, tag_a = chain.apply(audio.copy(), SR, stable_rand(42, "utt-1", 0))
    b, tag_b = chain.apply(audio.copy(), SR, stable_rand(42, "utt-1", 0))
    assert tag_a == tag_b
    assert np.array_equal(a, b)


def test_chain_differs_across_copies():
    audio = speech_like(4.0, seed=3)
    chain = AugmentChain({"gaussian_noise": {"p": 1.0}}, max_ops=1)
    a, _ = chain.apply(audio.copy(), SR, stable_rand(42, "utt-1", 0))
    b, _ = chain.apply(audio.copy(), SR, stable_rand(42, "utt-1", 1))
    assert not np.array_equal(a, b)


def test_chain_respects_max_ops():
    chain = AugmentChain({name: {"p": 1.0} for name in ("gaussian_noise", "gain", "band_limit")},
                         max_ops=1)
    _, tag = chain.apply(speech_like(4.0, seed=4), SR, stable_rand("k"))
    assert "+" not in tag


def test_unknown_op_is_rejected_early():
    with pytest.raises(KeyError, match="không tồn tại"):
        AugmentChain({"khong_ton_tai": {"p": 1.0}})


def test_noise_op_lowers_snr_but_keeps_signal():
    audio = speech_like(4.0, seed=5)
    out, _ = OPS["gaussian_noise"](audio.copy(), SR, stable_rand("n"), snr_range=(10.0, 10.0))
    residual = out - audio
    snr = 10 * np.log10(np.mean(audio**2) / np.mean(residual**2))
    assert 8.0 < snr < 12.0


@pytest.mark.skipif(not has_ffmpeg(), reason="cần ffmpeg")
def test_codec_changes_waveform_but_keeps_length():
    audio = speech_like(4.0, seed=6)
    out, tag = OPS["codec"](audio.copy(), SR, stable_rand("c"), codecs=("mp3",), bitrates=(32,))
    assert out.size == audio.size
    assert tag.startswith("mp3")
    assert not np.allclose(out, audio)          # nén mất dữ liệu ⇒ phải khác
