"""Fixture dùng chung: dataset giả lập + backbone/generator giả để test nhanh.

Test không tải model thật (WavLM ~400MB) — thay bằng backbone giả có cùng giao
diện, nên toàn bộ pipeline vẫn được chạy thật từ đầu đến cuối.
"""

from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf

from aidetector.features.backbones import Backbone
from aidetector.generate.base import KIND_CLONE, KIND_TTS, Generator

SR = 16_000


def speech_like(seconds: float, seed: int, sr: int = SR, formant_shift: float = 1.0) -> np.ndarray:
    """Tín hiệu giống tiếng nói: hoà âm có rung + đường bao biên độ + chút nhiễu."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * sr)) / sr
    f0 = (110 + rng.uniform(-25, 60)) * formant_shift
    vibrato = 1 + 0.02 * np.sin(2 * np.pi * 5.2 * t)
    audio = sum(
        (1.0 / k) * np.sin(2 * np.pi * f0 * k * t * vibrato + rng.uniform(0, 6.28))
        for k in range(1, 7)
    )
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 3.1 * t + rng.uniform(0, 6.28))
    audio = audio * envelope + 0.01 * rng.standard_normal(t.size)
    return (0.3 * audio / (np.max(np.abs(audio)) + 1e-9)).astype(np.float32)


SENTENCES = [
    "hôm nay trời rất đẹp và mát mẻ nên cả nhà cùng nhau đi dạo phố",
    "công nghệ trí tuệ nhân tạo đang thay đổi cách chúng ta làm việc mỗi ngày",
    "xin vui lòng chờ trong giây lát nhân viên sẽ hỗ trợ quý khách ngay",
    "học sinh cần rèn luyện thói quen đọc sách để mở rộng vốn từ vựng",
    "ngân hàng khuyến cáo khách hàng không cung cấp mã xác thực cho người lạ",
    "đội tuyển quốc gia sẽ có trận đấu quyết định vào tối thứ bảy này",
]


@pytest.fixture
def vivos_like(tmp_path):
    """Tạo dataset kiểu VIVOS: train/ + test/, waves/<SPK>/*.wav, prompts.txt."""
    root = tmp_path / "raw_vivos"
    for split, speakers in (("train", 6), ("test", 2)):
        prompts = []
        for s in range(speakers):
            spk = f"VIVOS{'SPK' if split == 'train' else 'DEV'}{s:02d}"
            spk_dir = root / split / "waves" / spk
            spk_dir.mkdir(parents=True, exist_ok=True)
            for u in range(4):
                utt = f"{spk}_R{u:03d}"
                audio = speech_like(4.0 + 0.5 * u, seed=hash((spk, u)) % 10_000,
                                    formant_shift=1.0 + 0.05 * s)
                sf.write(str(spk_dir / f"{utt}.wav"), audio, SR, subtype="PCM_16")
                prompts.append(f"{utt} {SENTENCES[u % len(SENTENCES)]}")
        (root / split / "prompts.txt").write_text("\n".join(prompts), encoding="utf-8")
    return root


class DummyTTS(Generator):
    """Engine giả: sinh audio có phổ khác rõ rệt so với `speech_like` để phân biệt được."""

    id = "dummy_tts"
    kind = KIND_TTS
    native_sample_rate = 24_000

    def voices(self):
        return ["voice_a", "voice_b"]

    def synthesize(self, text, voice=None, ref_audio=None, ref_text=None):
        seed = abs(hash((text, voice))) % 10_000
        rng = np.random.default_rng(seed)
        t = np.arange(int(5.0 * self.native_sample_rate)) / self.native_sample_rate
        # Sóng vuông cực đều + không rung: đặc trưng "máy" rất dễ tách.
        audio = np.sign(np.sin(2 * np.pi * 200 * t)) * 0.25
        audio += 0.002 * rng.standard_normal(t.size)
        return audio.astype(np.float32), self.native_sample_rate


class DummyClone(DummyTTS):
    id = "dummy_clone"
    kind = KIND_CLONE

    def voices(self):
        return []

    def synthesize(self, text, voice=None, ref_audio=None, ref_text=None):
        if not ref_audio:
            raise ValueError("cần ref_audio")
        return super().synthesize(text, voice, ref_audio, ref_text)


class DummyBackbone(Backbone):
    """Đặc trưng cầm tay: thống kê phổ đơn giản, đủ để tách hai lớp trong test."""

    id = "dummy"
    default_checkpoint = "dummy/v1"

    @property
    def hidden_size(self) -> int:
        return 16

    def load(self) -> None:
        self._model = "loaded"

    def embed(self, waveforms):
        import numpy as np

        out = []
        for wav in waveforms:
            spectrum = np.abs(np.fft.rfft(wav[: 16_000 * 3]))
            bands = np.array_split(spectrum, 16)
            vector = np.array([np.log1p(b.mean()) for b in bands], dtype=np.float32)
            if self.pooling == "mean_std":
                vector = np.concatenate([vector, vector * 0.5])
            out.append(vector)
        return np.stack(out)


@pytest.fixture(scope="session", autouse=True)
def _register_dummies():
    """Đăng ký các thành phần giả vào registry một lần cho cả phiên test."""
    from aidetector.features import backbones as bb
    from aidetector.generate import base as gb

    for cls in (DummyTTS, DummyClone):
        gb._REGISTRY.setdefault(cls.id, cls)
    bb._REGISTRY.setdefault(DummyBackbone.id, DummyBackbone)
    yield
