"""OmniVoice phải được nói rõ ngôn ngữ, nếu không nó đọc tiếng Việt bằng âm vị khác.

Bối cảnh: `OmniVoice.generate()` nhận `language` (mã như "vi") và tài liệu ghi rõ
"Performance is slightly better if you specify the language"; bỏ trống là chế độ
language-agnostic. Với model phủ 600+ ngôn ngữ, để nó tự đoán trên một câu tiếng
Việt là cách nhanh nhất để ra audio "không giống tiếng Việt".

`normalize_text` mặc định là False, trong khi transcript VIVOS toàn CHỮ HOA — dễ bị
đọc thành chuỗi chữ cái rời rạc.
"""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from aidetector.corpus.manifest import Manifest
from aidetector.corpus.spec import AudioSpec
from aidetector.generate import generate_fakes
from aidetector.generate.omnivoice import OmniVoiceGenerator
from aidetector.ingest import ingest_source
from aidetector.ingest.vivos import VivosAdapter

SPEC = AudioSpec()


@pytest.fixture
def fake_omnivoice(monkeypatch):
    """Thay gói omnivoice bằng bản giả, ghi lại đúng tham số đã được truyền."""
    calls: list[dict] = []
    module = types.ModuleType("omnivoice")

    class _OmniVoice:
        @classmethod
        def from_pretrained(cls, checkpoint, **kwargs):
            return cls()

        def generate(self, **kwargs):
            calls.append(kwargs)
            return [np.zeros(24_000 * 4, dtype=np.float32)]

    module.OmniVoice = _OmniVoice
    monkeypatch.setitem(sys.modules, "omnivoice", module)
    return calls


def test_language_is_passed_to_the_model(fake_omnivoice):
    generator = OmniVoiceGenerator(device="cpu")
    generator.synthesize("CŨNG LÊN TIẾNG ỦNG HỘ CÁC KIẾN NGHỊ NÀY",
                         ref_audio="ref.wav", ref_text="câu tham chiếu", language="vi")
    assert fake_omnivoice[0]["language"] == "vi", "bỏ trống ⇒ model phải tự đoán ngôn ngữ"


def test_language_defaults_to_vietnamese(fake_omnivoice):
    """Không ai truyền gì thì vẫn phải là tiếng Việt, không phải None."""
    generator = OmniVoiceGenerator(device="cpu")
    generator.synthesize("xin chào", ref_audio="ref.wav", ref_text="ref")
    assert fake_omnivoice[0]["language"] == "vi"


def test_language_option_can_override_the_default(fake_omnivoice):
    generator = OmniVoiceGenerator(device="cpu", language="en")
    generator.synthesize("hello there", ref_audio="ref.wav", ref_text="ref")
    assert fake_omnivoice[0]["language"] == "en"


def test_text_normalisation_is_on_by_default(fake_omnivoice):
    """Transcript VIVOS toàn chữ HOA; để False dễ bị đọc rời từng chữ cái."""
    generator = OmniVoiceGenerator(device="cpu")
    generator.synthesize("TOÀN CHỮ HOA", ref_audio="ref.wav", ref_text="ref")
    assert fake_omnivoice[0]["normalize_text"] is True

    generator = OmniVoiceGenerator(device="cpu", normalize_text=False)
    generator.synthesize("TOÀN CHỮ HOA", ref_audio="ref.wav", ref_text="ref")
    assert fake_omnivoice[1]["normalize_text"] is False


def test_reference_audio_and_text_both_reach_the_model(fake_omnivoice):
    generator = OmniVoiceGenerator(device="cpu")
    generator.synthesize("câu đích", ref_audio="/tmp/ref.wav", ref_text="câu tham chiếu")
    call = fake_omnivoice[0]
    assert call["ref_audio"] == "/tmp/ref.wav"
    assert call["ref_text"] == "câu tham chiếu"


def test_missing_reference_is_rejected_before_loading_the_model():
    generator = OmniVoiceGenerator(device="cpu")
    with pytest.raises(ValueError, match="ref_audio"):
        generator.synthesize("câu đích")


def test_generate_fakes_forwards_the_record_language(tmp_path, vivos_like, monkeypatch):
    """Ngôn ngữ đi từ manifest xuống engine, không bị rơi ở giữa đường."""
    seen: list[str | None] = []

    from aidetector.generate import base as gen_base

    class LanguageSpy(gen_base.Generator):
        id = "spy_clone"
        kind = gen_base.KIND_CLONE
        native_sample_rate = 24_000

        def voices(self):
            return []

        def synthesize(self, text, voice=None, ref_audio=None, ref_text=None, language=None):
            seen.append(language)
            return np.zeros(24_000 * 4, dtype=np.float32), self.native_sample_rate

    monkeypatch.setitem(gen_base._REGISTRY, LanguageSpy.id, LanguageSpy)

    manifest = Manifest(tmp_path / "corpus")
    ingest_source(manifest, VivosAdapter(), vivos_like, "vivos", SPEC, language="vi")
    generate_fakes(manifest, "spy_clone", SPEC, count=3)

    assert seen and set(seen) == {"vi"}
