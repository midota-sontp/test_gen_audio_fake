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


# --------------------------------------------------------------------- reference
# Ba lỗi dưới đây đều cho ra cùng một triệu chứng: engine chạy trơn, log sạch, nhưng
# giọng clone nghe KHÔNG giống người nói gốc.


def test_all_caps_reference_transcript_is_lowered(fake_omnivoice):
    """`normalize_text` của OmniVoice KHÔNG áp lên ref_text — ta phải tự làm.

    Mã nguồn thư viện ghi rõ: normalization chỉ chạy trên text đích, *"not ref_text,
    which must stay aligned with the reference audio"*. Transcript VIVOS toàn chữ HOA
    nên nếu đưa nguyên vào, cặp (audio, text) của reference gióng hàng kém và danh
    tính giọng lấy ra được cũng nhoè theo.
    """
    generator = OmniVoiceGenerator(device="cpu")
    generator.synthesize("câu đích", ref_audio="ref.wav",
                         ref_text="CŨNG LÊN TIẾNG ỦNG HỘ CÁC KIẾN NGHỊ NÀY")
    assert fake_omnivoice[0]["ref_text"] == "cũng lên tiếng ủng hộ các kiến nghị này"


def test_mixed_case_reference_transcript_is_left_alone(fake_omnivoice):
    generator = OmniVoiceGenerator(device="cpu")
    generator.synthesize("câu đích", ref_audio="ref.wav", ref_text="Hà Nội hôm nay mưa")
    assert fake_omnivoice[0]["ref_text"] == "Hà Nội hôm nay mưa"


def test_empty_reference_transcript_becomes_none(fake_omnivoice):
    """`""` nghĩa là "reference không nói gì"; `None` mới là "nhờ thư viện tự nhận dạng"."""
    generator = OmniVoiceGenerator(device="cpu")
    generator.synthesize("câu đích", ref_audio="ref.wav", ref_text="")
    assert fake_omnivoice[0]["ref_text"] is None


def test_sample_rate_comes_from_the_model_not_a_constant(monkeypatch):
    """Tokenizer chạy ở tần số khác 24 kHz thì resample sẽ dịch cả cao độ lẫn tốc độ."""
    module = types.ModuleType("omnivoice")

    class _OmniVoice:
        sampling_rate = 22_050

        @classmethod
        def from_pretrained(cls, checkpoint, **kwargs):
            return cls()

        def generate(self, **kwargs):
            return [np.zeros(22_050 * 4, dtype=np.float32)]

    module.OmniVoice = _OmniVoice
    monkeypatch.setitem(sys.modules, "omnivoice", module)

    generator = OmniVoiceGenerator(device="cpu")
    _, sample_rate = generator.synthesize("câu đích", ref_audio="ref.wav", ref_text="ref")
    assert sample_rate == 22_050


def test_decoder_knobs_reach_the_model(fake_omnivoice):
    generator = OmniVoiceGenerator(device="cpu", guidance_scale=3.0, num_step=48)
    generator.synthesize("câu đích", ref_audio="ref.wav", ref_text="ref")
    call = fake_omnivoice[0]
    assert call["guidance_scale"] == 3.0 and call["num_step"] == 48


def test_knobs_are_absent_when_not_configured(fake_omnivoice):
    """Không đặt thì không truyền — để mặc định của thư viện tự quyết."""
    generator = OmniVoiceGenerator(device="cpu")
    generator.synthesize("câu đích", ref_audio="ref.wav", ref_text="ref")
    assert "guidance_scale" not in fake_omnivoice[0]


# ------------------------------------------------------------------- biến thể
# Engine cloning không có `voices()`, nên mọi bản sinh vốn mang đúng một tag và một
# utt_id. Muốn so hai checkpoint thì phải phân biệt được chúng trong manifest.


def test_default_checkpoint_keeps_the_plain_tag():
    """Corpus đã sinh trước đó không được coi là thiếu rồi sinh lại từ đầu."""
    generator = OmniVoiceGenerator(device="cpu")
    assert generator.variant == "" and generator.tag(None) == "omnivoice"


def test_other_checkpoints_are_named_in_the_tag():
    generator = OmniVoiceGenerator(device="cpu", checkpoint="k2-fsa/OmniVoice")
    assert generator.tag(None) == "omnivoice:k2-fsa-omnivoice"


def test_two_checkpoints_live_side_by_side_in_one_corpus(tmp_path, vivos_like, monkeypatch):
    """Không có việc này thì lượt A/B thứ hai bị bỏ qua hết vì trùng utt_id."""
    from aidetector.generate import base as gen_base

    class CheckpointSpy(gen_base.Generator):
        id = "ckpt_spy"
        kind = gen_base.KIND_CLONE
        native_sample_rate = 24_000

        @property
        def variant(self):
            return self.options.get("checkpoint", "")

        def voices(self):
            return []

        def synthesize(self, text, voice=None, ref_audio=None, ref_text=None, language=None):
            return np.zeros(24_000 * 4, dtype=np.float32), self.native_sample_rate

    monkeypatch.setitem(gen_base._REGISTRY, CheckpointSpy.id, CheckpointSpy)

    manifest = Manifest(tmp_path / "corpus")
    ingest_source(manifest, VivosAdapter(), vivos_like, "vivos", SPEC, language="vi")
    first = generate_fakes(manifest, "ckpt_spy", SPEC, count=4)
    second = generate_fakes(manifest, "ckpt_spy", SPEC, count=4,
                            options={"checkpoint": "bản-b"})

    assert first["kept"] and second["kept"], "lượt thứ hai bị bỏ qua"
    assert second["skip_exists"] == 0
    tags = {f.generator for f in manifest.fakes}
    assert tags == {"ckpt_spy", "ckpt_spy:bản-b"}
    # Cùng engine ⇒ chia tập theo engine và lọc holdout vẫn hoạt động như cũ.
    assert {f.engine for f in manifest.fakes} == {"ckpt_spy"}


def test_rerunning_the_same_checkpoint_is_still_idempotent(tmp_path, vivos_like, monkeypatch):
    from aidetector.generate import base as gen_base

    class CheckpointSpy(gen_base.Generator):
        id = "ckpt_spy2"
        kind = gen_base.KIND_CLONE
        native_sample_rate = 24_000

        def voices(self):
            return []

        def synthesize(self, text, voice=None, ref_audio=None, ref_text=None, language=None):
            return np.zeros(24_000 * 4, dtype=np.float32), self.native_sample_rate

    monkeypatch.setitem(gen_base._REGISTRY, CheckpointSpy.id, CheckpointSpy)

    manifest = Manifest(tmp_path / "corpus")
    ingest_source(manifest, VivosAdapter(), vivos_like, "vivos", SPEC, language="vi")
    first = generate_fakes(manifest, "ckpt_spy2", SPEC, count=4)
    second = generate_fakes(manifest, "ckpt_spy2", SPEC, count=4)
    assert second["kept"] == 0 and second["skip_exists"] == first["kept"]


# ---------------------------------------------------------------------- text đích
# Triệu chứng: fake đọc đúng vài từ đầu rồi vỡ thành âm vô nghĩa.
# ASR trên một mẫu thật:
#   đích  : trên cơ sở đó những hành vi vi phạm sẽ được xử lý nghiêm
#   fake  : trên cơ sở đó nay nghe pi mc đưa ước ít xù lý nghe


def test_all_caps_target_text_is_lowered(fake_omnivoice):
    """`normalize_text=True` KHÔNG hạ chữ hoa — đọc mã thư viện thì rõ.

    Với ngôn ngữ không phải zh/en, `normalize_text` chỉ chạy `num2words` trên số
    nguyên rồi trả text về nguyên vẹn. Cả dự án tưởng cờ đó xử lý được transcript
    VIVOS toàn CHỮ HOA; nó chưa bao giờ xử lý.
    """
    generator = OmniVoiceGenerator(device="cpu")
    generator.synthesize("TRÊN CƠ SỞ ĐÓ NHỮNG HÀNH VI VI PHẠM SẼ ĐƯỢC XỬ LÝ NGHIÊM",
                         ref_audio="ref.wav", ref_text="ref")
    assert fake_omnivoice[0]["text"].startswith("trên cơ sở đó những hành vi")


def test_target_text_gets_a_sentence_ending(fake_omnivoice):
    """Thư viện thêm dấu kết cho ref_text nhưng không cho text đích.

    Transcript VIVOS không có dấu chấm nào, nên model phải tự đoán chỗ kết thúc.
    """
    generator = OmniVoiceGenerator(device="cpu")
    generator.synthesize("một câu không có dấu chấm", ref_audio="ref.wav", ref_text="ref")
    assert fake_omnivoice[0]["text"] == "một câu không có dấu chấm."


def test_existing_punctuation_is_not_doubled(fake_omnivoice):
    generator = OmniVoiceGenerator(device="cpu")
    for sentence in ("đã có dấu chấm.", "thế à?", "hay quá!"):
        generator.synthesize(sentence, ref_audio="ref.wav", ref_text="ref")
    assert [c["text"] for c in fake_omnivoice] == ["đã có dấu chấm.", "thế à?", "hay quá!"]


def test_mixed_case_target_text_is_left_alone(fake_omnivoice):
    """Chỉ chữa chữ hoa toàn bộ; câu viết thường bình thường không bị đụng."""
    generator = OmniVoiceGenerator(device="cpu")
    generator.synthesize("Hà Nội hôm nay mưa", ref_audio="ref.wav", ref_text="ref")
    assert fake_omnivoice[0]["text"] == "Hà Nội hôm nay mưa."


def test_target_and_reference_end_up_in_the_same_casing(fake_omnivoice):
    """Reference chữ thường mà text đích chữ HOA là bắt model gióng hai dạng khác nhau."""
    generator = OmniVoiceGenerator(device="cpu")
    generator.synthesize("CÂU ĐÍCH TOÀN HOA", ref_audio="ref.wav", ref_text="CÂU THAM CHIẾU TOÀN HOA")
    call = fake_omnivoice[0]
    assert call["text"] == "câu đích toàn hoa." and call["ref_text"] == "câu tham chiếu toàn hoa"
