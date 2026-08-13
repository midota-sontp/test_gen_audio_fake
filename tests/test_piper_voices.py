"""Giọng Piper phải mã hoá được thanh điệu, nếu không audio tiếng Việt sẽ mất dấu.

Bối cảnh: hai trong ba giọng `vi_VN` trên `rhasspy/piper-voices`
(`vi_VN-25hours_single-low`, `vi_VN-vivos-x_low`) có `phoneme_id_map` thiếu hẳn các
ký hiệu thanh điệu `1`–`8` mà eSpeak sinh ra. Với một câu tiếng Việt 87 phoneme,
9/9 ký hiệu thanh điệu bị bỏ — audio đọc ra mất dấu hoàn toàn. Thư viện piper chỉ
báo bằng dòng `Missing phoneme from id map: 2`, rất dễ trôi qua giữa hàng nghìn log.

Với dự án này thì tệ hơn "audio xấu": mô hình sẽ học lối tắt "mất thanh điệu ⇒ fake",
một manh mối không tồn tại ở các engine TTS tử tế.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aidetector.config import Config
from aidetector.generate.piper import DEFAULT_VOICES, TONE_SYMBOLS, PiperGenerator


def _fake_voice(data_dir: Path, name: str, *, family: str, tones: bool) -> None:
    """Dựng một cặp file giọng giả (.onnx rỗng + .onnx.json) để khỏi tải mạng."""
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / f"{name}.onnx").write_bytes(b"")
    id_map = {"a": 1, "b": 2, "k": 3}
    if tones:
        id_map.update({t: 10 + i for i, t in enumerate(TONE_SYMBOLS)})
    (data_dir / f"{name}.onnx.json").write_text(
        json.dumps({"language": {"family": family}, "phoneme_id_map": id_map}),
        encoding="utf-8",
    )


def _generator(data_dir: Path, voices: list[str], **options) -> PiperGenerator:
    return PiperGenerator(device="cpu", data_dir=str(data_dir), voices=voices, **options)


# --------------------------------------------------------------- phát hiện lỗi
def test_flags_a_tonal_voice_with_no_tone_symbols(tmp_path):
    _fake_voice(tmp_path, "vi_VN-xau-x_low", family="vi", tones=False)
    problem = _generator(tmp_path, ["vi_VN-xau-x_low"])._tone_problem("vi_VN-xau-x_low")
    assert problem is not None
    assert "thanh điệu" in problem and "mất dấu" in problem


def test_accepts_a_tonal_voice_that_encodes_tones(tmp_path):
    _fake_voice(tmp_path, "vi_VN-tot-medium", family="vi", tones=True)
    assert _generator(tmp_path, ["vi_VN-tot-medium"])._tone_problem("vi_VN-tot-medium") is None


def test_non_tonal_language_is_left_alone(tmp_path):
    """Tiếng Anh không có thanh điệu — thiếu chữ số là bình thường."""
    _fake_voice(tmp_path, "en_US-abc-medium", family="en", tones=False)
    assert _generator(tmp_path, ["en_US-abc-medium"])._tone_problem("en_US-abc-medium") is None


@pytest.mark.parametrize("family", ["vi", "zh", "th"])
def test_every_tonal_language_is_checked(tmp_path, family):
    name = f"{family}_XX-abc-low"
    _fake_voice(tmp_path / family, name, family=family, tones=False)
    assert _generator(tmp_path / family, [name])._tone_problem(name) is not None


# ------------------------------------------------------------- hành vi khi lọc
def test_bad_voices_are_dropped_and_good_ones_kept(tmp_path, caplog):
    _fake_voice(tmp_path, "vi_VN-tot-medium", family="vi", tones=True)
    _fake_voice(tmp_path, "vi_VN-xau-low", family="vi", tones=False)
    generator = _generator(tmp_path, ["vi_VN-xau-low", "vi_VN-tot-medium"])

    with caplog.at_level("WARNING"):
        kept = generator._drop_toneless_voices(list(generator.voices()))

    assert kept == ["vi_VN-tot-medium"]
    assert "vi_VN-xau-low" in caplog.text


def test_all_voices_bad_raises_instead_of_making_junk(tmp_path):
    """Sinh ra cả nghìn file mất dấu rồi mới phát hiện thì đã quá muộn."""
    _fake_voice(tmp_path, "vi_VN-xau-low", family="vi", tones=False)
    _fake_voice(tmp_path, "vi_VN-te-x_low", family="vi", tones=False)
    generator = _generator(tmp_path, ["vi_VN-xau-low", "vi_VN-te-x_low"])

    with pytest.raises(RuntimeError) as err:
        generator._drop_toneless_voices(list(generator.voices()))
    message = str(err.value)
    assert "vi_VN-vais1000-medium" in message      # gợi ý giọng dùng được
    assert "check_tones" in message                # và cách tắt kiểm tra nếu cố ý


def test_check_can_be_disabled_explicitly(tmp_path):
    _fake_voice(tmp_path, "vi_VN-xau-low", family="vi", tones=False)
    generator = _generator(tmp_path, ["vi_VN-xau-low"], check_tones=False)
    assert generator.check_tones is False


def test_missing_config_does_not_block_the_voice(tmp_path):
    """Không đọc được config thì để engine tự thử, đừng chặn oan."""
    (tmp_path / "vi_VN-la-medium.onnx").write_bytes(b"")
    assert _generator(tmp_path, ["vi_VN-la-medium"])._tone_problem("vi_VN-la-medium") is None


# ------------------------------------------------------------------ mặc định
def test_defaults_only_list_tone_capable_voices():
    assert DEFAULT_VOICES == ("vi_VN-vais1000-medium",)


def test_config_does_not_ship_the_toneless_voices():
    voices = Config.load("configs/default.yaml")["generate.voices.piper"]
    assert voices == ["vi_VN-vais1000-medium"]
    for bad in ("vi_VN-25hours_single-low", "vi_VN-vivos-x_low"):
        assert bad not in voices


# ----------------------------------------- kiểm tra trên giọng đã tải thật
@pytest.mark.parametrize(
    ("name", "expect_problem"),
    [
        ("vi_VN-vais1000-medium", False),
        ("vi_VN-25hours_single-low", True),
        ("vi_VN-vivos-x_low", True),
    ],
)
def test_real_downloaded_voices_match_expectations(name, expect_problem):
    """Chạy trên file giọng thật nếu máy đã tải; không có thì bỏ qua."""
    data_dir = Path("models/piper")
    if not (data_dir / f"{name}.onnx.json").exists():
        pytest.skip(f"chưa tải {name}")
    problem = _generator(data_dir, [name])._tone_problem(name)
    assert (problem is not None) is expect_problem, problem
