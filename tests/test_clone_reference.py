"""Reference cho voice cloning: ngắn quá thì clone ra người khác.

Zero-shot cloning lấy danh tính người nói TỪ reference, nên chất lượng reference là
biến quan trọng nhất — không phải checkpoint, không phải knob của engine. OmniVoice
nhận 3–25 giây, nhưng 3 giây là mức *chạy được*, không phải mức *đủ*.

Corpus của dự án ép mọi audio về 3–10 giây và VIVOS trung bình ~3,7 giây, nên nếu mỗi
fake chỉ dùng một utterance làm reference thì reference luôn nằm ở đúng cái mức tối
thiểu đó. Đó là lý do `_pick_reference` trả về NHIỀU bản ghi để ghép lại.
"""

from __future__ import annotations

import numpy as np
import soundfile as sf

from aidetector.corpus.manifest import Manifest
from aidetector.corpus.schema import LABEL_REAL, Record
from aidetector.corpus.spec import AudioSpec
from aidetector.generate import (
    MAX_REF_SECONDS,
    REF_VARIANTS,
    TARGET_REF_SECONDS,
    _is_partial_chunk,
    _materialize_reference,
    _pick_reference,
)

SPEC = AudioSpec()


def _add_real(manifest: Manifest, utt_id: str, speaker: str, seconds: float, text: str) -> Record:
    """Ghi một bản ghi REAL vào manifest kèm file audio thật (im lặng là đủ cho test)."""
    rec = Record(utt_id=utt_id, path="", label=LABEL_REAL, source="vivos",
                 speaker=speaker, text=text, duration=seconds)
    path = manifest.root / "audio" / f"{utt_id}.wav"
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.zeros(int(seconds * SPEC.sample_rate), dtype=np.float32),
             SPEC.sample_rate, subtype="PCM_16")
    rec.path = str(path.relative_to(manifest.root))
    rec.duration = seconds
    manifest.add(rec)
    return rec


def _corpus(tmp_path, n: int = 6, seconds: float = 3.5) -> tuple[Manifest, Record]:
    manifest = Manifest(tmp_path / "corpus")
    for i in range(n):
        _add_real(manifest, f"vivos-spk1-{i:02d}", "spk1", seconds, f"câu số {i}")
    _add_real(manifest, "vivos-spk2-00", "spk2", seconds, "người khác nói")
    target = manifest.get("vivos-spk1-00")
    assert target is not None
    return manifest, target


def test_reference_is_built_up_to_the_target_length(tmp_path):
    """Nhiều utterance ngắn được ghép lại chứ không lấy đúng một cái rồi thôi."""
    manifest, target = _corpus(tmp_path, n=6, seconds=3.5)
    refs = _pick_reference(manifest, target)
    total = sum(r.duration for r in refs)
    assert len(refs) > 1, "một utterance 3,5 giây là quá ngắn để clone giọng"
    assert total >= TARGET_REF_SECONDS * 0.8, f"chỉ ghép được {total:.1f} giây"


def test_reference_never_exceeds_the_engine_limit(tmp_path):
    """Quá 20–25 giây thì OmniVoice tự cảnh báo là chất lượng clone GIẢM."""
    manifest, target = _corpus(tmp_path, n=12, seconds=9.0)
    refs = _pick_reference(manifest, target)
    assert sum(r.duration for r in refs) <= MAX_REF_SECONDS


def test_reference_only_uses_the_same_speaker(tmp_path):
    manifest, target = _corpus(tmp_path)
    refs = _pick_reference(manifest, target)
    assert refs and {r.speaker for r in refs} == {"spk1"}


def test_reference_excludes_the_target_itself(tmp_path):
    """Reference trùng câu đích thì fake chỉ là bản đọc lại của chính real đó."""
    manifest, target = _corpus(tmp_path)
    assert target.utt_id not in {r.utt_id for r in _pick_reference(manifest, target)}


def test_records_without_transcript_are_not_used(tmp_path):
    manifest, target = _corpus(tmp_path, n=3)
    _add_real(manifest, "vivos-spk1-mute", "spk1", 8.0, "")
    assert "vivos-spk1-mute" not in {r.utt_id for r in _pick_reference(manifest, target)}


def test_pieces_of_a_split_file_are_not_used_as_reference(tmp_path):
    """Mọi mảnh của file bị cắt đều mang NGUYÊN transcript của file gốc.

    Đưa cặp (audio 3 giây, transcript của 12 giây) cho engine là phá gióng hàng
    audio–text, và bộ ước lượng độ dài của OmniVoice tính tốc độ nói bằng
    `số ký tự ref_text / thời lượng ref_audio` nên câu đích sẽ bị đọc nhanh vống lên.
    """
    manifest, target = _corpus(tmp_path, n=3)
    long_text = "một câu rất dài bị cắt thành nhiều đoạn nhưng transcript vẫn nguyên"
    _add_real(manifest, "vivos-split", "spk1", 9.0, long_text)     # đoạn 0
    _add_real(manifest, "vivos-split-1", "spk1", 9.0, long_text)   # đoạn 1

    assert _is_partial_chunk(manifest, manifest.get("vivos-split"))
    assert _is_partial_chunk(manifest, manifest.get("vivos-split-1"))
    assert not _is_partial_chunk(manifest, target)

    picked = {r.utt_id for r in _pick_reference(manifest, target)}
    assert not picked & {"vivos-split", "vivos-split-1"}


def test_reference_selection_is_deterministic(tmp_path):
    """Chạy lại `generate` phải ra đúng dataset cũ — kể cả reference đã dùng."""
    manifest, target = _corpus(tmp_path)
    first = [r.utt_id for r in _pick_reference(manifest, target)]
    second = [r.utt_id for r in _pick_reference(manifest, target)]
    assert first == second


def test_speaker_without_any_other_utterance_has_no_reference(tmp_path):
    manifest = Manifest(tmp_path / "corpus")
    lonely = _add_real(manifest, "vivos-solo", "solo", 4.0, "chỉ có một câu")
    assert _pick_reference(manifest, lonely) == []


def test_materialised_reference_matches_audio_with_transcript(tmp_path):
    """Audio ghép và transcript ghép phải cùng thứ tự — lệch là clone tệ hơn cả ngắn."""
    manifest, target = _corpus(tmp_path, n=4, seconds=4.0)
    work = tmp_path / "work"
    work.mkdir()
    refs = _pick_reference(manifest, target)
    path, text = _materialize_reference(manifest, refs, SPEC, work, {})

    audio, sample_rate = sf.read(path)
    assert sample_rate == SPEC.sample_rate
    expected = sum(r.duration for r in refs)
    # Cộng thêm khoảng lặng chèn giữa các đoạn, nên độ dài phải ≥ tổng các đoạn.
    assert expected <= len(audio) / sample_rate <= expected + len(refs) * 0.5
    for rec in refs:
        assert rec.text in text
    assert text.index(refs[0].text) < text.index(refs[-1].text)


def test_single_record_reference_uses_the_corpus_file_directly(tmp_path):
    """Không cần ghép thì không tạo file tạm."""
    manifest = Manifest(tmp_path / "corpus")
    _add_real(manifest, "vivos-a", "spk1", 4.0, "câu a")
    long_one = _add_real(manifest, "vivos-b", "spk1", 20.0, "câu b rất dài")
    target = manifest.get("vivos-a")

    refs = _pick_reference(manifest, target)
    assert refs == [long_one]
    path, text = _materialize_reference(manifest, refs, SPEC, tmp_path / "work", {})
    assert path == str(manifest.abs_path(long_one)) and text == long_one.text


def test_generate_fakes_gives_the_engine_a_long_reference(tmp_path, vivos_like, monkeypatch):
    """Kiểm tra đường đi thật: engine nhận được reference đã ghép, không phải 4 giây."""
    from aidetector.generate import base as gen_base, generate_fakes
    from aidetector.ingest import ingest_source
    from aidetector.ingest.vivos import VivosAdapter

    seen: list[tuple[float, str]] = []

    class RefSpy(gen_base.Generator):
        id = "ref_spy"
        kind = gen_base.KIND_CLONE
        native_sample_rate = 24_000

        def voices(self):
            return []

        def synthesize(self, text, voice=None, ref_audio=None, ref_text=None, language=None):
            audio, sample_rate = sf.read(ref_audio)
            seen.append((len(audio) / sample_rate, ref_text))
            return np.zeros(24_000 * 4, dtype=np.float32), self.native_sample_rate

    monkeypatch.setitem(gen_base._REGISTRY, RefSpy.id, RefSpy)

    manifest = Manifest(tmp_path / "corpus")
    ingest_source(manifest, VivosAdapter(), vivos_like, "vivos", SPEC, language="vi")
    generate_fakes(manifest, "ref_spy", SPEC, count=4)

    assert seen, "không sinh được mẫu nào"
    for seconds, ref_text in seen:
        assert seconds >= TARGET_REF_SECONDS * 0.7, f"reference chỉ {seconds:.1f} giây"
        assert seconds <= MAX_REF_SECONDS
        assert ref_text.strip(), "reference phải có transcript đi kèm"


def test_reference_workdir_is_cleaned_up(tmp_path, vivos_like, monkeypatch):
    """Thư mục reference tạm không được để lại rác sau khi sinh xong."""
    from aidetector.generate import base as gen_base, generate_fakes
    from aidetector.ingest import ingest_source
    from aidetector.ingest.vivos import VivosAdapter

    dirs: list[str] = []

    class DirSpy(gen_base.Generator):
        id = "dir_spy"
        kind = gen_base.KIND_CLONE
        native_sample_rate = 24_000

        def voices(self):
            return []

        def synthesize(self, text, voice=None, ref_audio=None, ref_text=None, language=None):
            from pathlib import Path

            parent = Path(ref_audio).parent
            if "aidetector-ref-" in str(parent):
                dirs.append(str(parent))
            return np.zeros(24_000 * 4, dtype=np.float32), self.native_sample_rate

    monkeypatch.setitem(gen_base._REGISTRY, DirSpy.id, DirSpy)

    manifest = Manifest(tmp_path / "corpus")
    ingest_source(manifest, VivosAdapter(), vivos_like, "vivos", SPEC, language="vi")
    generate_fakes(manifest, "dir_spy", SPEC, count=3)

    from pathlib import Path

    assert dirs, "không có reference nào được ghép"
    assert not Path(dirs[0]).exists()


def test_references_are_reused_across_targets_of_one_speaker(tmp_path):
    """Mỗi mẫu một file ghép riêng là trả giá đĩa mà không đổi lấy gì.

    Chạy thật sinh ~800 mẫu cloning; nếu mỗi mẫu ghi một reference 12 giây riêng thì
    đó là ~300 MB file tạm. Tổ hợp bốc theo (speaker, variant) nên số file bị chặn
    trên bởi số speaker × REF_VARIANTS.
    """
    manifest = Manifest(tmp_path / "corpus")
    for i in range(20):
        _add_real(manifest, f"vivos-spk1-{i:02d}", "spk1", 4.0, f"câu số {i}")

    combos = {
        tuple(r.utt_id for r in _pick_reference(manifest, rec))
        for rec in manifest.reals
    }
    assert 1 < len(combos) <= REF_VARIANTS * 2, f"{len(combos)} tổ hợp cho 20 mẫu"
