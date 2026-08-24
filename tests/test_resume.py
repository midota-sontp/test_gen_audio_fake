"""Nối lại công việc giữa hai phiên Kaggle.

`/kaggle/working` bị xoá khi phiên kết thúc, mà sinh vài nghìn mẫu cloning mất nhiều
giờ. Nên corpus phải đóng gói được, bung lại được, `generate` phải tiếp tục đúng chỗ đã
dừng, và phải có mốc để đẩy dữ liệu ra ngoài giữa chừng.
"""

from __future__ import annotations

import numpy as np
import pytest

from aidetector.corpus.manifest import Manifest
from aidetector.corpus.spec import AudioSpec
from aidetector.generate import base as gen_base, generate_fakes
from aidetector.ingest import ingest_source
from aidetector.ingest.vivos import VivosAdapter
from aidetector.packaging import pack_corpus, unpack_corpus

SPEC = AudioSpec()


class CountingClone(gen_base.Generator):
    """Engine cloning giả, đếm số lần thực sự phải tổng hợp."""

    id = "resume_spy"
    kind = gen_base.KIND_CLONE
    native_sample_rate = 24_000
    calls = 0

    def voices(self):
        return []

    def synthesize(self, text, voice=None, ref_audio=None, ref_text=None, language=None):
        type(self).calls += 1
        return np.zeros(24_000 * 4, dtype=np.float32), self.native_sample_rate


@pytest.fixture
def corpus(tmp_path, vivos_like, monkeypatch):
    monkeypatch.setitem(gen_base._REGISTRY, CountingClone.id, CountingClone)
    CountingClone.calls = 0
    manifest = Manifest(tmp_path / "corpus")
    ingest_source(manifest, VivosAdapter(), vivos_like, "vivos", SPEC, language="vi")
    return manifest


# ------------------------------------------------------------------- tiếp tục
def test_second_run_only_makes_what_is_missing(corpus):
    """Đây là toàn bộ cơ chế "tiếp tục": cùng --count, chỉ sinh phần còn thiếu."""
    first = generate_fakes(corpus, CountingClone.id, SPEC, count=6)
    assert first["kept"] and CountingClone.calls == 6

    second = generate_fakes(corpus, CountingClone.id, SPEC, count=12)
    assert second["skip_exists"] == 6, "phần đã sinh phải được nhận ra"
    assert CountingClone.calls == 12, "chỉ được tổng hợp 6 mẫu mới"


def test_generation_continues_after_a_pack_unpack_cycle(corpus, tmp_path):
    """Đường đi thật giữa hai phiên: sinh dở → pack → phiên mới unpack → sinh tiếp."""
    generate_fakes(corpus, CountingClone.id, SPEC, count=6)
    corpus.save()
    archive = pack_corpus(corpus.root, tmp_path / "corpus.zip")

    restored = unpack_corpus(archive, tmp_path / "session2")
    calls = CountingClone.calls
    result = generate_fakes(restored, CountingClone.id, SPEC, count=12)

    assert result["skip_exists"] == 6, "phiên mới phải nhận ra công của phiên trước"
    assert CountingClone.calls == calls + 6


def test_pack_then_unpack_round_trips_audio_and_manifest(corpus, tmp_path):
    generate_fakes(corpus, CountingClone.id, SPEC, count=6)
    corpus.save()
    before = {r.utt_id for r in corpus}

    restored = unpack_corpus(pack_corpus(corpus.root, tmp_path / "c.zip"), tmp_path / "out")

    assert {r.utt_id for r in restored} == before
    for rec in restored:
        assert restored.abs_path(rec).exists(), f"thiếu audio: {rec.utt_id}"


# --------------------------------------------------------------- đọc tiến độ
def test_dry_run_reports_what_is_left_without_touching_the_engine(corpus):
    generate_fakes(corpus, CountingClone.id, SPEC, count=5)
    calls = CountingClone.calls

    report = generate_fakes(corpus, CountingClone.id, SPEC, count=12, dry_run=True)

    assert CountingClone.calls == calls, "dry-run không được tổng hợp gì"
    assert report["skip_exists"] == 5 and report["todo"] == 7
    assert report["targets"] == 12 and report["kept"] == 0


def test_dry_run_count_matches_what_a_real_run_then_does(corpus):
    """Con số này dùng để tính thời gian còn lại, nên phải đúng chứ không phải ước lượng."""
    generate_fakes(corpus, CountingClone.id, SPEC, count=4)
    report = generate_fakes(corpus, CountingClone.id, SPEC, count=10, dry_run=True)

    before = CountingClone.calls
    generate_fakes(corpus, CountingClone.id, SPEC, count=10)
    assert CountingClone.calls - before == report["todo"]


def test_dry_run_breaks_progress_down_by_speaker(corpus):
    """Speaker là đơn vị chốt, nên tiến độ phải đọc được ở mức đó."""
    generate_fakes(corpus, CountingClone.id, SPEC, count=4)
    report = generate_fakes(corpus, CountingClone.id, SPEC, count=16, dry_run=True)

    total = report["speakers_done"] + report["speakers_partial"] + report["speakers_todo"]
    assert total == len({r.speaker for r in corpus.reals})
    assert report["speakers_done"] + report["speakers_partial"] > 0


# ------------------------------------------------------- chốt theo speaker
def test_generation_runs_one_speaker_at_a_time(corpus):
    """Chạy xen kẽ thì không bao giờ có thời điểm nào "xong một giọng" để chốt."""
    seen: list[str] = []
    generate_fakes(corpus, CountingClone.id, SPEC, count=12,
                   on_speaker_done=lambda spk, _: seen.append(spk))

    assert seen == sorted(seen), "speaker phải xong lần lượt"
    assert len(seen) == len(set(seen)), f"một speaker bị chốt nhiều lần: {seen}"


def test_every_speaker_still_gets_a_fair_share(corpus):
    """Gom theo speaker chỉ đổi thứ tự CHẠY; phần CHỌN vẫn round-robin như cũ."""
    generate_fakes(corpus, CountingClone.id, SPEC, count=8)
    per_speaker: dict[str, int] = {}
    for f in corpus.fakes:
        per_speaker[f.speaker] = per_speaker.get(f.speaker, 0) + 1

    assert len(per_speaker) >= 4, "fake bị dồn vào quá ít giọng"
    assert max(per_speaker.values()) - min(per_speaker.values()) <= 1


def test_hook_sees_a_manifest_already_written_to_disk(corpus):
    """Hook là để đẩy dữ liệu đi, nên lúc nó chạy manifest phải đã nằm trên đĩa."""
    observed: list[int] = []

    def hook(speaker, stats):
        from_disk = Manifest.load(corpus.root, required=True)
        observed.append(len([f for f in from_disk.fakes if f.speaker == speaker]))

    generate_fakes(corpus, CountingClone.id, SPEC, count=8, on_speaker_done=hook)
    assert observed and all(n > 0 for n in observed), observed


def test_a_failing_hook_does_not_abort_generation(corpus):
    """Mất mạng lúc đẩy dataset thì vẫn còn corpus trên đĩa; bỏ dở GPU thì không."""
    from aidetector.cli import _speaker_hook

    hook = _speaker_hook("exit 3", corpus)
    assert generate_fakes(corpus, CountingClone.id, SPEC, count=8,
                          on_speaker_done=hook)["kept"] == 8


def test_hook_receives_the_speaker_through_the_environment(corpus, tmp_path):
    from aidetector.cli import _speaker_hook

    log_file = tmp_path / "spk.txt"
    hook = _speaker_hook(f'echo "$AIDETECTOR_SPEAKER" >> {log_file}', corpus)
    generate_fakes(corpus, CountingClone.id, SPEC, count=8, on_speaker_done=hook)

    written = [l for l in log_file.read_text().splitlines() if l.strip()]
    assert written and set(written) <= {f.speaker for f in corpus.fakes}


# ------------------------------------------------------ sống sót khi bị ngắt
def test_manifest_survives_an_interrupt_mid_generation(tmp_path, vivos_like, monkeypatch):
    """`manifest.save()` vốn chỉ được gọi khi engine chạy XONG.

    Với vài nghìn mẫu × mấy giây, bị ngắt ở giữa nghĩa là file wav nằm trên đĩa mà
    manifest trống trơn — lần sau `generate` không thấy gì và sinh lại toàn bộ.
    """
    from aidetector import generate as gen_mod

    monkeypatch.setattr(gen_mod, "SAVE_EVERY", 2)

    class Exploding(gen_base.Generator):
        id = "boom"
        kind = gen_base.KIND_CLONE
        native_sample_rate = 24_000
        made = 0

        def voices(self):
            return []

        def synthesize(self, text, voice=None, ref_audio=None, ref_text=None, language=None):
            type(self).made += 1
            if type(self).made > 5:
                raise KeyboardInterrupt("phiên Kaggle hết giờ")
            return np.zeros(24_000 * 4, dtype=np.float32), self.native_sample_rate

    monkeypatch.setitem(gen_base._REGISTRY, Exploding.id, Exploding)
    manifest = Manifest(tmp_path / "corpus")
    ingest_source(manifest, VivosAdapter(), vivos_like, "vivos", SPEC, language="vi")

    with pytest.raises(KeyboardInterrupt):
        generate_fakes(manifest, Exploding.id, SPEC, count=20)

    # Đọc lại từ ĐĨA, không dùng đối tượng trong bộ nhớ.
    from_disk = Manifest.load(manifest.root, required=True)
    assert len(from_disk.fakes) >= 4, "công đã làm không được lưu lại"
    for rec in from_disk.fakes:
        assert from_disk.abs_path(rec).exists()


def test_periodic_save_never_leaves_a_half_written_manifest(tmp_path, vivos_like, monkeypatch):
    from aidetector import generate as gen_mod

    monkeypatch.setattr(gen_mod, "SAVE_EVERY", 1)
    monkeypatch.setitem(gen_base._REGISTRY, CountingClone.id, CountingClone)
    CountingClone.calls = 0

    manifest = Manifest(tmp_path / "corpus")
    ingest_source(manifest, VivosAdapter(), vivos_like, "vivos", SPEC, language="vi")
    generate_fakes(manifest, CountingClone.id, SPEC, count=5)

    from_disk = Manifest.load(manifest.root, required=True)
    assert len(from_disk) == len(manifest)
    assert not any(r.validate() for r in from_disk)
