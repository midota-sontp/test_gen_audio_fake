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


# --------------------------------------------------- cấu trúc thư mục chuẩn của corpus
def test_layout_puts_every_dataset_in_its_own_folder(tmp_path, vivos_like, monkeypatch):
    """<bộ> / real|fake / [engine /] <speaker> / NNNN.wav — mỗi bộ dữ liệu một thư mục."""
    import numpy as np

    from aidetector.corpus.manifest import Manifest
    from aidetector.generate import base as gen_base, generate_fakes
    from aidetector.ingest import ingest_source
    from aidetector.ingest.vivos import VivosAdapter

    class Clone(gen_base.Generator):
        id = "layout_spy"
        kind = gen_base.KIND_CLONE
        native_sample_rate = 24_000

        def voices(self):
            return []

        def synthesize(self, text, voice=None, ref_audio=None, ref_text=None, language=None):
            return np.zeros(24_000 * 4, dtype=np.float32), self.native_sample_rate

    monkeypatch.setitem(gen_base._REGISTRY, Clone.id, Clone)
    spec = AudioSpec()
    manifest = Manifest(tmp_path / "corpus")
    ingest_source(manifest, VivosAdapter(), vivos_like, "vivos", spec)
    generate_fakes(manifest, Clone.id, spec, count=8)

    for rec in manifest.reals:
        tang = rec.path.split("/")
        assert tang[0] == "vivos" and tang[1] == "real" and tang[2] == rec.speaker
        assert len(tang) == 4 and tang[3].endswith(".wav") and tang[3][:-4].isdigit()

    for rec in manifest.fakes:
        tang = rec.path.split("/")
        # Fake nằm trong thư mục của CHÍNH BỘ đã sinh ra nó (`source` thừa hưởng từ real
        # gốc), rồi mới tách theo engine. Tầng cuối vẫn là speaker của real gốc — nhờ vậy
        # đứng ở một giọng là thấy cả hai lớp của giọng đó cạnh nhau.
        assert tang[0] == "vivos" and tang[1] == "fake"
        assert tang[2] == Clone.id and tang[3] == rec.speaker


def test_index_is_assigned_once_and_never_reshuffled(tmp_path, vivos_like):
    """Số thứ tự nằm trong cột `path`. Suy lại từ thứ tự duyệt là phá idempotency."""
    from aidetector.corpus.manifest import Manifest
    from aidetector.ingest import ingest_source
    from aidetector.ingest.vivos import VivosAdapter

    spec = AudioSpec()
    corpus = tmp_path / "corpus"
    manifest = Manifest(corpus)
    ingest_source(manifest, VivosAdapter(), vivos_like, "vivos", spec)
    manifest.save()
    truoc = {r.utt_id: r.path for r in manifest}

    # Nạp lại từ đĩa rồi ingest tiếp: bản đã có phải giữ NGUYÊN đường dẫn.
    lai = Manifest.load(corpus, required=True)
    ingest_source(lai, VivosAdapter(), vivos_like, "vivos", spec)
    sau = {r.utt_id: r.path for r in lai}

    assert sau == truoc, "đường dẫn bị đánh số lại giữa hai lần chạy"
    assert len({p for p in truoc.values()}) == len(truoc), "hai bản ghi trùng đường dẫn"


def test_overwriting_a_record_keeps_its_place(tmp_path, vivos_like):
    """`generate --overwrite` cấp số mới là để lại file mồ côi và làm cây phình dần."""
    import numpy as np

    from aidetector.corpus.manifest import Manifest
    from aidetector.ingest import ingest_source
    from aidetector.ingest.vivos import VivosAdapter

    spec = AudioSpec()
    manifest = Manifest(tmp_path / "corpus")
    ingest_source(manifest, VivosAdapter(), vivos_like, "vivos", spec)

    rec = manifest.reals[0]
    cho_cu = rec.path
    manifest.write_audio(rec, np.zeros(spec.sample_rate * 4, dtype=np.float32), spec)
    assert rec.path == cho_cu


def test_migrate_moves_an_old_layout_and_is_idempotent(tmp_path, vivos_like):
    """Cột `path` là nguồn sự thật nên corpus cũ vẫn ĐỌC được; `migrate` để dọn cho đồng nhất."""
    from pathlib import Path

    from aidetector.corpus.manifest import Manifest
    from aidetector.ingest import ingest_source
    from aidetector.ingest.vivos import VivosAdapter

    spec = AudioSpec()
    corpus = tmp_path / "corpus"
    manifest = Manifest(corpus)
    ingest_source(manifest, VivosAdapter(), vivos_like, "vivos", spec)

    # Giả lập cấu trúc cũ: audio/real/<source>/<speaker>/<utt_id>.wav
    for rec in list(manifest):
        cu = Path("audio") / rec.path.replace(rec.path.split("/")[-1], f"{rec.utt_id}.wav")
        (corpus / cu).parent.mkdir(parents=True, exist_ok=True)
        (corpus / rec.path).rename(corpus / cu)
        rec.path = str(cu)
    manifest.save()

    ket_qua = Manifest.load(corpus, required=True)
    r1 = ket_qua.migrate_layout()
    assert r1["moved"] == len(ket_qua) and r1["missing"] == 0
    for rec in ket_qua:
        assert rec.path.startswith("vivos/real/")
        assert ket_qua.abs_path(rec).exists()

    r2 = ket_qua.migrate_layout()
    assert r2["moved"] == 0 and r2["kept"] == len(ket_qua), "chạy lại phải không xáo gì"


def test_migrate_survives_being_interrupted(tmp_path, vivos_like):
    """Bị ngắt giữa lúc dời file là đường mất dữ liệu thật: file ở chỗ mới, manifest
    trỏ chỗ cũ ⇒ `prune_missing` coi là mất và loại khỏi corpus.

    Manifest chỉ lưu SAU khi dời xong và phép cấp số là tất định, nên chạy lại tính ra
    đúng những đường dẫn cũ và nhận lại phần đã dời.
    """
    from pathlib import Path

    from aidetector.corpus.manifest import Manifest
    from aidetector.ingest import ingest_source
    from aidetector.ingest.vivos import VivosAdapter

    spec = AudioSpec()
    corpus = tmp_path / "corpus"
    manifest = Manifest(corpus)
    ingest_source(manifest, VivosAdapter(), vivos_like, "vivos", spec)
    for rec in list(manifest):
        cu = Path("audio") / rec.path.rsplit("/", 1)[0] / f"{rec.utt_id}.wav"
        (corpus / cu).parent.mkdir(parents=True, exist_ok=True)
        (corpus / rec.path).rename(corpus / cu)
        rec.path = str(cu)
    manifest.save()
    tong = len(manifest)

    # Lượt 1 bị ngắt: dời được một nửa, manifest KHÔNG được lưu.
    dang_do = Manifest.load(corpus, required=True)
    nua = len(dang_do) // 2
    for i, rec in enumerate(sorted(dang_do, key=lambda r: r.utt_id)):
        if i >= nua:
            break
        moi = dang_do.allocate_path(rec)
        (corpus / moi).parent.mkdir(parents=True, exist_ok=True)
        (corpus / rec.path).rename(corpus / moi)

    # Lượt 2 chạy lại trên manifest chưa đổi.
    lai = Manifest.load(corpus, required=True)
    r = lai.migrate_layout()
    assert r["resumed"] == nua, f"không nhận lại phần đã dời: {r}"
    assert r["missing"] == 0, "báo mất file trong khi chúng nằm ở chỗ mới"
    assert r["moved"] == tong - nua
    for rec in lai:
        assert lai.abs_path(rec).exists(), rec.utt_id
