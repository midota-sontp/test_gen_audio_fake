"""Tái hiện chuỗi lỗi gặp trên Kaggle 2026-08-13 và khoá lại từng mắt xích.

Kịch bản gốc: dataset VIVOS trên Kaggle bị bọc thêm một tầng thư mục
(`/kaggle/input/<slug>/vivos/train/...`). `probe()` chỉ soi đúng thư mục gốc nên
trượt VivosAdapter, rơi về FolderAdapter → 1 speaker → `--per-speaker 5` chỉ giữ 5
utterance → không có transcript → `generate` dùng câu dự phòng và BỊA ra speaker →
chia tập speaker-disjoint đẩy toàn bộ real sang test → val chỉ còn một lớp → EER
`nan` mọi epoch → không checkpoint nào được lưu → `evaluate` chết vì thiếu
`best.pt`, còn ô kết quả chết vì thiếu `metrics.json`.

Mỗi test dưới đây khoá một mắt xích để chuỗi đó không lặp lại.
"""

from __future__ import annotations

import shutil

import pytest

from aidetector.config import Config
from aidetector.corpus.manifest import Manifest
from aidetector.corpus.spec import AudioSpec
from aidetector.features import extract_features
from aidetector.features.backbones import build_backbone
from aidetector.generate import generate_fakes
from aidetector.ingest import detect_adapter, ingest_source
from aidetector.ingest.folder import FolderAdapter
from aidetector.ingest.vivos import VivosAdapter
from aidetector.splits import assign_splits
from aidetector.train import train

SPEC = AudioSpec()


# ─────────────────────────── mắt xích 1: dataset bị bọc thêm tầng
@pytest.mark.parametrize("depth", [1, 2, 3])
def test_detects_dataset_wrapped_in_extra_directories(tmp_path, vivos_like, depth):
    """Kaggle bọc dataset trong <slug>/…; auto-detect phải chui xuống tìm."""
    root = tmp_path / "kaggle_input"
    nested = root.joinpath(*[f"lop{i}" for i in range(depth)], "vivos")
    nested.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(vivos_like, nested)

    cls, score, effective = detect_adapter(root)
    assert cls is VivosAdapter, "vẫn rơi về FolderAdapter như lần chạy hỏng trên Kaggle"
    assert score > 0.5
    assert effective == nested


def test_wrapped_dataset_ingests_with_real_speakers(tmp_path, vivos_like):
    root = tmp_path / "kaggle_input"
    nested = root / "vivos"
    nested.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(vivos_like, nested)

    cls, _, effective = detect_adapter(root)
    manifest = Manifest(tmp_path / "corpus")
    stats = ingest_source(manifest, cls(), effective, "vivos", SPEC)

    assert stats["kept"] > 20
    assert stats["speakers"] >= 8, "gộp hết về 1 speaker chính là lỗi cũ"
    assert all(r.text for r in manifest), "transcript phải được đọc từ prompts.txt"


def test_shallow_root_still_wins_on_a_tie(tmp_path, vivos_like):
    """Không được chui xuống khi thư mục gốc đã khớp — sẽ bỏ sót nửa dataset."""
    cls, _, effective = detect_adapter(vivos_like)
    assert cls is VivosAdapter and effective == vivos_like


# ─────────────────────────── mắt xích 2: speaker suy ra từ thư mục sai tầng
def test_folder_adapter_reads_speaker_from_the_nearest_directory(tmp_path):
    """Bố cục <bộ>/<split>/<speaker>/x.wav — tầng đầu tiên KHÔNG phải speaker."""
    from tests.conftest import speech_like
    import soundfile as sf

    root = tmp_path / "loose"
    for split in ("train", "test"):
        for spk in ("SPK01", "SPK02"):
            d = root / split / spk
            d.mkdir(parents=True)
            sf.write(str(d / "a.wav"), speech_like(4.0, seed=hash(spk) % 999), 16_000,
                     subtype="PCM_16")

    manifest = Manifest(tmp_path / "corpus")
    ingest_source(manifest, FolderAdapter(), root, "loose", SPEC)
    assert {r.speaker for r in manifest} == {"spk01", "spk02"}


# ─────────────────────────── mắt xích 3: câu dự phòng bịa ra speaker
def test_fallback_sentences_do_not_invent_speakers(tmp_path):
    """Corpus không transcript ⇒ fake dự phòng phải để speaker rỗng."""
    from tests.conftest import speech_like

    from aidetector.corpus.schema import Record

    manifest = Manifest(tmp_path / "corpus")
    for i in range(4):                       # real không có transcript
        rec = Record(utt_id=f"r{i}", path="", label="real", source="silent",
                     speaker="mot_nguoi", text="")
        manifest.write_audio(rec, speech_like(4.0, seed=i), SPEC)

    generate_fakes(manifest, "dummy_tts", SPEC, count=8)
    fakes = manifest.fakes
    assert len(fakes) == 8, "mỗi câu dự phòng phải ra một utt_id riêng, không đè nhau"
    assert {r.speaker for r in fakes} == {""}, "bịa speaker sẽ làm hỏng bước chia tập"
    assert all(r.source == "fallback" for r in fakes)


def test_records_without_speaker_are_spread_across_splits(tmp_path):
    from tests.conftest import speech_like

    from aidetector.corpus.schema import Record

    manifest = Manifest(tmp_path / "corpus")
    for i in range(12):                      # real: 6 speaker, mỗi người 2 câu
        rec = Record(utt_id=f"r{i}", path="", label="real", source="s",
                     speaker=f"spk{i % 6}", text="")
        manifest.write_audio(rec, speech_like(4.0, seed=i), SPEC)
    for i in range(12):                      # fake không speaker
        rec = Record(utt_id=f"f{i}", path="", label="fake", source="fallback",
                     generator="dummy_tts:v", speaker="")
        manifest.write_audio(rec, speech_like(4.0, seed=100 + i), SPEC)

    assign_splits(manifest, ratios=(0.5, 0.25, 0.25), seed=5)
    for split in ("train", "val", "test"):
        recs = manifest.by_split(split)
        assert any(r.label == "fake" for r in recs), f"{split} mất hẳn lớp fake"


# ─────────────────────────── mắt xích 4: split thiếu lớp phải chặn ngay
def test_split_raises_when_a_class_disappears(tmp_path):
    """Toàn bộ real thuộc một speaker ⇒ dồn vào một split ⇒ phải báo lỗi ngay."""
    from tests.conftest import speech_like

    from aidetector.corpus.schema import Record

    manifest = Manifest(tmp_path / "corpus")
    for i in range(5):
        rec = Record(utt_id=f"r{i}", path="", label="real", source="s",
                     speaker="chi_mot_nguoi", text="")
        manifest.write_audio(rec, speech_like(4.0, seed=i), SPEC)
    for i in range(15):
        rec = Record(utt_id=f"f{i}", path="", label="fake", source="s",
                     generator="dummy_tts:v", speaker=f"gia{i}")
        manifest.write_audio(rec, speech_like(4.0, seed=50 + i), SPEC)

    with pytest.raises(ValueError) as err:
        assign_splits(manifest, seed=1)
    message = str(err.value)
    assert "thiếu một trong hai lớp" in message
    assert "speaker" in message, "lỗi phải chỉ ra nguyên nhân, không chỉ báo hỏng"


def test_split_can_be_lenient_when_asked(tmp_path):
    from tests.conftest import speech_like

    from aidetector.corpus.schema import Record

    manifest = Manifest(tmp_path / "corpus")
    for i in range(6):
        rec = Record(utt_id=f"r{i}", path="", label="real", source="s", speaker=f"s{i}")
        manifest.write_audio(rec, speech_like(4.0, seed=i), SPEC)
    report = assign_splits(manifest, seed=1, strict=False)   # không có fake nào
    assert report["speaker_leaks"] == []


# ─────────────────────────── mắt xích 5: train không được im lặng bỏ qua
def _prepare(tmp_path, drop_class_from_val: bool):
    from tests.conftest import speech_like

    from aidetector.corpus.schema import Record

    manifest = Manifest(tmp_path / "corpus")
    for i in range(12):
        label = "real" if i % 2 == 0 else "fake"
        rec = Record(utt_id=f"u{i}", path="", label=label, source="s",
                     generator="" if label == "real" else "dummy_tts:v",
                     speaker=f"spk{i}", split="train" if i < 8 else "val")
        manifest.write_audio(rec, speech_like(4.0, seed=i), SPEC)
    if drop_class_from_val:
        for rec in manifest.by_split("val"):
            if rec.label == "fake":
                rec.split = "train"

    backbone = build_backbone({"name": "dummy", "output_layer": 0, "pooling": "mean"}, "cpu")
    extract_features(manifest, backbone, SPEC, cache_root=tmp_path / "feat")
    cfg = Config({"model": {"head": "linear"},
                  "train": {"epochs": 3, "batch_size": 4, "lr": 1e-3}})
    return manifest, backbone, cfg


def test_train_fails_loudly_when_val_has_one_class(tmp_path):
    """Đây chính là chỗ lần chạy Kaggle im lặng cho ra 'best epoch -1'."""
    manifest, backbone, cfg = _prepare(tmp_path, drop_class_from_val=True)
    with pytest.raises(RuntimeError, match="không có mẫu"):
        train(manifest, backbone, cfg, cache_root=tmp_path / "feat",
              checkpoint_dir=tmp_path / "ckpt", report_dir=tmp_path / "rep", device="cpu")
    assert not (tmp_path / "ckpt" / "best.pt").exists()


def test_train_saves_a_checkpoint_when_data_is_sane(tmp_path):
    manifest, backbone, cfg = _prepare(tmp_path, drop_class_from_val=False)
    summary = train(manifest, backbone, cfg, cache_root=tmp_path / "feat",
                    checkpoint_dir=tmp_path / "ckpt", report_dir=tmp_path / "rep",
                    device="cpu")
    assert (tmp_path / "ckpt" / "best.pt").exists()
    assert summary["best_epoch"] >= 1


def test_non_finite_monitor_never_leaves_the_run_without_a_checkpoint(tmp_path, monkeypatch):
    """Chỉ số theo dõi hỏng ⇒ lùi về val_loss, tuyệt đối không kết thúc tay trắng."""
    import aidetector.train as train_module

    manifest, backbone, cfg = _prepare(tmp_path, drop_class_from_val=False)
    monkeypatch.setattr(train_module, "compute_eer", lambda *_: (float("nan"), float("nan")))
    summary = train(manifest, backbone, cfg, cache_root=tmp_path / "feat",
                    checkpoint_dir=tmp_path / "ckpt", report_dir=tmp_path / "rep",
                    device="cpu")
    assert (tmp_path / "ckpt" / "best.pt").exists()
    assert summary["best_epoch"] >= 1
