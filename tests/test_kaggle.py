"""Chạy trên Kaggle: nhận diện môi trường, config kế thừa, đóng gói corpus."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from aidetector.config import Config
from aidetector.corpus.manifest import MANIFEST_NAME, Manifest
from aidetector.corpus.spec import AudioSpec
from aidetector.env import KAGGLE, LOCAL, detect_platform, find_kaggle_datasets, free_space_gb
from aidetector.generate import generate_fakes
from aidetector.ingest import ingest_source
from aidetector.ingest.vivos import VivosAdapter
from aidetector.packaging import pack_corpus, unpack_corpus

SPEC = AudioSpec()


# ------------------------------------------------------------------ môi trường
def test_detects_kaggle_from_env(monkeypatch):
    monkeypatch.setenv("KAGGLE_KERNEL_RUN_TYPE", "Interactive")
    platform = detect_platform()
    assert platform.name == KAGGLE
    assert platform.work_dir == Path("/kaggle/working")
    assert platform.input_dir == Path("/kaggle/input")
    assert platform.is_ephemeral
    assert platform.session_hours == 9.0


def test_local_is_not_ephemeral(monkeypatch):
    monkeypatch.delenv("KAGGLE_KERNEL_RUN_TYPE", raising=False)
    monkeypatch.delenv("COLAB_GPU", raising=False)
    monkeypatch.setattr(Path, "is_dir", lambda self: False)
    platform = detect_platform()
    assert platform.name == LOCAL
    assert not platform.is_ephemeral


def test_free_space_walks_up_to_an_existing_parent(tmp_path):
    assert free_space_gb(tmp_path / "chua" / "ton" / "tai") > 0


def test_kaggle_dataset_listing_is_empty_off_kaggle():
    assert find_kaggle_datasets() == [] or Path("/kaggle/input").is_dir()


# --------------------------------------------------------------------- config
def test_kaggle_config_inherits_defaults_and_overrides_paths():
    cfg = Config.load("configs/kaggle.yaml")
    # Ghi đè
    assert cfg["paths.corpus"] == "/kaggle/working/corpus"
    assert cfg["features.device"] == "cuda"
    assert "omnivoice" in cfg["generate.engines"]
    assert cfg["features.batch_size"] == 32
    # Kế thừa từ default.yaml — chuẩn audio và backbone không được đổi
    assert cfg["audio.sample_rate"] == 16_000
    assert (cfg["audio.min_seconds"], cfg["audio.max_seconds"]) == (3.0, 10.0)
    assert cfg["features.backbone.name"] == "wavlm"
    assert cfg["augment.ops.codec"]["p"] == 0.5


def test_kaggle_config_paths_all_live_under_working():
    cfg = Config.load("configs/kaggle.yaml")
    for key in ("corpus", "features", "checkpoints", "reports"):
        assert cfg[f"paths.{key}"].startswith("/kaggle/working/")


# -------------------------------------------------------------------- đóng gói
@pytest.fixture
def corpus(tmp_path, vivos_like):
    manifest = Manifest(tmp_path / "corpus")
    ingest_source(manifest, VivosAdapter(), vivos_like, "vivos", SPEC)
    generate_fakes(manifest, "dummy_tts", SPEC, count=6)
    manifest.save()
    return manifest


def test_pack_then_unpack_round_trips(corpus, tmp_path):
    archive = pack_corpus(corpus.root, tmp_path / "corpus.zip")
    assert archive.exists()

    restored = unpack_corpus(archive, tmp_path / "restored")
    assert len(restored) == len(corpus)
    for rec in corpus:
        other = restored.get(rec.utt_id)
        assert other is not None and other.to_row() == rec.to_row()
        assert restored.abs_path(other).exists()


def test_pack_writes_manifest_at_archive_root(corpus, tmp_path):
    archive = pack_corpus(corpus.root, tmp_path / "corpus.zip")
    with zipfile.ZipFile(archive) as zf:
        names = zf.namelist()
    assert MANIFEST_NAME in names
    assert len(names) == len(corpus) + 1
    assert all(n == MANIFEST_NAME or n.startswith("audio/") for n in names)


def test_pack_can_skip_audio_for_a_metadata_only_archive(corpus, tmp_path):
    archive = pack_corpus(corpus.root, tmp_path / "meta.zip", include_audio=False)
    with zipfile.ZipFile(archive) as zf:
        assert zf.namelist() == [MANIFEST_NAME]


def test_unpack_rejects_an_unrelated_zip(tmp_path):
    bogus = tmp_path / "bogus.zip"
    with zipfile.ZipFile(bogus, "w") as zf:
        zf.writestr("readme.txt", "không phải corpus")
    with pytest.raises(ValueError, match="không phải gói corpus"):
        unpack_corpus(bogus, tmp_path / "out")


def test_unpack_drops_records_whose_audio_is_missing(corpus, tmp_path):
    archive = pack_corpus(corpus.root, tmp_path / "meta.zip", include_audio=False)
    restored = unpack_corpus(archive, tmp_path / "restored")
    assert len(restored) == 0            # manifest có nhưng không file nào ⇒ dọn sạch


def test_unpack_is_incremental(corpus, tmp_path):
    archive = pack_corpus(corpus.root, tmp_path / "corpus.zip")
    target = tmp_path / "restored"
    unpack_corpus(archive, target)
    first = {p: p.stat().st_mtime_ns for p in target.rglob("*.wav")}
    unpack_corpus(archive, target)       # lần hai không ghi lại file đã có
    assert {p: p.stat().st_mtime_ns for p in target.rglob("*.wav")} == first


# ------------------------------------------------------------------- notebook
def test_kaggle_notebook_is_valid_and_uses_the_kaggle_config():
    nb = json.loads(Path("notebooks/kaggle_pipeline.ipynb").read_text(encoding="utf-8"))
    assert nb["nbformat"] == 4
    assert nb["cells"], "notebook rỗng"
    for cell in nb["cells"]:
        assert cell["cell_type"] in ("markdown", "code")
        assert isinstance(cell["source"], list)

    text = "".join("".join(c["source"]) for c in nb["cells"])
    assert "configs/kaggle.yaml" in text
    for stage in ("ingest", "generate", "split", "augment", "features", "train", "evaluate"):
        assert f"aidetector {stage}" in text, f"notebook thiếu stage {stage}"
    # split phải đứng trước augment, đúng như thứ tự pipeline
    assert text.index("aidetector split") < text.index("aidetector augment")
    # phải nhắc người dùng lưu lại trước khi hết phiên
    assert "aidetector pack" in text and "unpack" in text
