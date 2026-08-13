"""Tự nhận diện loại dataset + các adapter tổng quát."""

from __future__ import annotations

import csv

import pytest
import soundfile as sf

from aidetector.corpus.manifest import Manifest
from aidetector.corpus.schema import LABEL_FAKE, LABEL_REAL
from aidetector.corpus.spec import AudioSpec
from aidetector.ingest import detect_adapter, ingest_source
from aidetector.ingest.common_voice import CommonVoiceAdapter
from aidetector.ingest.folder import FolderAdapter, LabeledFolderAdapter
from aidetector.ingest.vivos import VivosAdapter
from tests.conftest import SR, speech_like

SPEC = AudioSpec()


def _wav(path, seconds=4.0, seed=0):
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), speech_like(seconds, seed=seed), SR, subtype="PCM_16")


# ------------------------------------------------------------------ nhận diện
def test_detects_vivos(vivos_like):
    cls, score, root = detect_adapter(vivos_like)
    assert cls is VivosAdapter and score > 0.5
    assert root == vivos_like


def test_detects_common_voice(tmp_path):
    root = tmp_path / "cv-vi"
    _wav(root / "clips" / "a.wav", seed=1)
    _wav(root / "clips" / "b.wav", seed=2)
    with (root / "validated.tsv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh, delimiter="\t")
        writer.writerow(["client_id", "path", "sentence"])
        writer.writerow(["abcdef0123456789", "a.wav", "câu thứ nhất trong bộ dữ liệu"])
        writer.writerow(["abcdef0123456789", "b.wav", "câu thứ hai trong bộ dữ liệu"])
    cls, score, effective = detect_adapter(root)
    assert cls is CommonVoiceAdapter and score > 0.8 and effective == root

    manifest = Manifest(tmp_path / "corpus")
    ingest_source(manifest, cls(), root, "cv", SPEC)
    assert len(manifest) == 2
    assert all(r.speaker == "abcdef0123456789" for r in manifest)
    assert all(r.text for r in manifest)


def test_detects_labeled_folder(tmp_path):
    root = tmp_path / "mixed"
    _wav(root / "real" / "spk1" / "a.wav", seed=3)
    _wav(root / "fake" / "some_engine" / "b.wav", seed=4)
    cls, score, effective = detect_adapter(root)
    assert cls is LabeledFolderAdapter and score > 0.8 and effective == root

    manifest = Manifest(tmp_path / "corpus")
    ingest_source(manifest, cls(), root, "mixed", SPEC)
    labels = {r.label for r in manifest}
    assert labels == {LABEL_REAL, LABEL_FAKE}
    fake = next(r for r in manifest if r.label == LABEL_FAKE)
    assert fake.generator == "some_engine"       # tên thư mục thành tên engine


def test_falls_back_to_plain_folder(tmp_path):
    root = tmp_path / "loose"
    _wav(root / "spkA" / "x.wav", seed=5)
    _wav(root / "spkB" / "y.wav", seed=6)
    cls, _, effective = detect_adapter(root)
    assert cls is FolderAdapter and effective == root

    manifest = Manifest(tmp_path / "corpus")
    ingest_source(manifest, cls(), root, "loose", SPEC)
    assert {r.speaker for r in manifest} == {"spka", "spkb"}


def test_empty_directory_is_rejected(tmp_path):
    (tmp_path / "nothing").mkdir()
    with pytest.raises(ValueError, match="Không nhận diện được"):
        detect_adapter(tmp_path / "nothing")


# --------------------------------------------------------------- transcript
@pytest.mark.parametrize(
    ("filename", "content"),
    [
        ("metadata.csv", "x|nội dung câu nói của file x\n"),
        ("prompts.txt", "x nội dung câu nói của file x\n"),
    ],
)
def test_folder_adapter_finds_transcripts(tmp_path, filename, content):
    root = tmp_path / f"src-{filename}"
    _wav(root / "spk" / "x.wav", seed=7)
    (root / filename).write_text(content, encoding="utf-8")
    manifest = Manifest(tmp_path / f"corpus-{filename}")
    ingest_source(manifest, FolderAdapter(), root, "src", SPEC)
    assert next(iter(manifest)).text == "nội dung câu nói của file x"


def test_sidecar_txt_is_used_as_transcript(tmp_path):
    root = tmp_path / "sidecar"
    _wav(root / "spk" / "x.wav", seed=8)
    (root / "spk" / "x.txt").write_text("câu nói kèm theo file", encoding="utf-8")
    manifest = Manifest(tmp_path / "corpus")
    ingest_source(manifest, FolderAdapter(), root, "src", SPEC)
    assert next(iter(manifest)).text == "câu nói kèm theo file"


def test_vivos_split_hint_is_carried_through(vivos_like):
    manifest = Manifest(vivos_like.parent / "corpus")
    ingest_source(manifest, VivosAdapter(), vivos_like, "vivos", SPEC)
    hints = {r.split for r in manifest}
    assert hints == {"train", "test"}
    assert all(r.speaker.startswith("vivosdev") for r in manifest.by_split("test"))
