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


def test_kaggle_config_never_hardcodes_cuda():
    """Quên bật Accelerator không được làm chết pipeline giữa chừng."""
    cfg = Config.load("configs/kaggle.yaml")
    for key in ("generate.device", "features.device", "train.device"):
        assert cfg[key] == "auto", f"{key} phải là auto, không ghi cứng thiết bị"


def test_unavailable_device_falls_back_with_a_warning(monkeypatch, caplog):
    import torch

    from aidetector.utils import resolve_device

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with caplog.at_level("WARNING"):
        device = resolve_device("cuda")
    assert device != "cuda"
    assert "Accelerator" in caplog.text


def test_explicit_cpu_is_always_respected():
    from aidetector.utils import resolve_device

    assert resolve_device("cpu") == "cpu"


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
NOTEBOOK = Path("notebooks/aidetector_kaggle.ipynb")


@pytest.fixture(scope="module")
def notebook() -> dict:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def notebook_text(notebook) -> str:
    return "".join("".join(c["source"]) for c in notebook["cells"])


def test_there_is_exactly_one_notebook():
    """Một file duy nhất để import — hai bản song song chỉ gây nhầm."""
    assert sorted(p.name for p in Path("notebooks").glob("*.ipynb")) == [NOTEBOOK.name]


def test_notebook_is_valid_nbformat(notebook):
    assert notebook["nbformat"] == 4
    assert notebook["cells"], "notebook rỗng"
    for cell in notebook["cells"]:
        assert cell["cell_type"] in ("markdown", "code")
        assert isinstance(cell["source"], list)


def test_every_code_cell_parses_as_python(notebook):
    """Bỏ dòng magic `!...` của IPython rồi kiểm tra cú pháp Python thuần."""
    import ast

    for i, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        pure = "\n".join(l for l in source.splitlines() if not l.lstrip().startswith("!"))
        ast.parse(pure)  # ném SyntaxError kèm số dòng nếu hỏng


def test_notebook_is_self_contained(notebook_text):
    """Không được phụ thuộc vào việc clone repo hay mount dataset chứa code."""
    assert "_PAYLOAD" in notebook_text and "base64.b64decode" in notebook_text
    assert "git clone" not in notebook_text


def _load_builder():
    """Nạp scripts/build_kaggle_notebook.py (thư mục scripts/ không phải package)."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "build_kaggle_notebook", Path("scripts/build_kaggle_notebook.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_payload_is_deterministic():
    """Cùng mã nguồn phải cho cùng payload — nếu không, notebook diff bẩn mỗi lần build."""
    builder = _load_builder()
    files = builder.collect_files()
    assert builder.build_payload(files) == builder.build_payload(files)


def test_notebook_payload_matches_the_current_source(notebook_text):
    """Payload nhúng phải khớp code trong repo; lệch ⇒ chạy lại script build."""
    builder = _load_builder()
    _, sha, _ = builder.build_payload(builder.collect_files())
    assert sha in notebook_text, (
        "notebook đã lệch với mã nguồn — chạy: python scripts/build_kaggle_notebook.py"
    )


def test_notebook_covers_every_stage(notebook_text):
    assert "configs/kaggle.yaml" in notebook_text
    for stage in ("ingest", "generate", "validate", "split", "augment",
                  "features", "train", "evaluate", "pack"):
        assert f"aidetector {stage}" in notebook_text, f"notebook thiếu stage {stage}"


def test_dataset_phase_comes_before_training_phase(notebook_text):
    """Phải tạo + kiểm tra dataset xong mới tới huấn luyện."""
    assert notebook_text.index("PHẦN A") < notebook_text.index("PHẦN B")
    assert notebook_text.index("aidetector generate") < notebook_text.index("aidetector train")
    # split trước augment: bản augment chỉ được sinh cho train
    assert notebook_text.index("aidetector split") < notebook_text.index("aidetector augment")
    # đóng gói dataset nằm trong phần A, trước khi huấn luyện
    assert notebook_text.index("aidetector pack") < notebook_text.index("aidetector train")


def test_dataset_phase_has_a_smoke_switch_and_inspection(notebook_text):
    assert "SMOKE = True" in notebook_text
    assert "aidetector validate" in notebook_text
    # nghe thử + nhìn phổ trước khi tốn thời gian train
    assert "Audio(" in notebook_text and "specgram" in notebook_text
    # kiểm tra fake có real đối chứng
    assert "ref_utt_id" in notebook_text
