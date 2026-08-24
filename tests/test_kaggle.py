"""Chạy trên Kaggle: nhận diện môi trường, config kế thừa, đóng gói corpus."""

from __future__ import annotations

import json
import sys
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


@pytest.fixture(scope="module")
def notebook_code(notebook) -> str:
    """Chỉ phần ô code — markdown chứa ví dụ minh hoạ, không phải thứ được chạy."""
    return "".join("".join(c["source"])
                   for c in notebook["cells"] if c["cell_type"] == "code")


def _cells_with(notebook, needle) -> list[int]:
    return [i for i, c in enumerate(notebook["cells"])
            if c["cell_type"] == "code" and needle in "".join(c["source"])]


def _cell_src(notebook, needle) -> str:
    found = _cells_with(notebook, needle)
    assert found, f"không ô code nào chứa {needle!r}"
    return "".join(notebook["cells"][found[0]]["source"])


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


STAGES_IN_NOTEBOOK = ("ingest", "generate", "validate", "split", "augment",
                      "features", "train", "evaluate", "pack", "detect")


def test_notebook_covers_every_stage(notebook_text):
    assert "configs/kaggle.yaml" in notebook_text
    for stage in STAGES_IN_NOTEBOOK:
        assert f'run("{stage}"' in notebook_text, f"notebook thiếu stage {stage}"


def test_stages_run_through_the_fail_fast_helper(notebook_code):
    """`!python …` lỗi vẫn để notebook chạy tiếp — mọi stage phải đi qua run()."""
    assert "def run(*args, optional=False):" in notebook_code
    assert "raise SystemExit(" in notebook_code
    for stage in STAGES_IN_NOTEBOOK:
        assert f"!python -m aidetector {stage}" not in notebook_code, (
            f"stage {stage} vẫn gọi bằng `!python`, lỗi sẽ bị nuốt"
        )


def test_engine_is_optional_only_when_something_else_makes_fakes(notebook_code):
    """`optional` phải phụ thuộc việc còn engine khác gánh lớp fake hay không.

    OmniVoice cần GPU và tải checkpoint vài GB, nên khi Piper/Kokoro còn bật thì bỏ qua
    lỗi của nó là đúng. Nhưng tắt TTS rồi thì nó là nguồn fake DUY NHẤT: bỏ qua lúc đó
    là đưa cả phần huấn luyện vào một corpus không có lớp fake nào.
    """
    start = notebook_code.index('_hook = ["--after-speaker"')
    call = notebook_code[start:notebook_code.index("\n\n", start)]
    assert "optional=bool(TTS_ENGINES)" in call, call


def test_generate_syncs_at_speaker_boundaries(notebook_code):
    """Mốc đẩy dữ liệu là ranh giới speaker, và chỉ gắn khi đồng bộ đã sẵn sàng."""
    assert '"--after-speaker"' in notebook_code
    assert "if SYNC_READY else []" in notebook_code


def test_progress_is_checked_before_the_long_generation(notebook_code):
    """Bước sinh dài nhiều giờ — phải biết còn thiếu bao nhiêu TRƯỚC khi bắt đầu."""
    assert '"--dry-run"' in notebook_code
    assert notebook_code.index('"--dry-run"') < notebook_code.index('"--after-speaker"')


def test_dataset_phase_comes_before_training_phase(notebook_text):
    """Phải tạo + kiểm tra dataset xong mới tới huấn luyện."""
    assert notebook_text.index("PHẦN A") < notebook_text.index("PHẦN B")
    assert notebook_text.index('run("generate"') < notebook_text.index('run("train")')
    # split trước augment: bản augment chỉ được sinh cho train
    assert notebook_text.index('run("split")') < notebook_text.index('run("augment"')
    # đóng gói dataset nằm trong phần A, trước khi huấn luyện
    assert notebook_text.index('run("pack"') < notebook_text.index('run("train")')


def _bootstrap_cell(notebook: dict) -> str:
    return "".join(next(c["source"] for c in notebook["cells"]
                        if c["cell_type"] == "code" and "_PAYLOAD" in "".join(c["source"])))


def test_bootstrap_drops_stale_modules_from_the_kernel(notebook, tmp_path, monkeypatch):
    """Kernel Kaggle sống xuyên phiên: mã vừa bung phải thắng module đã import.

    Không gỡ sys.modules thì Python giữ nguyên bản cũ và ném những lỗi vô lý kiểu
    "cannot import name X" dù X nằm sờ sờ trong file.
    """
    import sys

    work = tmp_path / "ai-detector"
    old = work / "aidetector"
    (old / "corpus").mkdir(parents=True)
    (old / "__init__.py").write_text("")
    (old / "corpus" / "__init__.py").write_text("")
    (old / "corpus" / "spec.py").write_text("# bản cũ, chưa có AUDIO_EXTENSIONS\n")
    (old / "chi_co_o_ban_cu.py").write_text("")

    monkeypatch.syspath_prepend(str(work))
    monkeypatch.chdir(tmp_path)
    saved = {k: v for k, v in sys.modules.items() if k.startswith("aidetector")}
    for key in saved:
        del sys.modules[key]
    try:
        import aidetector.corpus.spec as stale_module

        assert not hasattr(stale_module, "AUDIO_EXTENSIONS")

        cell = _bootstrap_cell(notebook).replace(
            'Path("/kaggle/working/ai-detector")', f"Path({str(work)!r})"
        )
        exec(compile(cell, "bootstrap", "exec"), {})

        assert not any(m.startswith("aidetector") for m in sys.modules), \
            "module cũ vẫn còn trong kernel"
        # File chỉ tồn tại ở bản cũ phải bị dọn, không để lẫn vào bản mới.
        assert not (old / "chi_co_o_ban_cu.py").exists()

        from aidetector.corpus.spec import AUDIO_EXTENSIONS

        assert ".wav" in AUDIO_EXTENSIONS
    finally:
        for key in [k for k in sys.modules if k.startswith("aidetector")]:
            del sys.modules[key]
        sys.modules.update(saved)


def _run_picker(notebook, tmp_path, namespace):
    """Chạy ô A1 với /kaggle/input trỏ vào tmp_path.

    MAKE_DATASET phải do người gọi đặt: chính nó quyết định ô có dò dataset hay không.
    """
    cell = _cell_src(notebook, "detect_adapter(")
    exec(compile(cell.replace('Path("/kaggle/input")', f'Path({str(tmp_path)!r})'),
                 "picker", "exec"), namespace)
    return namespace


def test_dataset_picker_scores_every_mount(notebook, tmp_path):
    """Lấy bừa thư mục đầu tiên sẽ chết khi dataset rỗng đứng trước dataset thật."""
    (tmp_path / "aaa-rong").mkdir()                     # rỗng, đứng trước theo abc
    good = tmp_path / "zzz-vivos" / "vivos"
    good.mkdir(parents=True)
    for split in ("train", "test"):
        (good / split / "waves" / "SPK1").mkdir(parents=True)
        (good / split / "prompts.txt").write_text("SPK1_R001 xin chào", encoding="utf-8")

    namespace = _run_picker(notebook, tmp_path, {"MAKE_DATASET": True})
    assert namespace["RAW"] == str(tmp_path / "zzz-vivos")


def test_dataset_picker_stops_with_details_when_nothing_usable(notebook, tmp_path):
    (tmp_path / "chi-co-parquet").mkdir()
    (tmp_path / "chi-co-parquet" / "train.parquet").write_bytes(b"PAR1")

    with pytest.raises(SystemExit) as err:
        _run_picker(notebook, tmp_path, {"MAKE_DATASET": True})
    assert "Không dataset nào chứa audio" in str(err.value)
    assert ".parquet" in str(err.value)


def test_picker_leaves_the_real_dataset_alone_when_only_training(notebook, tmp_path):
    """MODE='train' không mount VIVOS — đòi cho được dataset real ở đây là dừng oan."""
    said = []
    namespace = _run_picker(notebook, tmp_path,
                            {"MAKE_DATASET": False, "skipped": said.append})
    assert namespace["RAW"] is None and said, "phải bỏ qua và nói rõ lý do"


def test_dataset_phase_has_a_smoke_switch_and_inspection(notebook_text):
    assert "SMOKE = True" in notebook_text
    assert 'run("validate")' in notebook_text
    # nghe thử + nhìn phổ trước khi tốn thời gian train
    assert "Audio(" in notebook_text and "specgram" in notebook_text
    # kiểm tra fake có real đối chứng
    assert "ref_utt_id" in notebook_text


def test_generated_sync_script_is_valid_python(notebook):
    """Script đồng bộ được sinh ra bằng f-string lồng trong ô notebook.

    Nó chỉ chạy trên Kaggle, giữa lượt sinh kéo dài nhiều giờ — hỏng cú pháp ở đó thì
    chỉ biết sau khi đã mất công. Sai một lớp `{{}}` là đủ vỡ, nên dựng ra và kiểm ở đây.
    """
    import ast
    import textwrap

    source = next("".join(c["source"]) for c in notebook["cells"]
                  if "SYNC_SCRIPT" in "".join(c["source"]))
    start = source.index("textwrap.dedent(")
    end = source.index("'''))", start) + len("'''")
    script = eval(source[start:end] + ")",  # noqa: S307 — chuỗi do chính repo sinh ra
                  {"textwrap": textwrap, "DATASET_ID": "owner/slug",
                   "SYNC_EVERY_MINUTES": 20})

    ast.parse(script)
    assert "kaggle" in script and "--force" in script
    # Đẩy toàn bộ: real + fake + manifest, không lọc nhãn.
    assert "--label" not in script
    assert "manifest.csv" in script and "corpus.zip" in script


def test_sync_script_is_written_before_the_generate_cell_uses_it(notebook):
    """Hook gọi sync_corpus.py, nên ô ghi file đó phải chạy trước.

    So theo chỉ số Ô, không theo vị trí chuỗi: ô A2b có nhắc tên cờ trong comment nên
    so bằng `.index()` trên toàn văn sẽ ra kết quả ngược mà vẫn trông có lý.
    """
    def first_cell(needle):
        return next(i for i, c in enumerate(notebook["cells"])
                    if c["cell_type"] == "code" and needle in "".join(c["source"]))

    assert first_cell("SYNC_SCRIPT.write_text") < first_cell('"--after-speaker"')


def test_dataset_is_checked_before_ingest(notebook):
    """Phải đọc dataset xem đã làm tới đâu TRƯỚC khi bắt đầu, không thì làm lại từ đầu."""
    def first_cell(needle):
        return next(i for i, c in enumerate(notebook["cells"])
                    if c["cell_type"] == "code" and needle in "".join(c["source"]))

    assert first_cell("vivos-fake-v2") < first_cell('run("ingest"')


def test_tts_is_off_but_still_one_switch_away(notebook_code):
    assert "TTS_ENGINES = []" in notebook_code
    assert "if TTS_ENGINES:" in notebook_code, "phải còn đường bật lại"


# ------------------------------------------------------------ MODE: A hay B hay cả hai
#
# Sinh fake bằng cloning mất nhiều giờ nên phần tạo dataset và phần huấn luyện thường
# không nằm cùng một phiên. `MODE` chọn phiên này làm gì, và mọi ô phải tôn trọng nó —
# Save & Run All là cách dùng thật, nên "bỏ qua bằng tay" không phải một lựa chọn.
def _mode_block(notebook_code: str) -> str:
    """Đoạn suy ra MAKE_DATASET/DO_TRAIN — lấy ra để CHẠY, chứ không chỉ khớp chuỗi."""
    start = notebook_code.index("if MODE not in")
    return notebook_code[start:notebook_code.index("\n", notebook_code.index("DO_TRAIN = ", start))]


@pytest.mark.parametrize("mode,phases", [
    ("dataset", (True, False)),
    ("train", (False, True)),
    ("both", (True, True)),
])
def test_mode_decides_which_phases_run(notebook_code, mode, phases):
    namespace = {"MODE": mode}
    exec(_mode_block(notebook_code), namespace)  # noqa: S102 — mã do chính repo sinh
    assert (namespace["MAKE_DATASET"], namespace["DO_TRAIN"]) == phases


def test_a_misspelled_mode_stops_the_notebook(notebook_code):
    """Gõ sai MODE mà chạy tiếp im lặng là bỏ cả phiên GPU — sai thì phải kêu ngay."""
    with pytest.raises(SystemExit):
        exec(_mode_block(notebook_code), {"MODE": "trian"})  # noqa: S102


def test_mode_is_set_before_anything_reads_it(notebook):
    """MODE và `skipped()` phải nằm ở ô cài thư viện: nó quyết định cài gói nào."""
    setup = _cells_with(notebook, 'MODE = "both"')
    assert len(setup) == 1, "MODE phải khai báo đúng một chỗ"
    assert "def skipped(" in "".join(notebook["cells"][setup[0]]["source"])

    readers = [i for i, c in enumerate(notebook["cells"])
               if c["cell_type"] == "code" and i != setup[0]
               and ("MAKE_DATASET" in "".join(c["source"])
                    or "DO_TRAIN" in "".join(c["source"]))]
    assert readers and min(readers) > setup[0]


@pytest.mark.parametrize("stage", ['run("ingest"', 'run("generate"', 'SYNC_READY ='])
def test_data_making_stages_only_run_when_making_a_dataset(notebook, stage):
    for i in _cells_with(notebook, stage):
        assert "MAKE_DATASET" in "".join(notebook["cells"][i]["source"]), \
            f"ô {i} chạy {stage} mà không hỏi MODE"


@pytest.mark.parametrize("stage", ['run("split")', 'run("augment"', 'run("features")',
                                   'run("train")', 'run("evaluate")', 'run("detect"',
                                   "make_archive"])
def test_training_stages_only_run_when_training(notebook, stage):
    for i in _cells_with(notebook, stage):
        assert "DO_TRAIN" in "".join(notebook["cells"][i]["source"]), \
            f"ô {i} chạy {stage} mà không hỏi MODE"


def test_corpus_restore_runs_in_every_mode(notebook):
    """Ô A1b là đường DUY NHẤT mang corpus vào phiên — gate nó thì MODE='train' chết."""
    src = _cell_src(notebook, 'run("unpack"')
    # Không cổng MODE nào được mở TRƯỚC lệnh bung — sau đó thì có (kiểm tra riêng cho
    # chế độ chỉ-huấn-luyện), nên so theo vị trí chứ không phải cả ô.
    before = src[:src.index('run("unpack"')]
    assert "MAKE_DATASET" not in before and "DO_TRAIN" not in before, before


def test_train_only_refuses_an_empty_or_one_sided_corpus(notebook):
    """Không corpus, hoặc corpus thiếu một lớp, thì phần B chỉ là mấy giờ GPU đổ đi."""
    src = _cell_src(notebook, 'run("unpack"')
    assert "if not MAKE_DATASET:" in src
    assert src.count("raise SystemExit") >= 2, src
    assert "chỉ có một lớp" in src


def test_train_only_installs_no_generation_engine(notebook):
    """Chỉ huấn luyện thì khỏi cài engine nào: đỡ vài phút và tránh xung đột transformers."""
    src = _cell_src(notebook, 'pip("-r", "requirements.txt")')
    gate = src.index("if not MAKE_DATASET:")
    assert gate < src.index('pip("omnivoice"')
    assert gate < src.index('pip("piper-tts"')


def test_real_dataset_detection_is_skipped_when_only_training(notebook):
    """MODE='train' mount corpus đã sinh, không mount VIVOS — đòi dataset real là dừng oan."""
    src = _cell_src(notebook, "detect_adapter(")
    assert src.index("if not MAKE_DATASET:") < src.index("Chưa add dataset nào")


# Khớp chuỗi chỉ chứng minh cái cổng CÓ MẶT. Ở đây chạy thật những ô stage với `run()`
# giả để xem MODE cho stage nào đi qua — đó mới là điều notebook hứa.
_STAGE_CELLS = ('run("ingest"', "*TTS_ENGINES", "xác nhận omnivoice đã ✔", '"--dry-run"',
                '"--after-speaker"', "k2-fsa/OmniVoice", 'run("split")',
                'run("features")', 'run("detect"')


def _stages_that_run(notebook, notebook_code, mode) -> set[str]:
    called: list[str] = []
    namespace = {
        "MODE": mode, "sys": sys, "SMOKE": False, "SYNC_READY": False,
        "TTS_ENGINES": ["piper"], "RAW": "/tmp/khong-dung", "N_REAL": 1,
        "PER_SPEAKER": 1, "N_FAKE_TTS": 1, "N_FAKE_CLONE": 1,
        "run": lambda *args, **kw: called.append(str(args[0])),
        "pip": lambda *args, **kw: None,
        "skipped": lambda what: None,
    }
    exec(_mode_block(notebook_code), namespace)  # noqa: S102 — dựng MAKE_DATASET/DO_TRAIN
    for needle in _STAGE_CELLS:
        exec(_cell_src(notebook, needle), namespace)  # noqa: S102
    return set(called)


@pytest.mark.parametrize("mode,phai_chay,khong_duoc_chay", [
    ("dataset", {"ingest", "generate"},
     {"split", "augment", "features", "train", "evaluate", "detect"}),
    ("train", {"split", "augment", "features", "train", "evaluate", "detect"},
     {"ingest", "generate"}),
])
def test_mode_gates_the_stages_that_actually_run(notebook, notebook_code, mode,
                                                phai_chay, khong_duoc_chay):
    called = _stages_that_run(notebook, notebook_code, mode)
    assert phai_chay <= called, f"MODE={mode} thiếu: {phai_chay - called}"
    assert not (khong_duoc_chay & called), f"MODE={mode} chạy thừa: {khong_duoc_chay & called}"


def test_both_still_runs_the_whole_pipeline(notebook, notebook_code):
    assert {"ingest", "generate", "split", "augment", "features", "train", "evaluate",
            "detect"} <= _stages_that_run(notebook, notebook_code, "both")
