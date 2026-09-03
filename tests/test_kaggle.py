"""Chạy trên Kaggle: nhận diện môi trường, config kế thừa, đóng gói corpus."""

from __future__ import annotations

import json
import os
import sys
import zipfile
from pathlib import Path

import pytest

from aidetector.config import Config
from aidetector.corpus.manifest import MANIFEST_NAME, Manifest, find_shards
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


def test_pack_writes_one_manifest_per_dataset(corpus, tmp_path):
    """Zip phải giữ nguyên cấu trúc tách theo bộ: bung ra là lại thành thư mục tự chứa."""
    archive = pack_corpus(corpus.root, tmp_path / "corpus.zip")
    with zipfile.ZipFile(archive) as zf:
        names = zf.namelist()
    bo = sorted(corpus.by_shard())
    assert [f"{ten}/{MANIFEST_NAME}" for ten in bo] == [n for n in names
                                                        if n.endswith(MANIFEST_NAME)]
    assert len(names) == len(corpus) + len(bo)
    assert all(n.split("/")[1] in (MANIFEST_NAME, "real", "fake", "augment")
               for n in names), names


def test_pack_can_skip_audio_for_a_metadata_only_archive(corpus, tmp_path):
    archive = pack_corpus(corpus.root, tmp_path / "meta.zip", include_audio=False)
    with zipfile.ZipFile(archive) as zf:
        assert zf.namelist() == [f"{ten}/{MANIFEST_NAME}" for ten in sorted(corpus.by_shard())]


def test_pack_round_trips_a_corpus_with_two_datasets(tmp_path, vivos_like):
    """Zip phải mang được nhiều bộ và bung ra đúng từng thư mục tự chứa của chúng."""
    import numpy as np

    from aidetector.corpus.schema import LABEL_REAL, Record, make_utt_id

    m = Manifest(tmp_path / "corpus")
    ingest_source(m, VivosAdapter(), vivos_like, "vivos", SPEC)
    for i in range(3):
        m.write_audio(
            Record(utt_id=make_utt_id("abc", "spk_abc", str(i)), path="", label=LABEL_REAL,
                   source="abc", speaker="spk_abc", text="câu của bộ abc"),
            np.zeros(4 * SPEC.sample_rate, np.float32), SPEC)
    m.save()

    archive = pack_corpus(m.root, tmp_path / "corpus.zip")
    lai = unpack_corpus(archive, tmp_path / "restored")

    assert len(lai) == len(m)
    assert {r.source for r in lai} == {"vivos", "abc"}
    for rec in m:
        assert lai.abs_path(lai.get(rec.utt_id)).exists()
    assert sorted(p.parent.name for p in find_shards(tmp_path / "restored")) == ["abc", "vivos"]


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
#: Hai file, hai việc: sinh dataset mất nhiều giờ GPU, huấn luyện chỉ cần corpus đã có.
#: Cả hai sinh ra từ CÙNG danh sách ô trong script build — xem `notebook` ở dưới.
NOTEBOOKS = {
    "dataset": Path("notebooks/aidetector_dataset.ipynb"),
    "train": Path("notebooks/aidetector_train.ipynb"),
}


@pytest.fixture(scope="module")
def notebook() -> dict:
    """TẤT CẢ ô, đúng thứ tự build — hai file là hai hình chiếu của danh sách này.

    Mọi test về NỘI DUNG pipeline soi ở đây: tách file không được làm mất ô nào, và
    thứ tự các stage là thứ tự trong danh sách này. Còn những gì thuộc từng FILE
    (nbformat, payload còn khớp repo, file có tự đủ không) thì soi qua `tep`.
    """
    builder = _load_builder()
    files = builder.collect_files()
    b64, sha, size = builder.build_payload(files)
    cells = builder.build_cells(builder.payload_literal(b64), sha, size, len(files))
    return {"cells": builder.cells_for(cells)}


@pytest.fixture(scope="module", params=sorted(NOTEBOOKS))
def tep(request) -> tuple[str, dict]:
    """(phần, notebook) của từng file thật trên đĩa."""
    return request.param, json.loads(NOTEBOOKS[request.param].read_text(encoding="utf-8"))


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


def test_there_are_exactly_two_notebooks():
    """Đúng hai file: một tạo dataset, một huấn luyện. Không có bản thứ ba lơ lửng."""
    assert (sorted(p.name for p in Path("notebooks").glob("*.ipynb"))
            == sorted(p.name for p in NOTEBOOKS.values()))


def test_notebook_is_valid_nbformat(tep):
    _, nb = tep
    assert nb["nbformat"] == 4
    assert nb["cells"], "notebook rỗng"
    for cell in nb["cells"]:
        assert cell["cell_type"] in ("markdown", "code")
        assert isinstance(cell["source"], list)


def test_every_code_cell_parses_as_python(tep):
    """Bỏ dòng magic `!...` của IPython rồi kiểm tra cú pháp Python thuần."""
    import ast

    _, nb = tep
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        pure = "\n".join(l for l in source.splitlines() if not l.lstrip().startswith("!"))
        ast.parse(pure)  # ném SyntaxError kèm số dòng nếu hỏng


def test_notebook_is_self_contained(tep):
    """MỖI file phải tự bung được mã nguồn: không clone repo, không dataset chứa code."""
    _, nb = tep
    text = "".join("".join(c["source"]) for c in nb["cells"])
    assert "_PAYLOAD" in text and "base64.b64decode" in text
    assert "git clone" not in text


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


def test_notebook_payload_matches_the_current_source(tep):
    """Payload nhúng phải khớp code trong repo; lệch ⇒ chạy lại script build."""
    builder = _load_builder()
    _, sha, _ = builder.build_payload(builder.collect_files())
    ten, nb = tep
    assert sha in "".join("".join(c["source"]) for c in nb["cells"]), (
        f"{NOTEBOOKS[ten].name} đã lệch với mã nguồn"
        " — chạy: python scripts/build_kaggle_notebook.py"
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


def test_engine_is_optional_only_when_something_else_makes_fakes(notebook):
    """`optional` phải phụ thuộc việc còn engine khác gánh lớp fake hay không.

    OmniVoice cần GPU và tải checkpoint vài GB, nên khi Piper/Kokoro còn bật thì bỏ qua
    lỗi của nó là đúng. Nhưng tắt TTS rồi thì nó là nguồn fake DUY NHẤT: bỏ qua lúc đó
    là đưa cả phần huấn luyện vào một corpus không có lớp fake nào.
    """
    call = _cell_src(notebook, "optional=bool(TTS_ENGINES)")
    assert "optional=bool(TTS_ENGINES)" in call, call


def test_generate_syncs_at_speaker_boundaries(notebook_code):
    """Mốc đẩy dữ liệu là ranh giới speaker, và chỉ gắn khi đồng bộ đã sẵn sàng."""
    assert '"--after-speaker"' in notebook_code
    assert "if SYNC_READY else []" in notebook_code
    assert "*SYNC_HOOK" in notebook_code, "cờ dựng ra mà không lệnh nào dùng"


def test_progress_is_checked_before_the_long_generation(notebook):
    """Bước sinh dài nhiều giờ — phải biết còn thiếu bao nhiêu TRƯỚC khi bắt đầu."""
    def cell(needle):
        return _cells_with(notebook, needle)[0]

    # So với ô sinh cloning — ô đó mới là ô dài nhiều giờ.
    assert cell('"--dry-run"') < cell("optional=bool(TTS_ENGINES)")


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


def _sync_script(notebook) -> str:
    """Dựng lại sync_corpus.py từ f-string lồng trong ô A2b."""
    import textwrap

    source = _cell_src(notebook, "SYNC_SCRIPT")
    start = source.index("textwrap.dedent(")
    end = source.index("'''))", start) + len("'''")
    return eval(source[start:end] + ")",  # noqa: S307 — chuỗi do chính repo sinh ra
                {"textwrap": textwrap, "DATASET_ID": "owner/slug",
                 "SYNC_EVERY_MINUTES": 20, "KEEP_OLD_VERSIONS": True})


def test_generated_sync_script_is_valid_python(notebook):
    """Script đồng bộ được sinh ra bằng f-string lồng trong ô notebook.

    Nó chỉ chạy trên Kaggle, giữa lượt sinh kéo dài nhiều giờ — hỏng cú pháp ở đó thì
    chỉ biết sau khi đã mất công. Sai một lớp `{{}}` là đủ vỡ, nên dựng ra và kiểm ở đây.
    """
    import ast

    script = _sync_script(notebook)
    ast.parse(script)
    assert "kaggle" in script and "--force" in script
    # Đẩy toàn bộ: real + fake + manifest, không lọc nhãn.
    assert "--label" not in script
    assert "manifest.csv" in script and "corpus.zip" in script


def test_sync_script_is_written_before_the_generate_cell_uses_it(notebook):
    """Hook gọi sync_corpus.py, nên ô ghi file đó phải chạy trước.

    So theo chỉ số Ô, không theo vị trí chuỗi: cùng một ô vừa ghi file vừa dựng cờ.
    """
    assert _cells_with(notebook, "SYNC_SCRIPT.write_text")[0] \
        <= _cells_with(notebook, '"--after-speaker"')[0] \
        < _cells_with(notebook, "*SYNC_HOOK")[0]


def test_dataset_is_checked_before_ingest(notebook):
    """Phải đọc dataset xem đã làm tới đâu TRƯỚC khi bắt đầu, không thì làm lại từ đầu."""
    def first_cell(needle):
        return next(i for i, c in enumerate(notebook["cells"])
                    if c["cell_type"] == "code" and needle in "".join(c["source"]))

    assert first_cell("vivos-fake-v2") < first_cell('run("ingest"')


def test_tts_is_off_but_still_one_switch_away(notebook_code):
    assert "TTS_ENGINES = []" in notebook_code
    assert "if TTS_ENGINES:" in notebook_code, "phải còn đường bật lại"


# ------------------------------------------------------------- MODE: file này làm phần nào
#
# Sinh fake bằng cloning mất nhiều giờ nên phần tạo dataset và phần huấn luyện không nằm
# cùng một phiên — chúng là hai file. `MODE` được script build chốt sẵn trong mỗi file, và
# mọi ô dùng chung phải tôn trọng nó: Save & Run All là cách dùng thật, nên "bỏ qua bằng
# tay" không phải một lựa chọn.
def _mode_block(notebook_code: str) -> str:
    """Đoạn suy ra MAKE_DATASET/DO_TRAIN — lấy ra để CHẠY, chứ không chỉ khớp chuỗi."""
    start = notebook_code.index("if MODE not in")
    return notebook_code[start:notebook_code.index("\n", notebook_code.index("DO_TRAIN = ", start))]


@pytest.mark.parametrize("mode,phases", [
    ("dataset", (True, False)),
    ("train", (False, True)),
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
    setup = _cells_with(notebook, "MODE = ")
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
    """Ô A1b là đường DUY NHẤT mang corpus vào phiên — bỏ qua nó thì MODE='train' chết."""
    src = _cell_src(notebook, 'run("unpack"')
    before = src[:src.index('run("unpack"')]
    assert "DO_TRAIN" not in before, before
    # MODE chỉ được quyết định NẠP TỪ ĐÂU, không được quyết định CÓ NẠP HAY KHÔNG: mọi
    # nhánh đều dẫn tới lệnh bung, và phiên train là phía RỘNG hơn (gộp mọi kho).
    assert "skipped(" not in before, before
    assert before.count("MAKE_DATASET") == 1, before


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
_STAGE_CELLS = ("INGEST_ADDED = ", "*TTS_ENGINES", "xác nhận omnivoice đã ✔", "_soluong = ",
                "optional=bool(TTS_ENGINES)", "k2-fsa/OmniVoice", 'run("split")',
                'run("features")', 'run("detect"')


def _stages_that_run(notebook, notebook_code, mode) -> set[str]:
    called: list[str] = []
    namespace = {
        "MODE": mode, "sys": sys, "SMOKE": False, "SYNC_READY": False,
        "TTS_ENGINES": ["piper"], "RAW": "/tmp/khong-dung", "N_REAL": 1,
        "PER_SPEAKER": 1, "N_FAKE_TTS": 1, "N_FAKE_CLONE": 1,
        "CORPUS": Path("/tmp/corpus-khong-ton-tai"), "SYNC_HOOK": [], "_nguon": [],
        # Ô A1c công bố: nguồn đã có trong kho chưa, và tên nó là gì.
        "_da_co": 0, "SOURCE": "nguon_test", "NGUON_DA_CO": {},
        # Helper do ô A1b định nghĩa; ở notebook nó là biến toàn cục, ở đây phải cấp.
        # Corpus tách theo bộ ⇒ nó trả về DANH SÁCH manifest, rỗng = chưa có corpus.
        "_cac_meta": lambda thu_muc: [],
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


def test_the_two_files_together_run_the_whole_pipeline(notebook, notebook_code):
    """Tách file không được làm rơi stage nào ra ngoài cả hai."""
    ca_hai = (_stages_that_run(notebook, notebook_code, "dataset")
              | _stages_that_run(notebook, notebook_code, "train"))
    assert {"ingest", "generate", "split", "augment", "features", "train", "evaluate",
            "detect"} <= ca_hai


# ----------------------------------------------- hai file: hình chiếu của một danh sách
#
# Hai file rời nhau thì nguy cơ số một là chúng lệch nhau: ô A1b ở file train nạp corpus
# theo một chuẩn, ô sinh ở file dataset ghi theo chuẩn khác, và chỗ lệch chỉ lộ ra sau
# nhiều giờ GPU. Nên hai file KHÔNG được sửa tay: chúng là hai hình chiếu của cùng một
# danh sách ô, và phần dùng chung phải giống nhau từng byte.
def _duoc_chieu(part: str) -> list[dict]:
    builder = _load_builder()
    files = builder.collect_files()
    b64, sha, size = builder.build_payload(files)
    cells = builder.build_cells(builder.payload_literal(b64), sha, size, len(files))
    import copy

    ra = copy.deepcopy(builder.cells_for(cells, part))
    kia = builder.TRAIN if part == builder.DATASET else builder.DATASET
    builder.dat_che_do(ra, part, builder.OUT[kia].name)
    return ra


def test_each_file_is_exactly_the_projection_of_the_shared_list(tep):
    """Sửa tay một file là mở đường cho hai file lệch nhau — build lại thì mất hết."""
    ten, nb = tep
    assert [c["source"] for c in nb["cells"]] == [c["source"] for c in _duoc_chieu(ten)], (
        f"{NOTEBOOKS[ten].name} lệch script build — chạy:"
        " python scripts/build_kaggle_notebook.py"
    )


def test_the_shared_cells_are_byte_identical_in_both_files():
    """Bung mã nguồn, `run()`, A1b nạp corpus: ba thứ cả hai file phải hiểu y như nhau."""
    nbs = {ten: json.loads(p.read_text(encoding="utf-8")) for ten, p in NOTEBOOKS.items()}
    src = {ten: ["".join(c["source"]) for c in nb["cells"]] for ten, nb in nbs.items()}
    chung = [s for s in src["dataset"] if s in src["train"]]
    for dau in ("_PAYLOAD = (", "def run(*args", "_cay_bung_san(", 'run("migrate")'):
        assert any(dau in s for s in chung), f"ô chứa {dau!r} phải dùng chung, không sao chép"
    # `MODE = ` là chỗ DUY NHẤT hai file được khác nhau trong phần dùng chung.
    assert 'MODE = "dataset"' in "".join(src["dataset"])
    assert 'MODE = "train"' in "".join(src["train"])


def test_each_file_defines_every_name_it_uses(tep):
    """Ô nào dùng tên gì thì tên đó phải được định nghĩa trong CHÍNH file đó, trước đó.

    Đây là rào duy nhất cho việc tách file: một helper nằm ở ô chỉ có trong file dataset
    mà file train lại gọi thì lỗi chỉ nổ ra trên Kaggle, sau khi đã tải xong mọi thứ.
    """
    import ast
    import builtins

    def dinh_nghia(tree):
        ra = set()
        for n in ast.walk(tree):
            if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)):
                ra.add(n.id)
            elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                ra.add(n.name)
            elif isinstance(n, ast.arg):
                ra.add(n.arg)
            elif isinstance(n, (ast.Import, ast.ImportFrom)):
                ra.update((a.asname or a.name).split(".")[0] for a in n.names)
            elif isinstance(n, ast.ExceptHandler) and n.name:
                ra.add(n.name)
            elif isinstance(n, (ast.Global, ast.Nonlocal)):
                ra.update(n.names)
        return ra

    ten, nb = tep
    co = set(dir(builtins)) | {"get_ipython", "display"}
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        src = "\n".join(l for l in "".join(cell["source"]).splitlines()
                         if not l.lstrip().startswith("!"))
        tree = ast.parse(src)
        co |= dinh_nghia(tree)          # tên định nghĩa trong chính ô này cũng hợp lệ
        thieu = {n.id for n in ast.walk(tree)
                 if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)} - co
        assert not thieu, f"{NOTEBOOKS[ten].name} ô {i} dùng tên chưa có: {sorted(thieu)}"


@pytest.mark.parametrize("part,khong_duoc_co", [
    ("dataset", ('run("split"', 'run("features"', 'run("train"', 'run("evaluate"')),
    ("train", ('run("ingest"', 'run("generate"', "SYNC_SCRIPT.write_text", "CONVERT")),
    ("dataset", ("STAGE_MODEL", "model-info.json")),
])
def test_neither_file_carries_the_other_half(part, khong_duoc_co):
    """Ô của phần kia còn nằm lại là mời người dùng chạy nó — hoặc chỉ để đọc rồi bỏ qua."""
    nb = json.loads(NOTEBOOKS[part].read_text(encoding="utf-8"))
    code = "".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    for dau in khong_duoc_co:
        assert dau not in code, f"{NOTEBOOKS[part].name} còn sót {dau!r}"



# ---------------------------------------- đồng bộ Kaggle Dataset: một nguồn, ba mốc
def test_dataset_id_is_declared_once_and_shared(notebook, notebook_code):
    """Đẩy lên một dataset rồi phiên sau nạp từ dataset khác là mất trắng mà không báo."""
    # Slug thật chỉ được xuất hiện MỘT lần; mọi chỗ khác đọc lại biến.
    assert notebook_code.count("sonpham12/vivos-fake-v2") == 1
    setup = _cells_with(notebook, "DATASET_ID = ")[0]
    users = _cells_with(notebook, "DATASET_ID")
    assert users[0] == setup and len(users) >= 3, users   # setup + A1b (nạp) + A2b (đẩy)
    # Script con nhận qua nội suy chứ không gõ lại slug.
    assert "DATASET_ID = {DATASET_ID!r}" in notebook_code


def test_restore_prefers_the_dataset_it_pushes_to(notebook):
    """Mount nhiều dataset thì "lấy cái cuối theo abc" là xổ số, không phải nối tiếp."""
    src = _cell_src(notebook, 'run("unpack"')
    assert 'DATASET_ID.split("/")[-1]' in src
    assert src.index('f"/kaggle/input/{slug}/**/{name}"') < src.index('f"/kaggle/input/**/{name}"')
    # Phiên tạo dataset dừng lại ở kho của chính nó: mọi thứ trong corpus lúc đó sẽ được
    # đẩy lên `DATASET_ID`, nên kéo bộ khác vào là bơm bộ lạ vào kho của bộ này.
    assert "if MAKE_DATASET:\n        return rieng" in src


def test_ingest_is_checkpointed_only_when_it_added_something(notebook):
    """Mốc sau ingest bịt lỗ "out lúc sinh giọng đầu"; nhưng không có gì mới thì đừng đẩy."""
    ingest = _cell_src(notebook, "INGEST_ADDED = ")
    assert 'run("ingest"' in ingest, "ô chốt mốc phải là ô ingest thật"

    sync = _cell_src(notebook, "SYNC_SCRIPT.write_text")
    assert "if SYNC_READY and INGEST_ADDED:" in sync
    assert sync.index("def sync_now(") < sync.index("if SYNC_READY and INGEST_ADDED:")


@pytest.mark.parametrize("cell", ["*TTS_ENGINES", "optional=bool(TTS_ENGINES)"])
def test_every_long_generate_checkpoints_at_speaker_boundaries(notebook, cell):
    """Lệnh sinh nào cũng phải chốt được tiến độ — không chỉ lệnh cloning."""
    assert "*SYNC_HOOK" in _cell_src(notebook, cell)


def test_sync_script_holds_its_cadence_even_when_a_push_fails(notebook):
    """Kaggle từ chối vì version trước còn xử lý là chuyện thường.

    Nếu chỉ chốt nhịp khi thành công thì mỗi ranh giới speaker lại gói và tải lại cả GB
    — hỏng liên tục thì đó là hammer, không phải retry.
    """
    script = _sync_script(notebook)
    # So với lệnh ĐẨY, không với lời gọi `kaggle` đầu tiên: trước đó còn một lượt đọc
    # manifest trên dataset để biết có được phép đẩy hay không.
    assert script.index("STAMP.touch()") < script.index('"datasets", "version"')
    assert "STAMP.write_text" not in script, "chốt nhịp phải xảy ra trước khi thử đẩy"


def test_the_last_push_of_a_session_ignores_the_rate_limit(notebook):
    """Nhịp chặn giữ cho phiên khỏi đứng chờ upload; bản chốt cuối thì phải đi."""
    sync = _cell_src(notebook, "SYNC_SCRIPT.write_text")
    assert '"--force"' in sync
    script = _sync_script(notebook)
    assert 'FORCE = "--force" in sys.argv' in script
    assert "if not FORCE and MIN_GAP" in script


def test_training_never_pushes_derived_audio_back(notebook):
    """`augment` ghi bản nhiễu/nén vào corpus. Đẩy lên là bơm dữ liệu phái sinh vào
    dataset, buộc mọi phiên sau tải thêm phần mà một lệnh augment sinh lại trong vài phút."""
    assert "SYNC_READY = MAKE_DATASET and kaggle_ready()" in _cell_src(notebook, "SYNC_READY = ")
    for i in _cells_with(notebook, 'run("augment"'):
        assert "sync" not in "".join(notebook["cells"][i]["source"]).lower()


# ------------------------------------- đẩy sau MỖI speaker: chạy nền, không chồng lượt
def test_the_default_cadence_is_every_speaker(notebook):
    """Mốc dày nhất mà corpus có: xong một giọng. Đẩy chặn dòng thì mới phải thưa ra."""
    assert "SYNC_EVERY_MINUTES = 0" in _cell_src(notebook, "SYNC_EVERY_MINUTES")


def test_the_push_runs_in_the_background(notebook):
    """Gói + upload là việc của CPU và mạng — để nó chen giữa hai speaker là trả bằng GPU.

    Cả ba thành phần đều bắt buộc: `nohup` để lượt đẩy sống qua lúc tiến trình generate
    kết thúc, `>>` + `2>&1` để cắt ống stdout (hook gọi bằng capture_output nên còn ống
    là còn đứng chờ, dù đã có `&`), và `&` để trả về ngay.
    """
    hook = next(l for l in _cell_src(notebook, "SYNC_HOOK = ").splitlines()
                if l.lstrip().startswith('f"nohup'))
    for part in ("nohup", ">>", "2>&1", "&"):
        assert part in hook, f"thiếu {part!r} trong: {hook}"


def test_two_pushes_never_overlap(notebook):
    """Hai lượt cùng lúc là cùng gói vào MỘT file zip mà lượt trước đang tải lên."""
    script = _sync_script(notebook)
    assert "LOCK.write_text(str(os.getpid()))" in script
    assert "while running():" in script
    assert script.index("while running():") < script.index("LOCK.write_text")
    # Khoá phải được nhả cả khi đẩy thất bại, bằng không cả phiên không đẩy được nữa.
    assert "finally:" in script and "LOCK.unlink(missing_ok=True)" in script


def test_a_forced_push_waits_for_the_background_one(notebook):
    """Lượt chốt cuối phải đợi lượt nền, không được bỏ qua — nó là bản chốt của cả phiên."""
    script = _sync_script(notebook)
    wait_block = script[script.index("while running():"):script.index("if not FORCE and MIN_GAP")]
    assert "if not FORCE:" in wait_block and "raise SystemExit(0)" in wait_block
    assert "time.sleep(" in wait_block


def test_the_background_log_is_readable_from_the_notebook(notebook):
    """Lượt đẩy nền không in vào ô nào; không có log thì nó là hộp đen."""
    setup = _cell_src(notebook, "SYNC_HOOK = ")
    assert "def sync_log(" in setup and "SYNC_LOG" in setup
    assert "sync_log()" in _cell_src(notebook, "sync_now()")


def test_old_versions_are_kept_unless_asked_otherwise(notebook):
    """Xoá version cũ là không lấy lại được, nên nó phải là lựa chọn, không phải mặc định.

    Mỗi lượt đẩy là ảnh chụp toàn bộ corpus nên bản mới nhất luôn là superset — bật cờ
    chỉ mất đường lùi, không mất dữ liệu. Nhưng đó vẫn là quyết định của người dùng.
    """
    assert "KEEP_OLD_VERSIONS = True" in _cell_src(notebook, "KEEP_OLD_VERSIONS")

    script = _sync_script(notebook)
    assert "if not KEEP_OLD:" in script
    assert '"--delete-old-versions"' in script
    # Cờ chỉ được dán vào `version`, không dán vào `create` (lần đầu chưa có gì để xoá).
    create = script[script.index('"datasets", "create"'):]
    assert "delete-old-versions" not in create


# ------------------------------------- chạy full: None = không áp trần, config tự cân
def test_full_run_passes_no_quota_flags(notebook):
    """`None` phải thành "không truyền cờ", không thành `--limit None`.

    Bỏ `--count` là để `fake_to_real_ratio: 1.0` trong config tự tính đúng một fake cho
    mỗi real — đó là định nghĩa "full" mà không phải gõ con số nào, và không lệch lớp.
    """
    a1 = _cell_src(notebook, "N_FAKE_CLONE")
    assert "= None, None, None, None" in a1, "lượt chạy thật phải mặc định không áp trần"

    ingest = _cell_src(notebook, "_tran = [")
    assert '["--limit", N_REAL] if N_REAL else []' in ingest
    assert '["--per-speaker", PER_SPEAKER] if PER_SPEAKER else []' in ingest

    gen = _cell_src(notebook, "optional=bool(TTS_ENGINES)")
    assert "*_soluong" in gen and '"--count"' not in gen
    assert '["--count", N_FAKE_CLONE] if N_FAKE_CLONE else []' in _cell_src(notebook, "_soluong =")


def test_the_notebook_reads_how_far_generation_got_before_starting(notebook):
    """Trước khi vào phiên nối tiếp, câu duy nhất đáng hỏi: còn bao nhiêu phải sinh."""
    a1b = _cell_src(notebook, 'run("unpack"')
    assert "is_usable" in a1b, "đích phải là real ĐỦ ĐIỀU KIỆN, không phải mọi real"
    assert "ref_utt_id for f in _m.fakes" in a1b
    assert "Tiến độ gen" in a1b
    # Đọc từ manifest, không nạp engine — phải chạy được trước cả khi cài omnivoice.
    assert "build_generator" not in a1b and "available_generators" not in a1b


def test_completion_is_confirmed_after_the_long_generation(notebook):
    """Hết phiên giữa đường là chuyện thường — phải biết còn nợ bao nhiêu."""
    # Chỉ tính ô đếm của bước SINH — ô A1c cũng dùng --dry-run nhưng để xem cấu trúc.
    dry = [i for i in _cells_with(notebook, '"--dry-run"')
           if "_soluong" in "".join(notebook["cells"][i]["source"])]
    gen = _cells_with(notebook, "optional=bool(TTS_ENGINES)")[0]
    assert len(dry) >= 2, "phải đếm cả trước và sau lượt sinh"
    assert dry[0] < gen < dry[1]


def test_sync_probes_the_tool_it_will_actually_use(notebook):
    """Cổng đồng bộ phải thử `kaggle` CLI, không suy diễn từ biến môi trường.

    Log phiên thật: `kaggle datasets files` chạy được ở A1b trong khi
    `UserSecretsClient` ném BackendError — cổng cũ kiểm Secrets nên tắt đồng bộ suốt 4
    giờ sinh, dù công cụ đẩy vốn xác thực được. Kiểm sai chỗ thì càng "an toàn" càng mất.
    """
    src = _cell_src(notebook, "def kaggle_ready(")
    assert "def kaggle_cli_ok(" in src
    assert '"datasets", "list", "-m"' in src, "phép thử phải cần xác thực mới qua được"
    # Trong thân kaggle_ready: thử CLI TRƯỚC, chỉ nạp secret khi nó không qua.
    body = src[src.index("def kaggle_ready("):]
    assert body.index("if kaggle_cli_ok():") < body.index("nap_credential()")


def test_both_kaggle_credential_styles_are_accepted(notebook):
    """Kaggle có token `KGAT_` (mới) và cặp username+key (legacy) — không thay nhau được.

    Đặt sai kiểu thì đồng bộ tắt im lặng, và đó là cách mất cả một phiên sinh.
    """
    src = _cell_src(notebook, "def kaggle_ready(")
    for ten in ("KAGGLE_API_TOKEN", "KAGGLE_USERNAME", "KAGGLE_KEY"):
        assert ten in src, f"không đọc secret {ten}"
    # Thiếu một kiểu không được làm hỏng kiểu kia.
    assert "except Exception:" in src and "pass" in src
    # Client cài sẵn có thể chưa biết biến KAGGLE_API_TOKEN — ghi ra file cho chắc.
    assert '".kaggle" / "access_token"' in src
    assert "chmod(0o600)" in src


def test_the_kaggle_client_is_upgraded_before_use(notebook):
    """Image Kaggle đang có kaggle 2.0.2 — bản đó có thể chưa biết token `KGAT_`."""
    assert 'pip("-U", "kaggle"' in _cell_src(notebook, 'pip("-r", "requirements.txt")')


def test_real_audio_is_quality_checked_before_any_fake_is_made(notebook):
    """Với engine cloning, mỗi real là KHUÔN sinh fake.

    Clip bị clipping hay gần im lặng thì fake dựng trên nó cũng là rác — phát hiện ở A4
    nghĩa là đã tốn hàng giờ GPU. Đọc lại ~8.000 file mất một phút.
    """
    kiem = _cells_with(notebook, 'run("validate"')
    sinh = _cells_with(notebook, "optional=bool(TTS_ENGINES)")[0]
    assert len(kiem) >= 2, "phải kiểm cả trước và sau lượt sinh"
    assert kiem[0] < sinh < kiem[1]
    # Trước khi sinh thì dọn luôn; sau khi sinh chỉ soi real+fake, không tự xoá.
    assert '"--fix"' in "".join(notebook["cells"][kiem[0]]["source"])
    assert '"--fix"' not in "".join(notebook["cells"][kiem[1]]["source"])


# ------------------------------------------ không được đè mất công của phiên trước
def test_a_session_that_cannot_load_the_corpus_stops(notebook):
    """`datasets version` là ảnh chụp TOÀN BỘ thư mục, không phải cộng dồn.

    Phiên quên Add Input sẽ ingest lại từ đầu rồi đẩy corpus 0 fake đè lên hàng nghìn
    fake của các phiên trước. Đi tiếp trong tình huống đó không bao giờ là điều đúng.
    """
    src = _cell_src(notebook, 'run("unpack"')
    assert "_co_du_lieu" in src
    assert "raise SystemExit(" in src
    # Phải chặn cả hai đường biết dataset có dữ liệu: manifest rời, và hỏi API.
    assert '"corpus.zip", "metadata.csv",' in src
    assert src.index("_co_du_lieu = (f\"{len(rows)} bản ghi") < src.index("if _co_du_lieu:")


# ------------------------------------------- Kaggle giải nén zip, ô nạp phải chịu được
def test_an_already_extracted_corpus_is_adopted_by_symlink(notebook):
    """Kaggle giải nén mọi .zip đưa lên dataset và không giữ lại bản nén.

    Nên `corpus.zip` thường KHÔNG có trong mount dù dataset vẫn đủ dữ liệu: nó nằm đó
    dưới dạng cây `real/ fake/`. Dừng vì "không thấy corpus.zip" trong tình huống đó là
    bỏ đi cả một corpus lành lặn — và trên phiên `train` thì là bỏ cả phiên.
    """
    src = _cell_src(notebook, 'run("unpack"')
    assert "_cay_bung_san(" in src and "_muon_cay(" in src
    # Thứ tự bắt buộc: zip trước (rõ ràng hơn), cây bung sẵn sau, DỪNG là đường cuối.
    assert (src.index('run("unpack"')
            < src.index("_xong, _thieu = _muon_cay(")
            < src.index("_co_du_lieu = ("))
    # Phân biệt gốc corpus thật với bản metadata.csv để rời: audio có mặt cạnh nó không.
    assert '(goc / r["path"]).exists()' in src
    # /kaggle/input chỉ-đọc: audio phải là symlink, manifest phải là bản copy ghi được,
    # vì split ghi cột `split` và validate ghi `checked` vào chính file đó.
    assert "os.symlink(nguon, dich)" in src
    # Corpus tách theo bộ ⇒ copy MỌI manifest, mỗi cái về đúng vị trí tương đối của nó.
    assert "for meta in _cac_meta(goc):" in src
    assert "dich_meta = CORPUS / meta.relative_to(goc)" in src
    assert "shutil.copy(meta, dich_meta)" in src
    # Manifest luôn mới hơn ảnh chụp một nhịp — bản ghi chưa có file phải bị loại.
    assert "prune_missing()" in src


# ------------------------------------------- một Kaggle Dataset = một bộ, gộp là add Input
def _kho_mot_bo(mount, ten_bo, n=3):
    """Dựng một Kaggle Dataset đã giải nén: đúng MỘT thư mục bộ, có cả real lẫn fake."""
    import numpy as np

    from aidetector.corpus.schema import LABEL_FAKE, LABEL_REAL, Record, make_utt_id

    cau = "hôm nay trời rất đẹp và mát mẻ nên cả nhà cùng nhau đi dạo phố"
    m = Manifest(mount)
    for i in range(n):
        for nhan, gen in ((LABEL_REAL, ""), (LABEL_FAKE, "dummy_tts:voice_a")):
            m.write_audio(
                Record(utt_id=make_utt_id(ten_bo, f"spk-{ten_bo}", f"{nhan}{i}"), path="",
                       label=nhan, source=ten_bo, speaker=f"spk-{ten_bo}", text=cau,
                       generator=gen),
                np.zeros(4 * SPEC.sample_rate, np.float32), SPEC)
    m.save()
    return m


def _run_a1b(notebook, tmp_path, mounts, make_dataset, dataset_id="ai/kho-vivos"):
    """Chạy ô A1b thật, với /kaggle/input và /kaggle/working trỏ vào tmp_path."""
    work = tmp_path / "working"
    work.mkdir(parents=True, exist_ok=True)
    cell = _cell_src(notebook, 'run("unpack"')
    cell = cell.replace("/kaggle/working", str(work)).replace("/kaggle/input", str(mounts))
    ns = {"DATASET_ID": dataset_id, "MAKE_DATASET": make_dataset,
          "CFG": "configs/kaggle.yaml", "run": lambda *a: None}
    exec(compile(cell, "a1b", "exec"), ns)
    return ns


def test_training_merges_every_mounted_store(notebook, tmp_path):
    """Một dataset một bộ ⇒ huấn luyện trên nhiều bộ là add nhiều Input, không phải sửa gì.

    Lấy đúng MỘT kho như trước là im lặng bỏ nửa dữ liệu: phiên vẫn chạy, số vẫn đẹp,
    và không dòng log nào nói rằng bộ thứ hai chưa bao giờ được nhìn tới.
    """
    mounts = tmp_path / "input"
    _kho_mot_bo(mounts / "kho-vivos", "vivos")
    _kho_mot_bo(mounts / "kho-abc", "abc")

    ns = _run_a1b(notebook, tmp_path, mounts, make_dataset=False)

    m = Manifest.load(ns["CORPUS"], required=True)
    assert {r.source for r in m} == {"vivos", "abc"}
    assert sorted(p.parent.name for p in find_shards(ns["CORPUS"])) == ["abc", "vivos"]
    assert ns["NGUON_DA_CO"] == {"vivos": 3, "abc": 3}
    # Mount chỉ-đọc: audio là symlink, manifest là bản copy ghi được.
    assert all(m.abs_path(r).exists() for r in m)
    assert all(m.abs_path(r).is_symlink() for r in m)
    assert not (ns["CORPUS"] / "vivos" / MANIFEST_NAME).is_symlink()


def _kho_cau_truc_cu(mount, ten_bo="vivos"):
    """Kho đẩy lên TRƯỚC khi corpus tách theo bộ: manifest gộp ở gốc, cây `real/<bộ>/…`."""
    import numpy as np

    from aidetector.corpus.manifest import manifest_csv
    from aidetector.corpus.schema import LABEL_REAL, Record, make_utt_id
    from aidetector.corpus.spec import save_audio

    recs = []
    for speaker in (f"{ten_bo}spk01", f"{ten_bo}spk02"):
        for i in range(2):
            duong = f"real/{ten_bo}/{speaker}/{i + 1:04d}.wav"
            (mount / duong).parent.mkdir(parents=True, exist_ok=True)
            save_audio(mount / duong, np.zeros(4 * SPEC.sample_rate, np.float32), SPEC)
            recs.append(Record(utt_id=make_utt_id(ten_bo, speaker, f"cu-{i}"), path=duong,
                               label=LABEL_REAL, source=ten_bo, speaker=speaker,
                               text="câu cũ", duration=4.0))
    (mount / MANIFEST_NAME).write_text(manifest_csv(recs), newline="", encoding="utf-8")
    return recs


def test_a_legacy_store_migrates_to_one_folder_per_set(notebook, tmp_path):
    """Đưa kho cũ về cấu trúc mới: nạp (symlink) → `migrate` → gói lại là đúng cây mới.

    Đây là đường duy nhất đổi được cấu trúc một Kaggle Dataset — không đổi tên file tại
    chỗ được, phải đẩy một version mới. Hai chỗ dễ vỡ mà chỉ ca này chạm tới: `migrate`
    dời SYMLINK trỏ vào mount chỉ-đọc, và `pack` phải đi theo symlink để zip có nội dung
    thật chứ không phải một liên kết gãy.
    """
    mounts = tmp_path / "input"
    cu = _kho_cau_truc_cu(mounts / "kho-vivos")

    ns = _run_a1b(notebook, tmp_path, mounts, make_dataset=True)

    corpus = ns["CORPUS"]
    m = Manifest.load(corpus, required=True)
    assert len(m) == len(cu)
    assert all(m.abs_path(r).is_symlink() for r in m), "mount chỉ-đọc ⇒ phải là symlink"

    m.migrate_layout()
    m.save()

    assert [p.parent.name for p in find_shards(corpus)] == ["vivos"]
    assert not (corpus / MANIFEST_NAME).exists(), "bảng gộp cũ phải được tách đi"
    for rec in m:
        assert rec.path.startswith("vivos/real/")
        assert m.abs_path(rec).exists()

    archive = pack_corpus(corpus, tmp_path / "corpus.zip")
    with zipfile.ZipFile(archive) as zf:
        names = zf.namelist()
        assert all(zf.read(n) for n in names), "zip phải mang nội dung thật, không phải symlink gãy"
    assert f"vivos/{MANIFEST_NAME}" in names
    assert all(n.startswith("vivos/") for n in names), names


def test_a_dataset_session_loads_only_its_own_store(notebook, tmp_path):
    """Phiên sinh đẩy nguyên corpus lên `DATASET_ID` — bộ lạ trong đó là bơm nhầm kho."""
    mounts = tmp_path / "input"
    _kho_mot_bo(mounts / "kho-vivos", "vivos")
    _kho_mot_bo(mounts / "kho-abc", "abc")

    ns = _run_a1b(notebook, tmp_path, mounts, make_dataset=True, dataset_id="ai/kho-vivos")
    assert ns["NGUON_DA_CO"] == {"vivos": 3}


def test_a_source_that_lives_in_two_inputs_stops_the_session(notebook, tmp_path):
    """Hai bản của cùng một bộ đánh số `0001.wav` độc lập nhau.

    `unpack` và symlink đều bỏ qua đường dẫn đã tồn tại, nên gộp lại là bản ghi của kho
    sau trỏ vào audio của kho trước — sai nội dung mà không phép kiểm nào ở dưới bắt được.
    """
    mounts = tmp_path / "input"
    _kho_mot_bo(mounts / "kho-a", "vivos")
    _kho_mot_bo(mounts / "kho-b", "vivos")

    with pytest.raises(SystemExit) as err:
        _run_a1b(notebook, tmp_path, mounts, make_dataset=False)
    assert "vivos" in str(err.value) and "HAI Input" in str(err.value)
    assert "kho-a" in str(err.value) and "kho-b" in str(err.value)


def test_the_real_picker_skips_our_own_corpus(notebook, tmp_path):
    """Cây corpus được chấm 0.95 — trên cả Common Voice (0.9) và VIVOS thiếu split (0.7).

    Không loại nó ra thì phiên đi ingest lại chính corpus của mình thành một nguồn mới.
    """
    _kho_mot_bo(tmp_path / "aaa-kho-vivos", "vivos")
    that = tmp_path / "zzz-vivos"
    (that / "train" / "waves" / "SPK1").mkdir(parents=True)
    (that / "train" / "prompts.txt").write_text("SPK1_R001 xin chào", encoding="utf-8")

    ns = _run_picker(notebook, tmp_path, {"MAKE_DATASET": True, "DATASET_ID": "ai/kho-vivos"})
    assert ns["RAW"] == str(that)


def test_the_push_refuses_a_store_with_more_than_one_source(notebook, tmp_path):
    """Kho của bộ này mà chứa bộ khác thì phiên train mount về sẽ thấy hai bộ một Input.

    Chạy thật script đã dựng: rào phải nằm TRƯỚC mọi lệnh `kaggle`, nên nó dừng được cả
    khi máy không có CLI lẫn token — đúng tình huống của một sandbox.
    """
    import subprocess

    script = _sync_script(notebook)
    assert script.index("len(theo_bo) > 1") < script.index('"datasets", "version"')

    _kho_mot_bo(tmp_path / "corpus", "vivos")
    _kho_mot_bo(tmp_path / "corpus", "abc")
    tep = tmp_path / "sync.py"
    tep.write_text(script.replace("/kaggle/working", str(tmp_path)), encoding="utf-8")

    ket = subprocess.run([sys.executable, str(tep)], capture_output=True, text=True)
    assert ket.returncode == 4, ket.stdout + ket.stderr
    assert "TỪ CHỐI ĐẨY" in ket.stdout and "abc, vivos" in ket.stdout


def test_the_push_tolerates_the_legacy_merged_manifest(notebook, tmp_path):
    """Corpus bung từ version cũ còn bảng gộp ở gốc BÊN CẠNH shard mới — vẫn là một bộ.

    Đếm số file manifest thay vì số thư mục bộ là từ chối đẩy đúng những phiên đang nối
    tiếp corpus cũ nhất.
    """
    import subprocess

    _kho_mot_bo(tmp_path / "corpus", "vivos")
    (tmp_path / "corpus" / MANIFEST_NAME).write_text(
        (tmp_path / "corpus" / "vivos" / MANIFEST_NAME).read_text(encoding="utf-8"),
        encoding="utf-8")
    tep = tmp_path / "sync.py"
    tep.write_text(_sync_script(notebook).replace("/kaggle/working", str(tmp_path)),
                   encoding="utf-8")

    ket = subprocess.run([sys.executable, str(tep)], capture_output=True, text=True)
    assert "TỪ CHỐI ĐẨY" not in ket.stdout, ket.stdout


# --------------------------------------- mô hình đi kho RIÊNG, không vào kho corpus
def test_the_model_goes_to_its_own_store(notebook, notebook_code):
    """Mỗi lượt đẩy là ảnh chụp toàn bộ staging: mô hình nằm trong kho corpus sẽ bị lượt
    đẩy corpus kế tiếp xoá khỏi version mới nhất."""
    assert notebook_code.count("sonpham12/aidetector-model") == 1
    setup = _cells_with(notebook, "MODEL_STORE_ID = ")[0]
    assert setup == _cells_with(notebook, "DATASET_ID = ")[0], "hai kho khai báo cạnh nhau"

    src = _cell_src(notebook, "STAGE_MODEL")
    # Rào bằng MÃ chứ không bằng comment — hai biến nằm cạnh nhau nên rất dễ copy nhầm.
    assert "if MODEL_STORE_ID == DATASET_ID:" in src
    assert src.index("MODEL_STORE_ID == DATASET_ID") < src.index('"datasets", "version"')
    # Và lệnh đẩy chỉ được nhắm vào kho mô hình. `DATASET_ID` vẫn được nhắc tới, nhưng
    # chỉ ở hai chỗ vô hại: rào ở trên và dòng ghi lại "học từ kho nào" trong model-info.
    assert '"-p", str(STAGE_MODEL)' in src
    assert f'"id": MODEL_STORE_ID' in src
    assert src.count("DATASET_ID") == 3, src   # rào (2 lần trong 1 dòng) + model-info
    assert '"store": DATASET_ID' in src


def test_the_model_push_is_train_only_and_never_blocks_the_session(notebook):
    """Không có token thì in nhắc rồi đi tiếp: `model.zip` ở ô B4 vẫn là đường lùi."""
    src = _cell_src(notebook, "STAGE_MODEL")
    assert "if not DO_TRAIN:" in src and "skipped(" in src
    assert "elif not kaggle_ready():" in src
    # Ô B4 (gói zip cho Output) phải chạy TRƯỚC — nó không cần mạng, không cần token.
    assert _cells_with(notebook, "make_archive")[0] < _cells_with(notebook, "STAGE_MODEL")[0]


def test_credentials_are_shared_by_both_push_paths(tep):
    """Hai file cùng đẩy, chỉ khác đẩy CÁI GÌ — đường xác thực thì phải đúng một."""
    ten, nb = tep
    code = "".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    assert code.count("def kaggle_ready(") == 1, f"{ten}: xác thực phải nằm ở đúng một ô"


def _fake_kaggle(tmp_path):
    """`kaggle` giả trên PATH: ghi lại argv rồi báo thành công."""
    bin_dir, log = tmp_path / "bin", tmp_path / "kaggle_argv.txt"
    bin_dir.mkdir()
    (bin_dir / "kaggle").write_text(f'#!/bin/sh\necho "$@" >> {log}\nexit 0\n', encoding="utf-8")
    (bin_dir / "kaggle").chmod(0o755)
    return bin_dir, log


def test_the_model_push_stages_checkpoint_reports_and_a_summary(notebook, tmp_path, monkeypatch):
    """Chạy thật ô B5 với `kaggle` giả: staging phải đủ ba thứ và lệnh phải nhắm đúng kho."""
    import numpy as np

    from aidetector.models import build_head, save_checkpoint

    work = tmp_path / "working"
    (work / "checkpoints").mkdir(parents=True)
    (work / "reports").mkdir(parents=True)
    save_checkpoint(work / "checkpoints" / "best.pt", build_head({"head": "linear"}, 4), {
        "model": {"head": "linear"}, "input_dim": 4,
        "norm_mean": [[0.0] * 4], "norm_std": [[1.0] * 4],
        "backbone": {"name": "wavlm", "checkpoint": "microsoft/wavlm-base-plus",
                     "output_layer": 6, "pooling": "mean"},
        "audio": {}, "epoch": 7, "val_eer": 0.04, "val_loss": 0.2, "threshold": 0.61,
    })
    (work / "reports" / "metrics.json").write_text(json.dumps(
        {"overall": {"eer": 0.05, "roc_auc": 0.98, "min_dcf": 0.1, "n_samples": 200,
                     "accuracy": 0.95, "threshold": 0.61}}), encoding="utf-8")
    (work / "reports" / "curves.png").write_bytes(b"PNG")

    bin_dir, log = _fake_kaggle(tmp_path)
    monkeypatch.setenv("PATH", f"{bin_dir}:{os.environ['PATH']}")

    cell = _cell_src(notebook, "STAGE_MODEL").replace("/kaggle/working", str(work))
    ns = {"DO_TRAIN": True, "MODEL_STORE_ID": "ai/kho-mo-hinh", "DATASET_ID": "ai/kho-vivos",
          "NGUON_DA_CO": {"vivos": 7367}, "kaggle_ready": lambda: True,
          "skipped": lambda *a: None}
    exec(compile(cell, "b5", "exec"), ns)

    stage = ns["STAGE_MODEL"]
    assert (stage / "checkpoints" / "best.pt").exists()
    assert (stage / "reports" / "metrics.json").exists()
    assert (stage / "reports" / "curves.png").exists()
    assert json.loads((stage / "dataset-metadata.json").read_text())["id"] == "ai/kho-mo-hinh"

    info = json.loads((stage / "model-info.json").read_text())
    assert info["eer"] == 0.05 and info["threshold"] == 0.61
    assert info["backbone"]["checkpoint"] == "microsoft/wavlm-base-plus"
    assert info["corpus"] == {"sources": {"vivos": 7367}, "store": "ai/kho-vivos"}

    lenh = log.read_text()
    assert "datasets version -p" in lenh and str(stage) in lenh
    assert "EER 5.00%" in lenh and "vivos" in lenh
    assert "ai/kho-vivos" not in lenh, "không được đẩy vào kho corpus"


def test_the_model_push_refuses_to_target_the_corpus_store(notebook, tmp_path):
    src = _cell_src(notebook, "STAGE_MODEL").replace("/kaggle/working", str(tmp_path))
    with pytest.raises(SystemExit) as err:
        exec(compile(src, "b5", "exec"),
             {"DO_TRAIN": True, "MODEL_STORE_ID": "ai/x", "DATASET_ID": "ai/x"})
    assert "MODEL_STORE_ID" in str(err.value)


def test_the_source_and_the_store_must_name_the_same_set(notebook):
    """SOURCE nói bộ này, DATASET_ID nói kho của bộ kia — đi tiếp là đẩy nhầm chỗ."""
    src = _cell_src(notebook, "CONVERT = None")
    assert "_bo_la = sorted(set(NGUON_DA_CO) - {SOURCE})" in src
    assert "if MAKE_DATASET and _bo_la:" in src
    assert src.index("_bo_la") < src.index("convert_and_verify(")


def test_the_push_refuses_to_shrink_the_dataset(notebook):
    """Rào thứ hai, độc lập với ô A1b: đếm trước khi đẩy."""
    script = _sync_script(notebook)
    assert "def dem_tren_dataset(" in script
    assert '"datasets", "download"' in script and '"manifest.csv"' in script
    assert "local < remote" in script
    assert script.index("local < remote") < script.index('"datasets", "version"')
    # Chặn được thì phải mở được: người dùng có thể thật sự muốn thu nhỏ.
    assert "--allow-shrink" in script
    # Đọc không được thì KHÔNG chặn — trục trặc mạng không được làm đứng lượt sinh.
    assert "remote is not None and local < remote" in script


def test_progress_state_is_uploaded_with_every_version(notebook):
    """Trạng thái phải nằm TRONG chính version đó, không suy ra từ nơi khác.

    progress.json vài KB nên đọc được trên trang dataset và tải riêng được — manifest
    của corpus đầy đủ là vài MB, còn corpus.zip là vài GB.
    """
    script = _sync_script(notebook)
    assert '"progress"' in script and '"progress.json"' in script
    # Sinh TRƯỚC khi đẩy, để nó luôn khớp với corpus.zip cùng version.
    assert script.index('"progress.json"') < script.index('"datasets", "version"')
    # progress.json thử trước; metadata/manifest chỉ là đường lùi cho version cũ.
    assert script.index('tai_ve("progress.json")') < script.index("f = tai_ve(ten)")


def test_the_resume_cell_reads_the_recorded_speaker_progress(notebook):
    src = _cell_src(notebook, 'run("unpack"')
    assert '_find("progress.json")' in src
    assert "speakers_done" in src and "speakers_todo" in src


def test_the_duration_window_is_one_knob_applied_to_every_stage(notebook, notebook_code):
    """Cửa sổ độ dài phải giống nhau ở mọi stage.

    Hạ ngưỡng cho `ingest` mà không hạ cho `generate` là real giữ tới 2s trong khi fake
    dưới 3s bị bỏ — chính độ dài thành dấu hiệu phân biệt hai lớp.
    """
    for ten in ("MIN_SECONDS = ", "MAX_SECONDS = "):
        assert notebook_code.count(ten) == 1, f"{ten} khai báo trùng ⇒ hai chuẩn"

    chay = _cells_with(notebook, "def run(")[0]
    dat = _cells_with(notebook, "MIN_SECONDS = ")[0]
    src = _cell_src(notebook, "def run(")
    # `run` đọc biến lúc GỌI nên khai báo sau cũng được; không được đọc lúc định nghĩa.
    for ten in ("MIN_SECONDS", "MAX_SECONDS"):
        assert f'globals().get("{ten}")' in src
    assert 'f"audio.{_k}={_v}"' in src
    assert chay < dat, "run phải sẵn sàng trước khi ô đặt hằng số chạy"

    # Không stage nào được tự đặt ngưỡng riêng — chỉ `run` dán vào, một lần, cho tất cả.
    rieng = [i for i, c in enumerate(notebook["cells"])
             if c["cell_type"] == "code" and i != chay
             and '"--set", "audio.' in "".join(c["source"])]
    assert not rieng, f"ô {rieng} tự truyền ngưỡng độ dài"


def test_the_upload_folder_is_wiped_before_each_push(notebook):
    """`datasets version` đẩy MỌI file trong thư mục staging.

    Một file sót lại từ lượt trước (vd manifest.csv tên cũ) sẽ lên dataset kèm theo và
    nằm đó mãi.
    """
    script = _sync_script(notebook)
    assert "shutil.rmtree(STAGE" in script
    assert script.index("shutil.rmtree(STAGE") < script.index('"pack"')


def test_sync_prefers_the_current_metadata_name(notebook):
    """Có cả hai tên thì phải lấy bản MỚI, ở cả hai chiều nạp và đẩy."""
    a1b = _cell_src(notebook, 'run("unpack"')
    assert '_find("metadata.csv") or _find("manifest.csv")' in a1b, "nối danh sách là lấy nhầm bản cũ"

    script = _sync_script(notebook)
    assert script.index('"metadata.csv"') < script.index('"manifest.csv"')
    # Manifest của từng bộ được copy ra STAGE, giữ đúng vị trí tương đối trong corpus.
    assert "dich = STAGE / f.relative_to(CORPUS)" in script


def test_an_empty_corpus_is_skipped_not_crashed(notebook):
    """Đẩy khi chưa có corpus phải là bỏ lượt, không phải StopIteration trần trụi."""
    script = _sync_script(notebook)
    assert "if not meta_local:" in script
    assert script.index("if not meta_local:") < script.index('"pack"')


def test_the_resumed_corpus_is_migrated_to_the_current_layout(notebook):
    """Corpus phiên trước có thể còn cây cũ; `migrate` giữ nguyên utt_id nên không sinh lại gì."""
    # `migrate` là một bước RIÊNG (ô A1d), không chôn trong ô nạp corpus — nó là tầng
    # convert cấu trúc, phải nhìn thấy được và chạy lại được một mình.
    assert _cells_with(notebook, 'run("unpack"')[0] < _cells_with(notebook, 'run("migrate")')[0]
    assert _cells_with(notebook, 'run("migrate")')[0] < _cells_with(notebook, "INGEST_ADDED = ")[0]


# ---------------------------------- convert (dataset lạ) và migrate (corpus cũ) tách đôi
def test_structure_conversion_is_verified_before_paying_for_it(notebook):
    """Adapter đọc sai cấu trúc là hỏng mọi thứ phía sau — và ingest 12.000 file mới biết
    là trả giá vô ích. Phép kiểm chỉ duyệt tên đường dẫn, không giải mã file nào."""
    o_convert = _cells_with(notebook, "convert_and_verify(")[0]
    assert o_convert < _cells_with(notebook, "INGEST_ADDED = ")[0], "phải kiểm trước khi nạp thật"


def test_the_convert_cell_is_the_only_place_that_knows_the_input_layout(notebook):
    """Mỗi bộ dữ liệu có cấu trúc riêng, và ô convert là chỗ DUY NHẤT được biết điều đó.

    Ba mức: adapter tự dò · chỉ định adapter · dev viết hàm CONVERT ngay trong ô. Mọi
    bước sau làm việc trên cây chuẩn và không cần biết dữ liệu vốn nằm thế nào.
    """
    src = _cell_src(notebook, "CONVERT = None")
    for phai_co in ("SOURCE  = ", "def CONVERT(raw, out)", '_nguon = ["--name", SOURCE]'):
        assert phai_co in src, f"ô convert thiếu {phai_co!r}"
    # Convert chỉ dựng lại CẤU TRÚC; chuẩn hoá audio là việc của ingest, làm hai lần là
    # bào mòn tín hiệu (trim ăn dần, clip sát ngưỡng rơi khỏi cửa sổ độ dài).
    assert "Không resample, không chuẩn mức, không cắt độ dài" in src

    # Và nó phải chạy TRƯỚC ingest, đồng thời quyết định tên nguồn ingest dùng.
    convert = _cells_with(notebook, "CONVERT = None")[0]
    ingest = _cells_with(notebook, "INGEST_ADDED = ")[0]
    assert convert < ingest
    assert "*_nguon" in "".join(notebook["cells"][ingest]["source"])


def test_migrate_is_its_own_step(notebook):
    """Tầng convert cấu trúc phải nhìn thấy được và chạy lại được một mình."""
    o_migrate = _cells_with(notebook, 'run("migrate")')
    assert len(o_migrate) == 1
    src = "".join(notebook["cells"][o_migrate[0]]["source"]).strip()
    assert src == 'run("migrate")', f"ô migrate phải chỉ làm một việc: {src!r}"


def test_convert_is_skipped_when_the_store_already_has_the_source(notebook):
    """Convert là bước tốn kém nhất (chép hàng nghìn file) và kết quả không đổi.

    Kho đã có nguồn đó thì cả convert lẫn ingest đều phải bỏ qua — hỏi kho trước, làm sau.
    """
    a1b = _cell_src(notebook, 'run("unpack"')
    assert "NGUON_DA_CO = {}" in a1b, "A1b phải công bố nguồn nào đã có trong kho"
    assert "NGUON_DA_CO[_r.source]" in a1b

    convert = _cell_src(notebook, "CONVERT = None")
    assert "_da_co = NGUON_DA_CO.get(SOURCE, 0)" in convert
    # Kho được hỏi trước, và câu trả lời đi thẳng vào hàm gộp qua `already=`.
    assert "already=_da_co" in convert
    assert convert.index("_da_co = ") < convert.index("convert_and_verify(")

    ingest = _cell_src(notebook, "INGEST_ADDED = ")
    assert "elif _da_co:" in ingest, "ingest cũng phải bỏ qua khi kho đã có nguồn"


def test_convert_and_verify_are_one_call(notebook):
    """Ba việc đi liền nhau nên nằm trong một hàm ở repo, không rải ra ô notebook.

    Tách ra thì rất dễ có đường đi bỏ qua phép kiểm — mà đường bị bỏ qua đúng là đường
    hay hỏng nhất: adapter sẵn có đọc sai tầng thư mục speaker của một bộ dữ liệu lạ.
    """
    src = _cell_src(notebook, "CONVERT = None")
    assert "from aidetector.ingest import convert_and_verify" in src
    assert src.count("convert_and_verify(") == 1, "phải đúng MỘT lời gọi"
    # Ô chỉ còn khai báo và in kết quả; không có nhánh nào tự đi vòng qua hàm đó.
    assert "if CONVERT is not None" not in src
    assert "RAW = _kq[\"root\"]" in src


def test_a_bad_input_stops_the_session(notebook):
    """`run()` dừng phiên khi lệnh trả mã khác 0 — verify phải dùng chính đường đó."""
    from aidetector.ingest import _preview
    from aidetector.ingest.base import SourceItem

    class Gia:
        name = "gia"

    def mot_speaker():
        for i in range(4):
            yield SourceItem(key=f"k{i}", audio_path=Path(f"/x/{i}.wav"),
                             speaker="mot_giong", text="xin chào các bạn hôm nay")

    r = _preview(Gia(), mot_speaker(), "nguon_la", 4)
    assert r["ok"] is False
    assert any("speaker" in s for s in r["problems"])

    def khong_text():
        for i in range(9):
            yield SourceItem(key=f"k{i}", audio_path=Path(f"/x/{i}.wav"),
                             speaker=f"spk{i}", text="")

    r = _preview(Gia(), khong_text(), "nguon_la", 9)
    assert r["ok"] is False and any("transcript" in s for s in r["problems"])
