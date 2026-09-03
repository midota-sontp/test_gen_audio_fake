"""CLI: parser + thứ tự stage."""

from __future__ import annotations

import pytest

from aidetector.cli import STAGES, build_parser, main


def test_run_without_stages_uses_full_default_order():
    """`run` không kèm stage nào phải parse được (argparse + choices từng làm lỗi)."""
    args = build_parser().parse_args(["run"])
    assert args.stages == []
    assert list(STAGES) == ["ingest", "generate", "split", "augment",
                            "features", "train", "evaluate"]


def test_split_runs_before_augment():
    """Bản augment chỉ được sinh cho train ⇒ split buộc phải xong trước."""
    assert STAGES.index("split") < STAGES.index("augment")
    assert STAGES.index("augment") < STAGES.index("features")


def test_run_rejects_unknown_stage(tmp_path):
    assert main(["run", "khong_ton_tai", "--corpus", str(tmp_path)]) == 2


def test_info_lists_registries(capsys):
    assert main(["info"]) == 0
    out = capsys.readouterr().out
    for expected in ("vivos", "common_voice", "piper", "kokoro", "omnivoice",
                     "wavlm", "wav2vec2", "whisper", "mlp", "deep_mlp"):
        assert expected in out
    assert "16000 Hz · mono · WAV/PCM_16 · 3-10s" in out


def test_set_override_is_parsed():
    args = build_parser().parse_args(["train", "--set", "train.lr=1e-4", "--set", "train.epochs=5"])
    assert args.set == ["train.lr=1e-4", "train.epochs=5"]


def test_ingest_accepts_path_positionally_and_as_option():
    assert build_parser().parse_args(["ingest", "/data/vivos"]).path == "/data/vivos"
    parsed = build_parser().parse_args(["ingest", "--path", "/data/vivos"])
    assert parsed.path_opt == "/data/vivos"


def test_missing_config_is_reported_cleanly(tmp_path):
    assert main(["info", "--config", str(tmp_path / "khong-co.yaml")]) == 1


@pytest.mark.parametrize("stage", STAGES)
def test_every_stage_has_a_subcommand(stage):
    args = build_parser().parse_args([stage])
    assert callable(args.func)


# ------------------------------- kiểm chất lượng phải chặn TRƯỚC khi sinh, không sau
def test_fixing_a_corpus_is_not_reported_as_a_failure(tmp_path, vivos_like, capsys):
    """`run()` ở notebook dừng cả phiên khi lệnh trả mã khác 0.

    `validate --fix` dọn xong thì corpus đã đạt chuẩn — trả 1 lúc đó là dừng phiên ngay
    sau khi việc vừa được sửa.
    """
    import soundfile as sf

    from aidetector.cli import main
    from aidetector.corpus.manifest import Manifest

    corpus = tmp_path / "corpus"
    common = ["--set", f"paths.corpus={corpus}"]
    assert main(["ingest", str(vivos_like), *common]) == 0

    manifest = Manifest.load(corpus, required=True)
    truoc = len(manifest)
    # Làm hỏng đúng một file: ghi đè bằng audio im lặng (check_quality → "silent").
    import numpy as np

    xau = manifest.abs_path(next(iter(manifest)))
    sf.write(str(xau), np.zeros(16_000 * 4, dtype="float32"), 16_000, subtype="PCM_16")

    assert main(["validate", *common]) == 1, "chưa sửa thì phải báo lỗi"
    assert main(["validate", "--fix", *common]) == 0, "đã dọn thì không phải lỗi"
    assert len(Manifest.load(corpus, required=True)) == truoc - 1
    assert main(["validate", *common]) == 0, "sau khi dọn phải sạch"


def test_fix_refuses_to_wipe_a_systemically_broken_corpus(tmp_path, vivos_like):
    """Hỏng cả một mảng lớn là lỗi chuỗi chuẩn hoá, không phải vài file xấu.

    Tự loại lúc đó là xoá corpus mà tưởng đang dọn rác.
    """
    from aidetector.cli import main
    from aidetector.corpus.manifest import Manifest

    corpus = tmp_path / "corpus"
    common = ["--set", f"paths.corpus={corpus}"]
    assert main(["ingest", str(vivos_like), *common]) == 0
    truoc = len(Manifest.load(corpus, required=True))

    # Siết min_seconds lên 30s ⇒ toàn bộ corpus "quá ngắn" theo spec mới.
    assert main(["validate", "--fix", "--set", "audio.min_seconds=30", *common]) == 1
    assert len(Manifest.load(corpus, required=True)) == truoc, "không được xoá gì"


def test_migrating_an_empty_corpus_is_not_a_failure(tmp_path, capsys):
    """Ô A1d chạy `migrate` TRƯỚC `ingest`, nên phiên đầu tiên luôn gặp corpus rỗng.

    Trả mã khác 0 lúc đó là `run()` ném SystemExit và dừng cả notebook ngay trước khi có
    gì để dọn — đúng lỗi một phiên Kaggle đã chết ở ô A1d với "Corpus rỗng".
    """
    corpus = tmp_path / "chua-co-gi"
    assert main(["migrate", "--set", f"paths.corpus={corpus}"]) == 0
    assert "rỗng" in capsys.readouterr().out


# ------------------------------------------ trạng thái nối tiếp, ghi vào dataset
def test_progress_records_which_speakers_are_finished(tmp_path, vivos_like, monkeypatch):
    """Đơn vị là speaker vì đó là ranh giới `generate` chốt tiến độ.

    Đích của một giọng là số real ĐỦ ĐIỀU KIỆN làm khuôn, không phải mọi real — lấy sai
    mẫu số thì tiến độ không bao giờ tới 100%.
    """
    import json

    import numpy as np

    from aidetector.cli import main
    from aidetector.generate import base as gen_base

    class Clone(gen_base.Generator):
        id = "progress_spy"
        kind = gen_base.KIND_CLONE
        native_sample_rate = 24_000

        def voices(self):
            return []

        def synthesize(self, text, voice=None, ref_audio=None, ref_text=None, language=None):
            return np.zeros(24_000 * 4, dtype=np.float32), self.native_sample_rate

    monkeypatch.setitem(gen_base._REGISTRY, Clone.id, Clone)
    corpus = tmp_path / "corpus"
    common = ["--set", f"paths.corpus={corpus}"]
    out = tmp_path / "progress.json"

    assert main(["ingest", str(vivos_like), *common]) == 0
    assert main(["progress", "--out", str(out), *common]) == 0
    state = json.loads(out.read_text(encoding="utf-8"))
    assert state["speakers_total"] == 8
    assert state["speakers_done"] == [] and len(state["speakers_todo"]) == 8
    assert state["targets_done"] == 0

    # Sinh hết: mọi giọng phải chuyển sang "xong".
    assert main(["generate", "--engines", Clone.id, *common]) == 0
    assert main(["progress", "--out", str(out), *common]) == 0
    state = json.loads(out.read_text(encoding="utf-8"))
    assert len(state["speakers_done"]) == 8, state["speakers_partial"]
    assert state["speakers_todo"] == [] and state["speakers_partial"] == {}
    assert state["targets_done"] == state["targets_total"] > 0
    assert state["dataset_records"] == state["real"] + state["fake"]


def test_progress_marks_a_half_finished_speaker_as_partial(tmp_path, vivos_like, monkeypatch):
    """Dở dang phải phân biệt với xong — nếu không, phiên sau bỏ qua phần còn nợ."""
    import json

    import numpy as np

    from aidetector.cli import main
    from aidetector.generate import base as gen_base

    class Clone(gen_base.Generator):
        id = "progress_spy2"
        kind = gen_base.KIND_CLONE
        native_sample_rate = 24_000

        def voices(self):
            return []

        def synthesize(self, text, voice=None, ref_audio=None, ref_text=None, language=None):
            return np.zeros(24_000 * 4, dtype=np.float32), self.native_sample_rate

    monkeypatch.setitem(gen_base._REGISTRY, Clone.id, Clone)
    corpus = tmp_path / "corpus"
    common = ["--set", f"paths.corpus={corpus}"]
    out = tmp_path / "progress.json"

    assert main(["ingest", str(vivos_like), *common]) == 0
    assert main(["generate", "--engines", Clone.id, "--count", "10", *common]) == 0
    assert main(["progress", "--out", str(out), *common]) == 0

    state = json.loads(out.read_text(encoding="utf-8"))
    assert state["speakers_partial"] or state["speakers_done"], state
    assert state["targets_done"] == 10
    for spk, v in state["speakers_partial"].items():
        assert 0 < v["fake"] < v["target"], (spk, v)


# ------------------------- "đã duyệt thì không duyệt lại" — trạng thái theo từng nguồn
def test_validation_is_not_repeated_for_records_already_approved(tmp_path, vivos_like, caplog):
    """Soi lại cả corpus mỗi phiên là đọc lại từng file audio — với 8.000 file là vài phút.

    Phần đã duyệt theo ĐÚNG chuẩn đó không thể đổi kết quả, nên nó phải được bỏ qua.
    """
    import logging

    from aidetector.cli import main
    from aidetector.corpus.manifest import Manifest

    corpus = tmp_path / "corpus"
    common = ["--set", f"paths.corpus={corpus}"]
    assert main(["ingest", str(vivos_like), *common]) == 0
    assert main(["validate", *common]) == 0

    # Dấu phải xuống đĩa, nếu không phiên sau lại đọc lại toàn bộ.
    tren_dia = Manifest.load(corpus, required=True)
    assert all(r.checked for r in tren_dia), "không bản ghi nào được đóng dấu"

    with caplog.at_level(logging.INFO):
        assert main(["validate", *common]) == 0
    assert "Bỏ qua" in caplog.text and "đã duyệt" in caplog.text

    caplog.clear()
    with caplog.at_level(logging.INFO):
        assert main(["validate", "--recheck", *common]) == 0
    assert "Bỏ qua" not in caplog.text, "--recheck phải soi lại tất cả"


def test_changing_the_standard_invalidates_every_approval(tmp_path, vivos_like, caplog):
    """"Đã duyệt" chỉ có nghĩa khi nói rõ duyệt theo chuẩn NÀO."""
    import logging

    from aidetector.cli import main

    corpus = tmp_path / "corpus"
    common = ["--set", f"paths.corpus={corpus}"]
    assert main(["ingest", str(vivos_like), *common]) == 0
    assert main(["validate", *common]) == 0

    # Ngưỡng đổi ⇒ vân tay đổi ⇒ không bản ghi nào còn được coi là đã duyệt.
    with caplog.at_level(logging.INFO):
        main(["validate", "--set", "audio.min_seconds=2.0", *common])
    assert "Bỏ qua" not in caplog.text


def test_progress_breaks_state_down_by_source(tmp_path, vivos_like):
    """Phiên sau hỏi "dataset_A đã có trên kho chưa, duyệt tới đâu" mà không tải cả manifest."""
    import json

    from aidetector.cli import main

    corpus = tmp_path / "corpus"
    common = ["--set", f"paths.corpus={corpus}"]
    out = tmp_path / "progress.json"

    assert main(["ingest", str(vivos_like), "--name", "dataset_A", *common]) == 0
    assert main(["validate", *common]) == 0
    assert main(["progress", "--out", str(out), *common]) == 0

    state = json.loads(out.read_text(encoding="utf-8"))
    assert "dataset_a" in state["by_source"], state["by_source"]
    o = state["by_source"]["dataset_a"]
    assert o["real"] == 32 and o["fake"] == 0 and o["approved"] == 32


def test_a_schema_error_can_actually_be_fixed(tmp_path, vivos_like):
    """Trước đây lỗi schema chỉ được ĐẾM, không vào danh sách loại.

    Hệ quả: corpus dính một bản ghi sai schema là `validate` đỏ vĩnh viễn, ô A4 dừng
    notebook sau nhiều giờ sinh, và `--fix` không có đường ra.
    """
    from aidetector.cli import main
    from aidetector.corpus.manifest import Manifest

    corpus = tmp_path / "corpus"
    common = ["--set", f"paths.corpus={corpus}"]
    assert main(["ingest", str(vivos_like), *common]) == 0

    m = Manifest.load(corpus, required=True)
    rec = m.reals[0]
    rec.generator = "piper:vi"          # real mà có generator ⇒ sai schema
    m.save()

    assert main(["validate", *common]) == 1
    assert main(["validate", "--fix", *common]) == 0
    assert main(["validate", *common]) == 0
    assert rec.utt_id not in Manifest.load(corpus, required=True)


def test_tightening_the_standard_does_not_silently_delete(tmp_path, vivos_like):
    """Siết ngưỡng rồi để `--fix` lặng lẽ xoá phần không còn lọt là mất dữ liệu mà
    không ai quyết. Đổi chuẩn là một quyết định."""
    from aidetector.cli import main
    from aidetector.corpus.manifest import Manifest

    corpus = tmp_path / "corpus"
    common = ["--set", f"paths.corpus={corpus}"]
    assert main(["ingest", str(vivos_like), *common]) == 0
    assert main(["validate", *common]) == 0          # đóng dấu theo chuẩn 3.0s
    truoc = len(Manifest.load(corpus, required=True))

    # Siết lên 5s: fixture dài 4,0–5,5s nên một phần rớt — nhưng chúng TỪNG đạt chuẩn.
    siet = ["--set", "audio.min_seconds=5.0", *common]
    assert main(["validate", "--fix", *siet]) == 1, "phải từ chối, không tự xoá"
    assert len(Manifest.load(corpus, required=True)) == truoc

    assert main(["validate", "--fix", "--force", *siet]) == 0
    assert len(Manifest.load(corpus, required=True)) < truoc


def test_prune_files_removes_the_audio_too(tmp_path, vivos_like):
    """Để file lại thì mỗi phiên `ingest` nạp lại rồi cấp số MỚI, tích dần file mồ côi."""
    import numpy as np
    import soundfile as sf

    from aidetector.cli import main
    from aidetector.corpus.manifest import Manifest

    corpus = tmp_path / "corpus"
    common = ["--set", f"paths.corpus={corpus}"]
    assert main(["ingest", str(vivos_like), *common]) == 0

    m = Manifest.load(corpus, required=True)
    xau = m.reals[0]
    duong_dan = m.abs_path(xau)
    sf.write(str(duong_dan), np.zeros(16_000 * 4, dtype="float32"), 16_000, subtype="PCM_16")

    assert main(["validate", "--fix", "--prune-files", *common]) == 0
    assert not duong_dan.exists(), "file mồ côi vẫn nằm lại"
    assert xau.utt_id not in Manifest.load(corpus, required=True)
