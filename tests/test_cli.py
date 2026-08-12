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
