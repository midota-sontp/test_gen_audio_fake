"""Chạy trọn pipeline từ dataset thô đến suy luận, dùng engine/backbone giả."""

from __future__ import annotations

import numpy as np
import pytest

from aidetector.augment import AugmentChain, augment_corpus
from aidetector.config import Config
from aidetector.corpus.manifest import Manifest
from aidetector.corpus.schema import LABEL_FAKE, LABEL_REAL
from aidetector.corpus.spec import AudioSpec
from aidetector.detect import Detector
from aidetector.evaluate import evaluate
from aidetector.features import FeatureStore, extract_features
from aidetector.features.backbones import build_backbone
from aidetector.generate import generate_fakes
from aidetector.ingest import ingest_source
from aidetector.ingest.vivos import VivosAdapter
from aidetector.splits import assign_splits
from aidetector.train import train

SPEC = AudioSpec()


@pytest.fixture
def corpus(tmp_path, vivos_like):
    manifest = Manifest(tmp_path / "corpus")
    ingest_source(manifest, VivosAdapter(), vivos_like, "vivos", SPEC)
    manifest.save()
    return manifest


# ------------------------------------------------------------------ ingest
def test_ingest_normalises_everything(corpus):
    assert len(corpus) > 0
    assert len(corpus.reals) == len(corpus)
    for rec in corpus:
        assert rec.label == LABEL_REAL
        assert rec.sample_rate == 16_000
        assert 3.0 <= rec.duration <= 10.0
        assert rec.text                     # prompts.txt đã được ghép vào
        assert corpus.abs_path(rec).exists()
        assert rec.path.startswith("audio/real/vivos/")


def test_ingest_is_idempotent(corpus, vivos_like):
    before = len(corpus)
    stats = ingest_source(corpus, VivosAdapter(), vivos_like, "vivos", SPEC)
    assert len(corpus) == before
    assert stats["kept"] == 0 and stats["skip_exists"] > 0


def test_ingest_respects_limits(tmp_path, vivos_like):
    manifest = Manifest(tmp_path / "c2")
    ingest_source(manifest, VivosAdapter(), vivos_like, "vivos", SPEC, per_speaker=2)
    from collections import Counter

    counts = Counter(r.speaker for r in manifest)
    assert counts and max(counts.values()) <= 2


# ------------------------------------------------------------------ generate
def test_generate_tts_pairs_with_real(corpus):
    n_real = len(corpus.reals)
    stats = generate_fakes(corpus, "dummy_tts", SPEC, count=n_real)
    assert stats["kept"] > 0
    fakes = corpus.fakes
    assert len(fakes) >= n_real * 0.5
    for rec in fakes:
        assert rec.label == LABEL_FAKE
        assert rec.generator.startswith("dummy_tts:")
        assert rec.engine == "dummy_tts"
        assert rec.ref_utt_id in corpus            # fake luôn có real đối chứng
        assert corpus.get(rec.ref_utt_id).speaker == rec.speaker
        assert corpus.get(rec.ref_utt_id).text == rec.text


def test_generate_clone_uses_a_different_utterance_as_reference(corpus):
    stats = generate_fakes(corpus, "dummy_clone", SPEC, count=8)
    assert stats["kept"] > 0
    assert all(r.engine == "dummy_clone" for r in corpus.fakes)


def test_generate_is_idempotent(corpus):
    generate_fakes(corpus, "dummy_tts", SPEC, count=6)
    before = len(corpus.fakes)
    stats = generate_fakes(corpus, "dummy_tts", SPEC, count=6)
    assert len(corpus.fakes) == before
    assert stats["skip_exists"] > 0


def test_multiple_engines_coexist(corpus):
    generate_fakes(corpus, "dummy_tts", SPEC, count=6)
    generate_fakes(corpus, "dummy_clone", SPEC, count=6)
    engines = {r.engine for r in corpus.fakes}
    assert engines == {"dummy_tts", "dummy_clone"}


# ------------------------------------------------------------------ augment
def test_augment_keeps_clean_and_adds_noisy(corpus):
    generate_fakes(corpus, "dummy_tts", SPEC, count=len(corpus.reals))
    assign_splits(corpus, seed=1)
    clean_before = [r.utt_id for r in corpus if not r.augment]

    chain = AugmentChain({"gaussian_noise": {"p": 1.0}, "gain": {"p": 0.5}}, max_ops=2)
    stats = augment_corpus(corpus, SPEC, chain, copies=1, splits=("train",), seed=1)

    assert stats["created"] > 0
    assert all(utt in corpus for utt in clean_before)          # bản clean còn nguyên
    augmented = [r for r in corpus if r.augment]
    assert augmented
    for rec in augmented:
        parent = corpus.get(rec.parent_utt_id)
        assert parent is not None
        assert rec.split == parent.split == "train"            # không rò rỉ sang split khác
        assert rec.label == parent.label
        assert 3.0 <= rec.duration <= 10.0


def test_augment_covers_both_classes(corpus):
    generate_fakes(corpus, "dummy_tts", SPEC, count=len(corpus.reals))
    assign_splits(corpus, seed=1)
    chain = AugmentChain({"gaussian_noise": {"p": 1.0}}, max_ops=1)
    augment_corpus(corpus, SPEC, chain, copies=1, splits=("train",), seed=1)
    labels = {r.label for r in corpus if r.augment}
    assert labels == {LABEL_REAL, LABEL_FAKE}


# ------------------------------------------------------------------ splits
def test_splits_are_speaker_disjoint(corpus):
    generate_fakes(corpus, "dummy_tts", SPEC, count=len(corpus.reals))
    report = assign_splits(corpus, ratios=(0.6, 0.2, 0.2), seed=7)
    assert report["speaker_leaks"] == []
    assert sum(sum(c.values()) for c in report["counts"].values()) == len(corpus)


def test_holdout_generator_only_appears_in_test(corpus):
    generate_fakes(corpus, "dummy_tts", SPEC, count=10)
    generate_fakes(corpus, "dummy_clone", SPEC, count=10)
    assign_splits(corpus, seed=7, holdout_generators=["dummy_clone"])
    held = [r for r in corpus if r.engine == "dummy_clone"]
    assert held and all(r.split == "test" for r in held)


# ------------------------------------------------------------------ features
def _dummy_backbone():
    return build_backbone({"name": "dummy", "output_layer": 0, "pooling": "mean"}, "cpu")


def test_features_cache_is_keyed_by_utt_id(corpus, tmp_path):
    generate_fakes(corpus, "dummy_tts", SPEC, count=8)
    assign_splits(corpus, seed=3)
    backbone = _dummy_backbone()
    result = extract_features(corpus, backbone, SPEC, cache_root=tmp_path / "feat")
    assert result["extracted"] == len(corpus)
    assert result["dim"] == 16

    again = extract_features(corpus, backbone, SPEC, cache_root=tmp_path / "feat")
    assert again["extracted"] == 0 and again["cached"] == len(corpus)

    store = FeatureStore(tmp_path / "feat", backbone)
    X, y, kept = store.load_many(list(corpus))
    assert X.shape == (len(corpus), 16) and len(kept) == len(corpus)
    assert set(np.unique(y)) == {0, 1}


def test_changing_backbone_settings_uses_a_new_cache(tmp_path):
    a = build_backbone({"name": "dummy", "output_layer": 0, "pooling": "mean"}, "cpu")
    b = build_backbone({"name": "dummy", "output_layer": 3, "pooling": "mean"}, "cpu")
    c = build_backbone({"name": "dummy", "output_layer": 0, "pooling": "mean_std"}, "cpu")
    assert len({a.cache_key, b.cache_key, c.cache_key}) == 3
    assert c.output_dim == 2 * a.output_dim


# ------------------------------------------------- train / evaluate / detect
@pytest.fixture
def trained(corpus, tmp_path):
    generate_fakes(corpus, "dummy_tts", SPEC, count=len(corpus.reals))
    assign_splits(corpus, ratios=(0.6, 0.2, 0.2), seed=11)
    corpus.save()

    backbone = _dummy_backbone()
    extract_features(corpus, backbone, SPEC, cache_root=tmp_path / "feat")

    cfg = Config({
        "seed": 42,
        "audio": {},
        "model": {"head": "mlp", "hidden_dim": 32, "dropout": 0.1},
        "train": {"epochs": 40, "batch_size": 8, "lr": 1e-3,
                  "early_stopping": {"monitor": "val_eer", "patience": 15}},
    })
    summary = train(
        corpus, backbone, cfg,
        cache_root=tmp_path / "feat",
        checkpoint_dir=tmp_path / "ckpt",
        report_dir=tmp_path / "rep",
        device="cpu",
    )
    return corpus, backbone, cfg, summary, tmp_path


def test_train_produces_checkpoint_and_history(trained):
    _, _, _, summary, tmp_path = trained
    assert (tmp_path / "ckpt" / "best.pt").exists()
    assert (tmp_path / "rep" / "history.csv").exists()
    assert summary["best_epoch"] >= 1
    # Hai lớp trong dữ liệu giả tách nhau rất rõ ⇒ EER phải rất thấp.
    assert summary["best_val_eer"] < 0.2


def test_evaluate_writes_metrics_and_breakdown(trained):
    corpus, backbone, _, _, tmp_path = trained
    result = evaluate(
        corpus, backbone,
        checkpoint=tmp_path / "ckpt" / "best.pt",
        cache_root=tmp_path / "feat",
        report_dir=tmp_path / "rep",
        split="test", device="cpu", make_plots=False,
    )
    assert (tmp_path / "rep" / "metrics.json").exists()
    assert (tmp_path / "rep" / "predictions.csv").exists()
    assert 0.0 <= result["overall"]["eer"] <= 1.0
    assert result["overall"]["n_real"] > 0 and result["overall"]["n_fake"] > 0
    assert any(k.startswith("dummy_tts") for k in result["by_generator"])


def test_detector_scores_a_file(trained):
    corpus, _, _, _, tmp_path = trained
    detector = Detector(checkpoint=tmp_path / "ckpt" / "best.pt", device="cpu")
    real = corpus.reals[0]
    out = detector.predict(corpus.abs_path(real))
    assert out["label"] in ("REAL", "FAKE")
    assert 0.0 <= out["score_fake"] <= 1.0
    assert out["n_chunks"] >= 1


def test_detector_pads_short_audio_and_warns(trained, tmp_path):
    """Audio ngắn hơn chuẩn vẫn được chấm điểm, nhưng phải kèm cảnh báo."""
    import soundfile as sf

    from tests.conftest import speech_like

    path = tmp_path / "tooshort.wav"
    sf.write(str(path), speech_like(1.0, seed=99), 16_000, subtype="PCM_16")
    detector = Detector(checkpoint=tmp_path / "ckpt" / "best.pt", device="cpu")
    out = detector.predict(path)
    assert "error" not in out
    assert out["label"] in ("REAL", "FAKE")
    assert out["duration"] == 1.0
    assert "ngắn hơn chuẩn" in out["warning"]


def test_detector_reports_unreadable_file(trained, tmp_path):
    path = tmp_path / "khong-phai-audio.wav"
    path.write_text("đây chỉ là văn bản", encoding="utf-8")
    detector = Detector(checkpoint=tmp_path / "ckpt" / "best.pt", device="cpu")
    assert "error" in detector.predict(path)


def test_detector_threshold_comes_from_checkpoint(trained, tmp_path):
    detector = Detector(checkpoint=tmp_path / "ckpt" / "best.pt", device="cpu")
    assert 0.0 < detector.threshold < 1.0
    assert detector.threshold == pytest.approx(detector.meta["threshold"])


# ------------------------------------------------------------------ manifest
def test_manifest_roundtrip(corpus, tmp_path):
    generate_fakes(corpus, "dummy_tts", SPEC, count=4)
    corpus.save()
    reloaded = Manifest.load(corpus.root, required=True)
    assert len(reloaded) == len(corpus)
    for rec in corpus:
        other = reloaded.get(rec.utt_id)
        assert other is not None
        assert other.to_row() == rec.to_row()
