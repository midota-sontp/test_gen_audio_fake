"""Tự nhận diện loại dataset + các adapter tổng quát."""

from __future__ import annotations

import csv
from pathlib import Path

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


# ------------------------------------------------- `--limit` phải cắt đều mọi speaker
#
# Adapter duyệt theo thư mục nên nó trả hết giọng này mới sang giọng khác. Cắt theo thứ
# tự đó là những speaker cuối bảng không có lấy một utterance — mà chia tập là
# speaker-disjoint và fake chỉ sinh được cho speaker đã có real, nên mất speaker ở đây
# là mất luôn ở mọi bước sau.
def test_limit_covers_every_speaker_instead_of_the_first_few(tmp_path, vivos_like):
    from aidetector.corpus.manifest import Manifest
    from aidetector.ingest import ingest_source
    from aidetector.ingest.vivos import VivosAdapter

    manifest = Manifest(tmp_path / "corpus")
    # 8 speaker × 4 utt = 32; xin 8 ⇒ mỗi giọng đúng một utterance.
    result = ingest_source(manifest, VivosAdapter(), vivos_like, "vivos",
                           AudioSpec(), limit=8)

    speakers = {r.speaker for r in manifest.reals}
    assert len(speakers) == 8, f"chỉ phủ {len(speakers)}/8 speaker: {sorted(speakers)}"
    assert result["kept"] == 8


def test_limit_is_a_corpus_total_not_a_per_run_quota(tmp_path, vivos_like):
    """Corpus sống qua nhiều phiên Kaggle — chạy lại cùng lệnh phải ra cùng corpus.

    Nếu `limit` là "thêm bao nhiêu lần này" thì mỗi phiên lại cộng thêm 4000 real, tỉ lệ
    real/fake trượt dần và không ai thấy cho tới lúc đọc bảng cân bằng ở A4.
    """
    from aidetector.corpus.manifest import Manifest
    from aidetector.ingest import ingest_source
    from aidetector.ingest.vivos import VivosAdapter

    manifest = Manifest(tmp_path / "corpus")
    first = ingest_source(manifest, VivosAdapter(), vivos_like, "vivos", AudioSpec(), limit=8)
    n_after_first = len(manifest.reals)

    second = ingest_source(manifest, VivosAdapter(), vivos_like, "vivos", AudioSpec(), limit=8)

    assert n_after_first == first["kept"] == 8
    assert len(manifest.reals) == 8, "lượt hai nạp thêm ⇒ limit đang là quota mỗi lượt"
    assert second["kept"] == 0 and second["already"] == 8


def test_a_source_already_complete_is_not_a_failure(tmp_path, vivos_like, capsys):
    """`run("ingest")` ở notebook dừng cả phiên khi lệnh trả mã khác 0."""
    from aidetector.cli import main

    corpus = tmp_path / "corpus"
    args = ["ingest", str(vivos_like), "--limit", "8",
            "--set", f"paths.corpus={corpus}"]
    assert main(args) == 0
    assert main(args) == 0, "nguồn đã đủ ⇒ không nạp gì, nhưng đó không phải lỗi"


def test_in_memory_sources_keep_streaming(tmp_path):
    """Gom cả nguồn vào RAM để rải theo speaker chỉ được khi item mang đường dẫn."""
    import numpy as np

    from aidetector.ingest import _spread_by_speaker
    from aidetector.ingest.base import SourceItem

    items = [SourceItem(key=f"k{i}", audio=np.zeros(16_000, dtype=np.float32),
                        sample_rate=16_000, speaker=f"spk{i % 3}") for i in range(6)]
    assert [i.key for i in _spread_by_speaker(iter(items))] == [i.key for i in items]


def test_a_source_that_drops_most_of_itself_says_so_loudly(tmp_path, caplog):
    """VIVOS thật bị loại 3437/7437 utterance vì ngắn hơn min_seconds=3s.

    Đó là trần cứng cho `N_REAL` — nguồn 11.660 câu chỉ cấp được ~6.300. Ở dạng INFO nó
    nằm lẫn giữa hàng chục dòng log và người ta chỉ phát hiện khi thấy corpus hụt.
    """
    import logging

    import soundfile as sf

    from aidetector.corpus.manifest import Manifest
    from aidetector.ingest import ingest_source
    from aidetector.ingest.folder import FolderAdapter

    root = tmp_path / "ngan"
    root.mkdir()
    # 2 file đạt chuẩn · 5 file 2,5s (dưới 3s nhưng lấy lại được nếu hạ ngưỡng)
    # · 3 file 1s (hạ ngưỡng cũng không lấy lại được).
    for i, seconds in enumerate([4.0, 4.0] + [2.5] * 5 + [1.0] * 3):
        sf.write(str(root / f"spk{i % 3}_{i}.wav"),
                 speech_like(seconds, seed=i), SR, subtype="PCM_16")

    manifest = Manifest(tmp_path / "corpus")
    with caplog.at_level(logging.WARNING):
        result = ingest_source(manifest, FolderAdapter(), root, "ngan", AudioSpec())

    assert result["kept"] == 2 and result["drop_invalid"] == 8
    # Lý do, không chỉ số lượng — và con số dùng để quyết định hạ ngưỡng hay không.
    assert result["drops"]["too_short"] == 8
    assert result["drops"]["too_short_but_over_ref"] == 5
    assert "too_short=8" in caplog.text
    assert "min_seconds" in caplog.text and "lấy lại được 5" in caplog.text


def test_the_per_speaker_cap_says_when_it_is_the_binding_limit(tmp_path, vivos_like, caplog):
    """"Vì sao chỉ có N?" phải trả lời được từ log.

    Log phiên thật: giữ 5395 · bỏ 3700 — cộng lại 9095 trong khi adapter trả ra 12420.
    Phần chênh 3325 là utterance chưa hề được xét vì speaker đã đủ `--per-speaker`, mà
    dòng tổng kết lại không in bộ đếm đó, nên con số 5395 trông như từ trên trời.
    """
    import logging

    from aidetector.corpus.manifest import Manifest
    from aidetector.ingest import ingest_source
    from aidetector.ingest.vivos import VivosAdapter

    manifest = Manifest(tmp_path / "corpus")
    with caplog.at_level(logging.WARNING):
        result = ingest_source(manifest, VivosAdapter(), vivos_like, "vivos",
                               AudioSpec(), per_speaker=2)   # 8 giọng × 4 câu

    assert result["kept"] == 16 and result["skip_speaker_full"] == 16
    assert "chạm trần speaker" in caplog.text or "đủ trần --per-speaker" in caplog.text
    assert "--per-speaker=2" in caplog.text
    assert "chứ không phải nguồn đã cạn" in caplog.text


def test_the_summary_line_accounts_for_every_item_the_adapter_yielded(tmp_path, vivos_like):
    """kept + drop + skip_speaker_full + skip_exists phải cộng lại đúng số item."""
    from aidetector.corpus.manifest import Manifest
    from aidetector.ingest import ingest_source
    from aidetector.ingest.vivos import VivosAdapter

    manifest = Manifest(tmp_path / "corpus")
    r = ingest_source(manifest, VivosAdapter(), vivos_like, "vivos", AudioSpec(),
                      per_speaker=3)
    tong = r["kept"] + r["drop_invalid"] + r["skip_speaker_full"] + r["skip_exists"]
    assert tong == 32, f"thiếu {32 - tong} item không bộ đếm nào nhận"


def test_no_phantom_ceiling_when_the_whole_source_was_examined(tmp_path, vivos_like, caplog):
    """Xét hết nguồn rồi thì `kept` CHÍNH LÀ trần.

    Log phiên thật: giữ 8246 rồi báo "cả nguồn chỉ cấp được khoảng 8,245" — con số ngoại
    suy xấp xỉ chính nó, đọc lên như thể còn thiếu một utterance ở đâu đó.
    """
    import logging

    import soundfile as sf

    from aidetector.corpus.manifest import Manifest
    from aidetector.ingest import ingest_source
    from aidetector.ingest.folder import FolderAdapter

    root = tmp_path / "ngan"
    root.mkdir()
    for i, seconds in enumerate([4.0, 4.0] + [1.0] * 6):
        sf.write(str(root / f"spk{i % 2}_{i}.wav"), speech_like(seconds, seed=i),
                 SR, subtype="PCM_16")

    manifest = Manifest(tmp_path / "corpus")
    with caplog.at_level(logging.WARNING):
        ingest_source(manifest, FolderAdapter(), root, "ngan", AudioSpec())

    assert "Đã xét hết nguồn" in caplog.text
    assert "chỉ cấp được" not in caplog.text


# ------------------------------------------- nhập lại từ cây chuẩn (bộ data bên ngoài)
def _canonical_tree(tmp_path, vivos_like):
    """Dựng cây chuẩn bằng chính pipeline: ingest → corpus → đó là cây cần nhập lại."""
    from aidetector.corpus.manifest import Manifest
    from aidetector.ingest import ingest_source
    from aidetector.ingest.vivos import VivosAdapter

    corpus = tmp_path / "phat_hanh"
    m = Manifest(corpus)
    ingest_source(m, VivosAdapter(), vivos_like, "vivos", AudioSpec())
    m.save()
    return corpus, m


def test_canonical_tree_is_detected_over_the_generic_folder_adapter(tmp_path, vivos_like):
    from aidetector.ingest import detect_adapter

    corpus, _ = _canonical_tree(tmp_path, vivos_like)
    adapter, score, effective = detect_adapter(corpus)
    assert adapter.name == "canonical", f"nhận nhầm thành {adapter.name}"
    assert score >= 0.9 and effective == corpus


def test_canonical_import_round_trips_speakers_and_text(tmp_path, vivos_like):
    """Convert một lần rồi mọi thứ phía sau không cần biết bộ data vốn có cấu trúc gì."""
    from aidetector.corpus.manifest import Manifest
    from aidetector.ingest import ingest_source
    from aidetector.ingest.canonical import CanonicalAdapter

    corpus, goc = _canonical_tree(tmp_path, vivos_like)
    moi = Manifest(tmp_path / "corpus_moi")
    ket_qua = ingest_source(moi, CanonicalAdapter(), corpus, "phat_hanh", AudioSpec())

    assert ket_qua["kept"] == len(goc.reals)
    assert {r.speaker for r in moi.reals} == {r.speaker for r in goc.reals}
    # Transcript phải theo đúng từng file: tên file chỉ là số thứ tự nên khoá theo stem
    # sẽ trộn lẫn transcript giữa các giọng.
    assert all(r.text for r in moi.reals)
    assert {r.text for r in moi.reals} == {r.text for r in goc.reals}


def test_canonical_import_is_idempotent(tmp_path, vivos_like):
    from aidetector.corpus.manifest import Manifest
    from aidetector.ingest import ingest_source
    from aidetector.ingest.canonical import CanonicalAdapter

    corpus, _ = _canonical_tree(tmp_path, vivos_like)
    moi = Manifest(tmp_path / "corpus_moi")
    ingest_source(moi, CanonicalAdapter(), corpus, "phat_hanh", AudioSpec())
    truoc = {r.utt_id for r in moi}
    ingest_source(moi, CanonicalAdapter(), corpus, "phat_hanh", AudioSpec())
    assert {r.utt_id for r in moi} == truoc


def test_canonical_import_leaves_fake_to_the_pipeline(tmp_path, vivos_like):
    """Fake nhập từ ngoài không có `ref_utt_id` ⇒ không ghép cặp được với real nào."""
    from aidetector.corpus.manifest import Manifest
    from aidetector.ingest import ingest_source
    from aidetector.ingest.canonical import CanonicalAdapter

    corpus, _ = _canonical_tree(tmp_path, vivos_like)
    (corpus / "fake" / "engine_la" / "spk").mkdir(parents=True)
    import shutil

    nguon = next((corpus / "real").glob("*/*/*.wav"))
    shutil.copy(nguon, corpus / "fake" / "engine_la" / "spk" / "0001.wav")

    moi = Manifest(tmp_path / "corpus_moi")
    ingest_source(moi, CanonicalAdapter(), corpus, "phat_hanh", AudioSpec())
    assert not moi.fakes, "fake phải do `generate` sinh, không nhập từ ngoài"


def test_the_documented_convert_example_actually_works(tmp_path):
    """Ví dụ trong ô A1c phải chạy được thật, không chỉ trông hợp lý.

    Dựng đúng bộ dữ liệu mẫu (speaker nằm trong TÊN FILE), convert theo đúng cách tài
    liệu mô tả, rồi khẳng định `canonical` đọc lại được cả speaker lẫn transcript.
    """
    import csv
    import shutil

    from aidetector.corpus.manifest import Manifest
    from aidetector.ingest import detect_adapter, ingest_source
    from conftest import SENTENCES

    SOURCE = "dataset_b"
    raw = tmp_path / "dataset-b" / "audio"
    raw.mkdir(parents=True)
    cau = {}
    for spk, n in (("001_nguyen_van_a", 4), ("002_tran_thi_b", 4), ("003_le_van_c", 4)):
        for i in range(1, n + 1):
            ten = f"{spk}_{i:04d}.wav"
            sf.write(str(raw / ten), speech_like(4.0, seed=hash(ten) % 9999), SR,
                     subtype="PCM_16")
            cau[ten] = SENTENCES[i % len(SENTENCES)]
    with (raw.parent / "labels.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["file", "transcript"])
        for ten, txt in cau.items():
            w.writerow([ten, txt])

    # ── đúng những gì tài liệu bảo dev làm ────────────────────────────────
    out = tmp_path / "converted"
    rows = []
    for wav in sorted(raw.glob("*.wav")):
        speaker = wav.name.rsplit("_", 1)[0]          # speaker nằm trong tên file
        dich = out / "real" / SOURCE / speaker / wav.name
        dich.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(wav, dich)
        rows.append((str(dich.relative_to(out)), cau[wav.name]))
    with (out / "metadata.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["path", "text"])
        w.writerows(rows)

    # ── cây đó phải được nhận và đọc đúng ─────────────────────────────────
    adapter, score, effective = detect_adapter(out)
    assert adapter.name == "canonical" and effective == out, (adapter.name, score)

    m = Manifest(tmp_path / "corpus")
    kq = ingest_source(m, adapter(), out, SOURCE, AudioSpec())
    assert kq["kept"] == 12 and kq["speakers"] == 3
    assert {r.speaker for r in m.reals} == {"001_nguyen_van_a", "002_tran_thi_b",
                                            "003_le_van_c"}
    assert all(r.text for r in m.reals), "transcript phải khớp theo từng đường dẫn"
    assert all(r.path.startswith(f"real/{SOURCE}/") for r in m.reals)


# --------------------------- convert + đánh giá đầu vào trong MỘT bước
def _bo_du_lieu_la(tmp_path):
    """Bộ dữ liệu lạ: speaker nằm trong TÊN FILE, transcript ở labels.csv."""
    import csv

    from conftest import SENTENCES

    raw = tmp_path / "dataset-b" / "audio"
    raw.mkdir(parents=True)
    cau = {}
    for spk in ("001_a", "002_b", "003_c"):
        for i in range(1, 5):
            ten = f"{spk}_{i:04d}.wav"
            sf.write(str(raw / ten), speech_like(4.0, seed=hash(ten) % 9999), SR,
                     subtype="PCM_16")
            cau[ten] = SENTENCES[i % len(SENTENCES)]
    with (raw.parent / "labels.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["file", "transcript"])
        w.writerows(cau.items())
    return raw, cau


def _convert_dung(cau, tang_nguon="dataset_b"):
    import csv
    import shutil

    def convert(raw, out):
        rows = []
        for wav in sorted(raw.glob("*.wav")):
            speaker = wav.name.rsplit("_", 1)[0]
            dich = out / "real" / tang_nguon / speaker / wav.name
            dich.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(wav, dich)
            rows.append((str(dich.relative_to(out)), cau[wav.name]))
        with (out / "metadata.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["path", "text"])
            w.writerows(rows)

    return convert


def test_convert_and_verify_runs_both_in_one_step(tmp_path):
    from aidetector.ingest import convert_and_verify

    raw, cau = _bo_du_lieu_la(tmp_path)
    kq = convert_and_verify("dataset_b", raw, _convert_dung(cau),
                            out=tmp_path / "converted")

    assert kq["converted"] and not kq["skipped"]
    assert kq["report"]["ok"] and kq["report"]["speakers"] == 3
    assert kq["report"]["with_text"] == 12
    assert Path(kq["root"]).name == "converted"


def test_the_store_short_circuits_everything(tmp_path):
    """Đã có trong kho ⇒ không gọi CONVERT, không kiểm gì cả."""
    from aidetector.ingest import convert_and_verify

    raw, _ = _bo_du_lieu_la(tmp_path)

    def khong_duoc_goi(a, b):
        raise AssertionError("CONVERT bị gọi dù kho đã có nguồn này")

    kq = convert_and_verify("dataset_b", raw, khong_duoc_goi,
                            out=tmp_path / "converted", already=8246)
    assert kq["skipped"] and kq["already"] == 8246
    assert not (tmp_path / "converted").exists()


def test_a_convert_writing_the_wrong_folder_name_is_caught(tmp_path):
    """`SOURCE` là khoá hỏi kho — lệch một chữ là phiên sau convert lại từ đầu."""
    from aidetector.ingest import convert_and_verify

    raw, cau = _bo_du_lieu_la(tmp_path)
    with pytest.raises(ValueError, match="tên nguồn phải khớp"):
        convert_and_verify("dataset_b", raw, _convert_dung(cau, tang_nguon="go_nham"),
                           out=tmp_path / "converted")


def test_input_without_transcript_is_rejected(tmp_path):
    """Fake không ghép cặp được với real nào — cả thiết kế corpus dựa vào việc ghép cặp."""
    import shutil

    from aidetector.ingest import convert_and_verify

    raw, _ = _bo_du_lieu_la(tmp_path)

    def convert(r, out):
        for wav in sorted(r.glob("*.wav")):
            dich = out / "real" / "dataset_b" / wav.name.rsplit("_", 1)[0] / wav.name
            dich.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(wav, dich)          # cố ý không ghi metadata.csv

    with pytest.raises(ValueError, match="transcript"):
        convert_and_verify("dataset_b", raw, convert, out=tmp_path / "converted")


def test_verification_runs_even_without_a_convert_function(tmp_path):
    """Đường KHÔNG có CONVERT là đường hay hỏng nhất: adapter sẵn có đọc sai tầng speaker."""
    from aidetector.ingest import convert_and_verify

    phang = tmp_path / "mot_dong"
    phang.mkdir()
    for i in range(6):
        sf.write(str(phang / f"{i}.wav"), speech_like(4.0, seed=i), SR, subtype="PCM_16")

    with pytest.raises(ValueError, match="speaker"):
        convert_and_verify("phang", phang)


def test_flat_recordings_are_screened_then_placed(tmp_path, caplog):
    """`/dataset_A/56456456456456.mp3` → đánh giá → đạt → thư mục riêng, tên `_001`.

    File không đạt thì KHÔNG vào cây: đỡ cho `ingest` giải mã rồi `validate` loại lại.
    """
    import logging

    from aidetector.ingest import convert_flat_recordings

    raw = tmp_path / "dataset-a"
    raw.mkdir()
    for ten in ("56456456456456", "78978978978978", "12312312312312"):
        sf.write(str(raw / f"{ten}.wav"), speech_like(4.0, seed=hash(ten) % 999), SR,
                 subtype="PCM_16")
    # Ba kiểu không đạt, mỗi kiểu một lý do chuẩn hoá không sửa được.
    sf.write(str(raw / "qua_ngan.wav"), speech_like(1.0, seed=7), SR, subtype="PCM_16")
    sf.write(str(raw / "rate_thap.wav"), speech_like(4.0, seed=8)[::2], 8_000,
             subtype="PCM_16")
    (raw / "khong_doc_duoc.wav").write_bytes(b"khong phai wav")

    out = tmp_path / "converted"
    with caplog.at_level(logging.INFO):
        kq = convert_flat_recordings(raw, out, source="dataset_a")

    assert kq["recordings"] == 6 and kq["kept"] == 3
    assert set(kq["rejected"]) == {"ngắn hơn 3s", "sample rate 8000 < 16000",
                                   "đọc không được (LibsndfileError)"}

    goc = out / "real" / "dataset_a"
    assert sorted(p.name for p in goc.iterdir()) == [
        "12312312312312", "56456456456456", "78978978978978"]
    # Quy ước tên: <stem>_001 trong thư mục <stem>.
    assert (goc / "56456456456456" / "56456456456456_001.wav").exists()
    # Chỉ CHÉP, không giải mã: byte phải y hệt file gốc.
    assert (goc / "56456456456456" / "56456456456456_001.wav").read_bytes() == \
        (raw / "56456456456456.wav").read_bytes()


def test_two_files_of_one_speaker_get_separate_numbers(tmp_path):
    """Nhiều file dồn về một speaker thì không được đè nhau."""
    from aidetector.ingest import convert_flat_recordings

    raw = tmp_path / "raw" / "nguoi_a"
    raw.mkdir(parents=True)
    for i in (1, 2, 3):
        sf.write(str(raw / f"lan{i}.wav"), speech_like(4.0, seed=i), SR, subtype="PCM_16")

    out = tmp_path / "converted"
    kq = convert_flat_recordings(raw.parent, out, source="d", speaker_from="parent")

    assert kq["kept"] == 3
    tep = sorted(p.name for p in (out / "real" / "d" / "nguoi_a").iterdir())
    assert tep == ["nguoi_a_001.wav", "nguoi_a_002.wav", "nguoi_a_003.wav"]


def test_screening_leaves_post_normalisation_checks_to_validate(tmp_path):
    """Clipping và gần-im-lặng chỉ có nghĩa SAU chuẩn hoá — sàng ở nguồn là sai đối tượng.

    File im lặng hoàn toàn nhưng dài và đúng sample rate vẫn qua được cửa convert; nó bị
    `validate` loại sau khi `ingest` chuẩn hoá. Đó là đúng chỗ.
    """
    import numpy as np

    from aidetector.ingest import convert_flat_recordings

    raw = tmp_path / "raw"
    raw.mkdir()
    sf.write(str(raw / "im_lang.wav"), np.zeros(SR * 4, dtype="float32"), SR,
             subtype="PCM_16")
    kq = convert_flat_recordings(raw, tmp_path / "out", source="d")
    assert kq["kept"] == 1 and not kq["rejected"]


def test_the_screening_function_can_be_replaced(tmp_path):
    """Mỗi bộ dữ liệu có kiểu rác riêng — cái gì đáng loại ở nguồn thì chỉ người biết bộ
    đó mới nói được, nên phép đánh giá phải truyền vào được."""
    from aidetector.ingest import convert_flat_recordings

    raw = tmp_path / "raw"
    raw.mkdir()
    for ten in ("tot_a", "tot_b", "tot_c", "NHAP_x", "NHAP_y"):
        sf.write(str(raw / f"{ten}.wav"), speech_like(4.0, seed=hash(ten) % 99), SR,
                 subtype="PCM_16")

    def cua_toi(f):
        # Quy ước riêng của bộ này: tiền tố NHAP_ là bản thu thử, không dùng.
        return "bản thu thử (NHAP_)" if f.stem.startswith("NHAP_") else None

    kq = convert_flat_recordings(raw, tmp_path / "out", source="d", screen=cua_toi)

    assert kq["recordings"] == 5 and kq["kept"] == 3
    assert kq["rejected"] == {"bản thu thử (NHAP_)": 2}


def test_a_custom_screen_can_build_on_the_default(tmp_path):
    """Thêm luật riêng mà không mất ba phép sàng mặc định."""
    from aidetector.ingest import convert_flat_recordings, screen_source_file

    raw = tmp_path / "raw"
    raw.mkdir()
    sf.write(str(raw / "dai_va_tot.wav"), speech_like(4.0, seed=1), SR, subtype="PCM_16")
    sf.write(str(raw / "qua_ngan.wav"), speech_like(1.0, seed=2), SR, subtype="PCM_16")
    sf.write(str(raw / "loai_tay.wav"), speech_like(4.0, seed=3), SR, subtype="PCM_16")

    def cua_toi(f):
        return screen_source_file(f) or ("trong danh sách đen"
                                         if f.stem == "loai_tay" else None)

    kq = convert_flat_recordings(raw, tmp_path / "out", source="d", screen=cua_toi)
    assert kq["kept"] == 1
    assert set(kq["rejected"]) == {"ngắn hơn 3s", "trong danh sách đen"}
