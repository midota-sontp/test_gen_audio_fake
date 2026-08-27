"""Corpus tách theo BỘ DỮ LIỆU: mỗi bộ một thư mục tự chứa.

    corpus/
      vivos/metadata.csv · vivos/real/<speaker>/ · vivos/fake/<engine>/<speaker>/
      abc/metadata.csv   · abc/real/<speaker>/   · abc/fake/<engine>/<speaker>/

Trong bộ nhớ vẫn là MỘT bảng hợp nhất — chia tập speaker-disjoint, cân bằng lớp và
huấn luyện đều phải nhìn toàn bộ dữ liệu cùng lúc. Những test dưới đây canh đúng hai
điều đó cùng lúc: trên đĩa thì tách, trong bộ nhớ thì hợp.
"""

from __future__ import annotations

import csv

import numpy as np

from aidetector.corpus.manifest import (
    MANIFEST_NAME, SUPERSEDED_NAME, Manifest, find_manifest, find_shards,
)
from aidetector.corpus.schema import LABEL_FAKE, LABEL_REAL, Record, make_utt_id
from aidetector.corpus.spec import AudioSpec

SPEC = AudioSpec()


def _ghi(manifest: Manifest, source: str, speaker: str, n: int = 2,
         generator: str = "") -> list[Record]:
    """Thêm `n` bản ghi thật vào corpus, có audio trên đĩa."""
    ra = []
    label = LABEL_FAKE if generator else LABEL_REAL
    for i in range(n):
        key = f"{speaker}-{i}"
        rec = Record(
            utt_id=make_utt_id(source, speaker, key + generator),
            path="", label=label, source=source, speaker=speaker,
            text="một câu tiếng việt đủ dài để dùng làm khuôn sinh fake",
            generator=generator,
        )
        ra.append(manifest.write_audio(rec, np.zeros(4 * SPEC.sample_rate, np.float32), SPEC))
    return ra


def _hai_bo(root) -> Manifest:
    m = Manifest(root)
    _ghi(m, "vivos", "vivosspk01")
    _ghi(m, "vivos", "vivosspk02")
    _ghi(m, "vivos", "vivosspk01", generator="omnivoice:cp")
    _ghi(m, "abc", "nguyen_van_a")
    _ghi(m, "abc", "nguyen_van_a", generator="omnivoice:cp")
    m.save()
    return m


# ----------------------------------------------------- trên đĩa: mỗi bộ một thư mục
def test_each_dataset_gets_its_own_manifest(tmp_path):
    corpus = tmp_path / "corpus"
    _hai_bo(corpus)

    assert [p.parent.name for p in find_shards(corpus)] == ["abc", "vivos"]
    # Không còn bảng gộp ở gốc — đó là cả điểm của cấu trúc này.
    assert find_manifest(corpus) is None

    for bo in ("vivos", "abc"):
        with (corpus / bo / MANIFEST_NAME).open(newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        assert rows, f"{bo} không có bản ghi nào"
        assert {r["source"] for r in rows} == {bo}, "manifest của một bộ chỉ kể bộ đó"


def test_a_dataset_folder_holds_all_of_its_audio_and_nothing_else(tmp_path):
    corpus = tmp_path / "corpus"
    m = _hai_bo(corpus)

    for rec in m:
        assert rec.path.split("/")[0] == rec.source
        assert m.abs_path(rec).exists()
        assert m.abs_path(rec).is_relative_to(corpus / rec.source)

    # Xoá một bộ = xoá một thư mục. Không sót file nào của nó ở ngoài.
    con_lai = {p for p in corpus.rglob("*.wav") if not p.is_relative_to(corpus / "abc")}
    assert all("abc" not in p.parts for p in con_lai)


def test_fake_lives_in_the_folder_of_the_dataset_that_produced_it(tmp_path):
    """Fake thừa hưởng `source` của real gốc, nên nó phải về đúng thư mục bộ đó."""
    corpus = tmp_path / "corpus"
    m = _hai_bo(corpus)

    for rec in m.fakes:
        tang = rec.path.split("/")
        assert tang[0] == rec.source and tang[1] == "fake" and tang[2] == "omnivoice"


# ------------------------------------------------- trong bộ nhớ: một bảng hợp nhất
def test_loading_unions_every_dataset(tmp_path):
    corpus = tmp_path / "corpus"
    goc = _hai_bo(corpus)

    lai = Manifest.load(corpus, required=True)
    assert len(lai) == len(goc)
    assert {r.source for r in lai} == {"vivos", "abc"}
    for rec in goc:
        assert lai.get(rec.utt_id).to_row() == rec.to_row()


def test_adding_a_dataset_leaves_the_other_untouched(tmp_path):
    """Thêm bộ mới không được ghi lại một byte nào của bộ cũ."""
    corpus = tmp_path / "corpus"
    _hai_bo(corpus)
    truoc = (corpus / "vivos" / MANIFEST_NAME).read_bytes()

    m = Manifest.load(corpus, required=True)
    _ghi(m, "bo_thu_ba", "spk_moi")
    m.save()

    assert (corpus / "vivos" / MANIFEST_NAME).read_bytes() == truoc
    assert len(Manifest.load(corpus, required=True)) == len(m)


def test_emptying_a_dataset_does_not_resurrect_it(tmp_path):
    """`validate --fix` loại sạch một bộ ⇒ file manifest của nó phải được ghi lại rỗng.

    Bỏ qua nó là để bản cũ nằm lại trên đĩa, và lượt nạp sau dựng ngược đúng những bản
    ghi vừa bị loại — im lặng và rất khó lần ra.
    """
    corpus = tmp_path / "corpus"
    _hai_bo(corpus)

    m = Manifest.load(corpus, required=True)
    for rec in [r for r in m if r.source == "abc"]:
        m.remove(rec.utt_id)
    m.save()

    lai = Manifest.load(corpus, required=True)
    assert {r.source for r in lai} == {"vivos"}
    assert (corpus / "abc" / MANIFEST_NAME).exists(), "phải ghi lại rỗng, không bỏ mặc"


# ------------------------------------------------------------ corpus cấu trúc cũ
def _corpus_cu(root) -> Manifest:
    """Dựng corpus theo cấu trúc CŨ: một manifest gộp ở gốc, cây `real/<bộ>/<speaker>/`."""
    root.mkdir(parents=True, exist_ok=True)
    recs = []
    for source, speaker in (("vivos", "vivosspk01"), ("vivos", "vivosspk02")):
        for i in range(2):
            duong = f"real/{source}/{speaker}/{i + 1:04d}.wav"
            rec = Record(utt_id=make_utt_id(source, speaker, f"cu-{i}"), path=duong,
                         label=LABEL_REAL, source=source, speaker=speaker,
                         text="câu cũ", duration=4.0)
            (root / duong).parent.mkdir(parents=True, exist_ok=True)
            from aidetector.corpus.spec import save_audio

            save_audio(root / duong, np.zeros(4 * SPEC.sample_rate, np.float32), SPEC)
            recs.append(rec)
    from aidetector.corpus.manifest import manifest_csv

    (root / MANIFEST_NAME).write_text(manifest_csv(recs), newline="", encoding="utf-8")
    return Manifest(root, recs)


def test_an_old_flat_corpus_is_still_readable(tmp_path):
    """Cột `path` là nguồn sự thật nên corpus đã đẩy lên Kaggle vẫn đọc được nguyên vẹn."""
    corpus = tmp_path / "corpus"
    cu = _corpus_cu(corpus)

    m = Manifest.load(corpus, required=True)
    assert len(m) == len(cu)
    for rec in m:
        assert m.abs_path(rec).exists(), "không tra được file của corpus cũ"


def test_saving_an_old_corpus_splits_it_by_dataset_and_keeps_the_original(tmp_path):
    corpus = tmp_path / "corpus"
    _corpus_cu(corpus)

    m = Manifest.load(corpus, required=True)
    m.save()

    assert [p.parent.name for p in find_shards(corpus)] == ["vivos"]
    # Bảng gộp cũ được đổi tên chứ không xoá: mất dữ liệu là không lấy lại được.
    assert not (corpus / MANIFEST_NAME).exists()
    assert (corpus / SUPERSEDED_NAME).exists()
    # Và nó không được nạp lại nữa — nếu không, bản ghi đã loại sẽ sống lại.
    assert find_manifest(corpus) is None
    assert len(Manifest.load(corpus, required=True)) == len(m)


def test_migrate_moves_an_old_corpus_into_per_dataset_folders(tmp_path):
    corpus = tmp_path / "corpus"
    _corpus_cu(corpus)

    m = Manifest.load(corpus, required=True)
    kq = m.migrate_layout()
    assert kq["moved"] == len(m) and kq["missing"] == 0
    m.save()

    for rec in m:
        assert rec.path.startswith("vivos/real/")
        assert m.abs_path(rec).exists()
    # Cây cũ không còn file nào sót lại.
    assert not list(corpus.glob("real/**/*.wav"))

    lan_hai = m.migrate_layout()
    assert lan_hai["moved"] == 0 and lan_hai["kept"] == len(m), "chạy lại phải không xáo gì"
