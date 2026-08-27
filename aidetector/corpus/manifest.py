"""Đọc/ghi `corpus/manifest.csv` + các thao tác tra cứu, thống kê, ghi audio."""

from __future__ import annotations

import csv
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np

from ..utils import ensure_dir, get_logger
from .schema import (
    COLUMNS, LABEL_FAKE, LABEL_REAL, Record, audio_folder, audio_name, shard_name,
)
from .spec import AudioSpec, DEFAULT_SPEC, save_audio

log = get_logger("aidetector.corpus.manifest")

MANIFEST_NAME = "metadata.csv"
#: Tên cũ, vẫn đọc được. Corpus đã đóng gói ở các phiên trước dùng tên này, và bỏ đọc nó
#: nghĩa là mọi corpus.zip đang có trên Kaggle thành rác.
LEGACY_MANIFEST_NAME = "manifest.csv"
#: Manifest GỘP ở gốc corpus là cấu trúc cũ (một bảng cho mọi bộ dữ liệu). Vẫn đọc được,
#: nhưng `save()` chuyển nó sang tên này vì nội dung đã nằm đủ trong từng shard — để lại
#: dưới tên cũ thì mỗi lượt `load` lại dựng ngược cả những bản ghi vừa bị loại.
SUPERSEDED_NAME = "metadata.goc-cu.csv"


def find_manifest(root: Path) -> Path | None:
    """Manifest GỘP ở gốc corpus — cấu trúc cũ. `None` khi corpus đã tách theo bộ."""
    root = Path(root)
    for name in (MANIFEST_NAME, LEGACY_MANIFEST_NAME):
        if (root / name).exists():
            return root / name
    return None


def find_shards(root: Path) -> list[Path]:
    """`metadata.csv` của từng bộ dữ liệu: `<root>/<bộ>/metadata.csv`.

    Chỉ soi một tầng: thư mục con của gốc corpus là thư mục bộ dữ liệu, không sâu hơn.
    Cây cũ (`real/`, `fake/` nằm thẳng ở gốc) không có manifest trong đó nên trả rỗng —
    đúng thứ để `load` biết corpus này chưa tách.
    """
    root = Path(root)
    if not root.is_dir():
        return []
    ra: list[Path] = []
    for thu_muc in sorted(p for p in root.iterdir() if p.is_dir()):
        for name in (MANIFEST_NAME, LEGACY_MANIFEST_NAME):
            if (thu_muc / name).exists():
                ra.append(thu_muc / name)
                break
    return ra


def _doc_csv(path: Path) -> list[Record]:
    with path.open(newline="", encoding="utf-8") as fh:
        return [Record.from_row(row) for row in csv.DictReader(fh)]


def manifest_csv(records: Iterable[Record]) -> str:
    """Nội dung CSV của một nhóm bản ghi — dùng cho cả ghi đĩa và đóng gói zip."""
    import io

    buf = io.StringIO(newline="")
    writer = csv.DictWriter(buf, fieldnames=list(COLUMNS))
    writer.writeheader()
    for rec in records:
        writer.writerow(rec.to_row())
    return buf.getvalue()


class Manifest:
    """Bảng bản ghi corpus, giữ trong bộ nhớ, ghi ra CSV nguyên tử.

    Khoá chính là `utt_id`: thêm lại cùng id sẽ ghi đè chứ không nhân bản, nên
    mọi stage đều chạy lại được mà không sinh rác.

    Trên đĩa, bảng này được **tách theo bộ dữ liệu**: mỗi bộ một thư mục tự chứa
    `<bộ>/real/`, `<bộ>/fake/` và `<bộ>/metadata.csv` của riêng nó. Trong bộ nhớ vẫn là
    MỘT bảng hợp nhất, vì chia tập speaker-disjoint, cân bằng lớp và huấn luyện đều phải
    nhìn toàn bộ dữ liệu cùng lúc.

    Cột `path` tính từ gốc corpus ở cả hai cấu trúc, nên corpus cũ (một manifest gộp ở
    gốc) vẫn đọc và tra cứu được nguyên vẹn; `save()` là lúc nó được tách ra, `migrate`
    là lúc file audio được dời về cây mới.
    """

    def __init__(self, root: str | Path, records: Iterable[Record] = ()) -> None:
        self.root = Path(root)
        self._records: dict[str, Record] = {r.utt_id: r for r in records}
        # Số cuối đã cấp cho mỗi thư mục. Dựng lười vì corpus lớn thì quét toàn bộ chỉ
        # để ghi một file là phí; cấp xong thì số nằm trong `path`, không tính lại.
        self._so_cuoi: dict[str, int] = {}
        #: Bộ nào ĐÃ có file manifest trên đĩa lúc nạp. Cần nhớ để `save()` còn ghi lại
        #: cả bộ vừa mất hết bản ghi (`validate --fix` loại sạch chẳng hạn): bỏ qua nó
        #: là để file cũ nằm lại và lượt nạp sau dựng ngược đúng những bản ghi đã loại.
        self._bo_tren_dia: set[str] = set()
        #: Manifest gộp ở gốc, nếu corpus này còn ở cấu trúc cũ.
        self._goc_cu: Path | None = None

    # -------------------------------------------------------------- vào/ra đĩa
    def shard_path(self, source: str) -> Path:
        """`metadata.csv` của một bộ dữ liệu."""
        return self.root / shard_name(source) / MANIFEST_NAME

    def by_shard(self) -> dict[str, list[Record]]:
        """Bản ghi gom theo bộ dữ liệu, đúng cách chúng được ghi ra đĩa."""
        ra: dict[str, list[Record]] = defaultdict(list)
        for rec in self.sorted():
            ra[shard_name(rec.source)].append(rec)
        # Bộ từng có mặt trên đĩa mà giờ rỗng vẫn phải xuất hiện, để `save` ghi lại một
        # file rỗng thay vì để bản cũ nằm lại.
        for ten in self._bo_tren_dia:
            ra.setdefault(ten, [])
        return dict(ra)

    @classmethod
    def load(cls, root: str | Path, required: bool = False) -> "Manifest":
        root = Path(root)
        shards = find_shards(root)
        goc_cu = find_manifest(root)

        if not shards and goc_cu is None:
            if required:
                raise FileNotFoundError(
                    f"Chưa có {root / '<bộ>' / MANIFEST_NAME}. "
                    f"Hãy chạy `python -m aidetector ingest ...` trước."
                )
            return cls(root)

        records: dict[str, Record] = {}
        for path in shards:
            for rec in _doc_csv(path):
                records[rec.utt_id] = rec
        log.debug("Đã nạp %d bản ghi từ %d bộ", len(records), len(shards))

        if goc_cu is not None:
            # Cấu trúc cũ. Bản ghi đã có trong shard thì SHARD thắng: shard là bản mới,
            # manifest gộp chỉ còn là di sản chờ `save()` chuyển đi.
            them = 0
            for rec in _doc_csv(goc_cu):
                if rec.utt_id not in records:
                    records[rec.utt_id] = rec
                    them += 1
            if them:
                log.info("Nạp thêm %d bản ghi từ manifest gộp cũ %s — lượt `save` tới sẽ"
                         " tách chúng về từng bộ (`migrate` dời file audio)", them, goc_cu)

        m = cls(root, records.values())
        m._bo_tren_dia = {p.parent.name for p in shards}
        m._goc_cu = goc_cu
        return m

    def save(self) -> Path:
        ensure_dir(self.root)
        theo_bo = self.by_shard()
        for ten, recs in sorted(theo_bo.items()):
            dich = ensure_dir(self.root / ten) / MANIFEST_NAME
            tmp = dich.with_suffix(".csv.tmp")
            tmp.write_text(manifest_csv(recs), newline="", encoding="utf-8")
            os.replace(tmp, dich)
        self._bo_tren_dia = set(theo_bo)
        log.info("Đã ghi %d bản ghi vào %d bộ: %s", len(self._records), len(theo_bo),
                 ", ".join(f"{k}={len(v)}" for k, v in sorted(theo_bo.items())))

        # Nội dung manifest gộp cũ giờ đã nằm đủ trong các shard. Đổi tên chứ không xoá:
        # mất dữ liệu vì một lượt ghi hỏng là không lấy lại được, còn file này thì rẻ.
        if self._goc_cu is not None and self._goc_cu.exists():
            giu = self.root / SUPERSEDED_NAME
            os.replace(self._goc_cu, giu)
            log.info("Manifest gộp cũ đã được tách theo bộ — giữ bản gốc ở %s", giu)
            self._goc_cu = None
        return self.root

    # ------------------------------------------------------------------ thao tác
    def __len__(self) -> int:
        return len(self._records)

    def __iter__(self) -> Iterator[Record]:
        return iter(self._records.values())

    def __contains__(self, utt_id: object) -> bool:
        return utt_id in self._records

    def get(self, utt_id: str) -> Record | None:
        return self._records.get(utt_id)

    def add(self, rec: Record) -> None:
        errs = rec.validate()
        if errs:
            raise ValueError(f"Bản ghi {rec.utt_id} không hợp lệ: {'; '.join(errs)}")
        self._records[rec.utt_id] = rec

    def remove(self, utt_id: str) -> None:
        self._records.pop(utt_id, None)

    def sorted(self) -> list[Record]:
        return sorted(self._records.values(), key=lambda r: (r.label, r.source, r.speaker, r.utt_id))

    def filter(self, **criteria) -> list[Record]:
        """`filter(label="real", split="train")` — so khớp bằng ==."""
        return [
            r for r in self._records.values()
            if all(getattr(r, k, None) == v for k, v in criteria.items())
        ]

    def by_split(self, split: str) -> list[Record]:
        return [r for r in self._records.values() if r.split == split]

    @property
    def reals(self) -> list[Record]:
        return [r for r in self._records.values() if r.label == LABEL_REAL]

    @property
    def fakes(self) -> list[Record]:
        return [r for r in self._records.values() if r.label == LABEL_FAKE]

    def speakers(self, label: str | None = None) -> list[str]:
        return sorted({r.speaker for r in self._records.values()
                       if r.speaker and (label is None or r.label == label)})

    def abs_path(self, rec: Record) -> Path:
        return self.root / rec.path

    # ------------------------------------------------- ghi audio đúng chuẩn
    def allocate_path(self, rec: Record) -> str:
        """Cấp đường dẫn cho một bản ghi MỚI: thư mục chuẩn + số thứ tự kế tiếp.

        Số cấp một lần rồi lưu trong cột `path`; lần chạy sau đọc lại chứ không suy ra
        từ thứ tự duyệt. Không tái sử dụng số đã cấp kể cả khi bản ghi bị xoá — số cũ
        trỏ tới file cũ trên đĩa, dùng lại là hai bản ghi cùng một file.
        """
        folder = audio_folder(rec)
        if folder not in self._so_cuoi:
            lon_nhat = 0
            for r in self._records.values():
                if r.path.startswith(folder + "/"):
                    stem = Path(r.path).stem
                    if stem.isdigit():
                        lon_nhat = max(lon_nhat, int(stem))
            self._so_cuoi[folder] = lon_nhat
        self._so_cuoi[folder] += 1
        return f"{folder}/{audio_name(self._so_cuoi[folder])}"

    def write_audio(
        self, rec: Record, audio: np.ndarray, spec: AudioSpec = DEFAULT_SPEC
    ) -> Record:
        """Ghi mảng audio vào đúng vị trí chuẩn, cập nhật metadata rồi thêm vào manifest."""
        # Ghi đè bản ghi đã có (vd `generate --overwrite`) thì giữ nguyên chỗ cũ: cấp số
        # mới là để lại một file mồ côi và làm cây phình sau mỗi lượt chạy lại.
        cu = self._records.get(rec.utt_id)
        rec.path = cu.path if (cu is not None and cu.path) else self.allocate_path(rec)
        rec.duration = round(len(audio) / spec.sample_rate, 3)
        rec.sample_rate = spec.sample_rate
        rec.channels = spec.channels
        save_audio(self.root / rec.path, audio, spec)
        self.add(rec)
        return rec

    def migrate_layout(self, dry_run: bool = False) -> dict:
        """Đưa mọi bản ghi về đúng cấu trúc thư mục hiện hành, giữ nguyên `utt_id`.

        Cột `path` là nguồn sự thật nên corpus cũ vẫn ĐỌC được mà không cần chuyển; hàm
        này để dọn cho đồng nhất một cây duy nhất. Idempotent: bản ghi đã đúng chỗ thì
        giữ nguyên số cũ, chạy lại lần hai không xáo lại gì.
        """
        dung_cho: list[Record] = []
        self._so_cuoi = {}
        for rec in sorted(self._records.values(), key=lambda r: r.utt_id):
            folder = audio_folder(rec)
            p = Path(rec.path)
            if p.parent.as_posix() == folder and p.stem.isdigit():
                self._so_cuoi[folder] = max(self._so_cuoi.get(folder, 0), int(p.stem))
            else:
                dung_cho.append(rec)

        # Manifest chỉ được lưu SAU khi mọi file đã dời xong, và phép cấp số ở trên là
        # tất định (duyệt theo utt_id, `_so_cuoi` dựng từ chính các bản ghi đã đúng chỗ).
        # Nhờ hai điều đó, bị ngắt giữa chừng thì manifest còn nguyên ⇒ chạy lại tính ra
        # ĐÚNG những đường dẫn cũ, và nhánh "nguồn mất nhưng đích đã có" nhận ra phần đã
        # dời để bỏ qua. Lưu dở giữa chừng mới là thứ phá được tính tất định đó.
        chuyen = thieu = tiep_tuc = 0
        bo_lai: set[Path] = set()      # thư mục vừa bị lấy hết file — dọn ở cuối
        for rec in dung_cho:
            cu = self.root / rec.path
            moi = self.allocate_path(rec)
            dich = self.root / moi
            if not cu.exists():
                if dich.exists():
                    tiep_tuc += 1       # lượt trước đã dời file này rồi
                    rec.path = moi
                    continue
                thieu += 1
                rec.path = moi          # file mất thật; prune_missing sẽ dọn nếu cần
                continue
            if not dry_run:
                dich.parent.mkdir(parents=True, exist_ok=True)
                os.replace(cu, dich)
                bo_lai.add(cu.parent)
            rec.path = moi
            chuyen += 1
        if tiep_tuc:
            log.info("Nhận lại %d file đã dời ở lượt bị ngắt trước", tiep_tuc)
        if not dry_run:
            self._don_thu_muc_rong(bo_lai)

        log.info("Chuyển cấu trúc: %d bản ghi đã đúng chỗ · %d chuyển · %d thiếu file",
                 len(self._records) - len(dung_cho), chuyen, thieu)
        return {"kept": len(self._records) - len(dung_cho), "moved": chuyen,
                "resumed": tiep_tuc, "missing": thieu}

    def _don_thu_muc_rong(self, thu_muc: Iterable[Path]) -> int:
        """Xoá những thư mục vừa trở nên RỖNG sau khi dời file, leo dần lên tới gốc.

        Không dọn thì cây cũ nằm lại dưới dạng thư mục trống cạnh các thư mục bộ dữ liệu —
        `ls corpus/` không còn đọc được là "mỗi bộ một thư mục" nữa. Chỉ `rmdir`, không
        `rmtree`: thư mục còn bất cứ thứ gì thì lệnh này thất bại và ta dừng ngay ở đó,
        nên một bản ghi chưa được dời không bao giờ bị xoá theo.
        """
        xoa = 0
        for goc in sorted(thu_muc, key=lambda p: len(p.parts), reverse=True):
            hien = goc
            while hien != self.root and hien.is_relative_to(self.root):
                try:
                    hien.rmdir()
                except OSError:
                    break              # còn file/thư mục con ⇒ dừng, và dừng luôn nhánh
                xoa += 1
                hien = hien.parent
        if xoa:
            log.info("Đã dọn %d thư mục rỗng của cây cũ", xoa)
        return xoa

    def prune_missing(self) -> int:
        """Bỏ các bản ghi trỏ tới file không còn tồn tại."""
        gone = [r.utt_id for r in self._records.values() if not self.abs_path(r).exists()]
        for utt_id in gone:
            del self._records[utt_id]
        if gone:
            log.warning("Đã loại %d bản ghi mất file audio", len(gone))
        return len(gone)

    # ----------------------------------------------------------------- thống kê
    def stats(self) -> dict:
        recs = list(self._records.values())
        by_label = Counter(r.label for r in recs)
        by_generator = Counter(r.generator for r in recs if r.generator)
        # Fake thừa hưởng `source` của utterance real gốc, nên chỉ đếm real ở đây
        # để con số phản ánh đúng "dữ liệu thật đến từ đâu".
        by_source = Counter(r.source for r in recs if r.source and r.label == LABEL_REAL)
        by_split: dict[str, Counter] = defaultdict(Counter)
        for r in recs:
            by_split[r.split or "(chưa chia)"][r.label] += 1
        hours = sum(r.duration for r in recs) / 3600
        return {
            "total": len(recs),
            "hours": round(hours, 2),
            "by_label": dict(by_label),
            "by_source": dict(by_source),
            "by_generator": dict(by_generator),
            "by_split": {k: dict(v) for k, v in sorted(by_split.items())},
            "speakers_real": len(self.speakers(LABEL_REAL)),
            "augmented": sum(1 for r in recs if r.augment),
        }

    def summary(self) -> str:
        s = self.stats()
        lines = [
            f"Corpus: {self.root}",
            f"  Tổng: {s['total']} utt · {s['hours']} giờ · "
            f"real={s['by_label'].get(LABEL_REAL, 0)} fake={s['by_label'].get(LABEL_FAKE, 0)} "
            f"(augment: {s['augmented']})",
        ]
        if s["by_source"]:
            lines.append("  Nguồn real: " + ", ".join(f"{k}={v}" for k, v in sorted(s["by_source"].items())))
        # Corpus tách theo bộ: in ra từng thư mục kèm số bản ghi của nó, vì đó là đơn vị
        # thêm/bỏ/đẩy đi được — cột `by_source` chỉ đếm real nên không thay được chỗ này.
        theo_bo = self.by_shard()
        if theo_bo:
            lines.append("  Bộ dữ liệu: " + ", ".join(
                f"{k}/={len(v)}" for k, v in sorted(theo_bo.items())))
        if s["by_generator"]:
            lines.append("  Generator : " + ", ".join(f"{k}={v}" for k, v in sorted(s["by_generator"].items())))
        for split, counts in s["by_split"].items():
            lines.append(
                f"  {split:<12}: real={counts.get(LABEL_REAL, 0)} fake={counts.get(LABEL_FAKE, 0)}"
            )
        lines.append(f"  Speaker real: {s['speakers_real']}")
        return "\n".join(lines)
