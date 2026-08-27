"""Adapter cho **cây chuẩn** của chính repo này — đường nhập cho bộ dữ liệu bên ngoài.

    <root>/
      metadata.csv                 (tuỳ chọn; `manifest.csv` cũng đọc được)
      real/<nguồn>/<speaker>/0001.wav
      fake/<engine>/<speaker>/0001.wav

Đây là định dạng mà `pack`/`migrate` sinh ra, nên nó là thứ để trao đổi giữa các máy:
convert một bộ dữ liệu về cây này **một lần**, rồi mọi thứ phía sau không cần biết nó
vốn có cấu trúc gì.

Chỉ nhập phần **real**. Fake luôn do `generate` của chính pipeline này sinh, và nó
idempotent nên nhập lại fake từ ngoài chỉ tạo ra một lớp bản ghi không có `ref_utt_id`
— tức fake không ghép cặp được với real nào, đúng thứ mà cả thiết kế corpus tránh.
Muốn mang nguyên corpus (cả fake, giữ nguyên utt_id) thì dùng `pack`/`unpack`.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterator

from ..corpus.spec import AUDIO_EXTENSIONS
from ..utils import get_logger
from .base import SourceAdapter, SourceItem, register

log = get_logger("aidetector.ingest.canonical")

METADATA_NAMES = ("metadata.csv", "manifest.csv")


def _metadata(root: Path) -> Path | None:
    for name in METADATA_NAMES:
        if (root / name).exists():
            return root / name
    return None


#: Hai cây cùng được nhận, và chúng khác nhau ở chỗ tầng `real/` nằm đâu:
#:
#:   real/<nguồn>/<speaker>/*.wav     cây NHẬP TỪ NGOÀI — đúng thứ `CONVERT` phải dựng
#:   <bộ>/real/<speaker>/*.wav        cây CORPUS của pipeline này (mỗi bộ một thư mục)
#:
#: Nhận cả hai vì cả hai đều xuất hiện thật: cái trước là hợp đồng của bước convert, cái
#: sau là thứ `pack`/`migrate` sinh ra và người ta hay mount lại để nhập vào corpus khác.
_MAU_CAY = ("real/*/*/*", "*/real/*/*")


def _audio_real(root: Path) -> list[Path]:
    """Mọi file audio thuộc lớp real trong cây, không quan tâm cây thuộc dạng nào."""
    ra: list[Path] = []
    for mau in _MAU_CAY:
        ra += [p for p in root.glob(mau) if p.suffix.lower() in AUDIO_EXTENSIONS]
    return sorted(set(ra))


def _cac_metadata(root: Path) -> list[Path]:
    """`metadata.csv` ở gốc (cây nhập) và/hoặc của từng bộ (cây corpus)."""
    ra = []
    goc = _metadata(root)
    if goc is not None:
        ra.append(goc)
    for thu_muc in sorted(p for p in root.iterdir() if p.is_dir()) if root.is_dir() else []:
        con = _metadata(thu_muc)
        if con is not None:
            ra.append(con)
    return ra


@register
class CanonicalAdapter(SourceAdapter):
    name = "canonical"
    description = "Cây chuẩn của aidetector (real/<nguồn>/<speaker>/*.wav + metadata.csv)"

    @classmethod
    def probe(cls, root: Path) -> float:
        # Đúng số tầng mới là cây này (xem `_MAU_CAY`); thiếu một tầng thì
        # `FolderAdapter` xử lý hợp hơn, đừng giành.
        if not _audio_real(root):
            return 0.0
        return 0.95 if _cac_metadata(root) else 0.7

    def count_hint(self, root: Path) -> int | None:
        return len(_audio_real(root)) or None

    @staticmethod
    def _transcripts(root: Path) -> dict[str, str]:
        """Transcript theo đường dẫn TƯƠNG ĐỐI — khoá duy nhất không đụng nhau.

        Tên file trong cây này chỉ là số thứ tự trong thư mục, nên `0001` xuất hiện ở
        mọi speaker; khoá theo stem như các adapter khác là trộn lẫn transcript giữa
        các giọng.
        """
        table: dict[str, str] = {}
        cac = _cac_metadata(root)
        for path in cac:
            with path.open(newline="", encoding="utf-8") as fh:
                for row in csv.DictReader(fh):
                    duong = row.get("path") or ""
                    # Chỉ lấy lớp real, ở cả hai cây: `real/…` (cây nhập) và `<bộ>/real/…`
                    # (cây corpus). Bỏ fake — nhập lại fake là tạo ra bản ghi không có
                    # `ref_utt_id`, đúng thứ cả thiết kế corpus tránh.
                    la_real = duong.startswith("real/") or "/real/" in duong
                    if la_real and row.get("text"):
                        table[duong] = row["text"]
        if cac:
            log.info("Đọc %d transcript từ %d file metadata", len(table), len(cac))
        return table

    def iter_items(self, root: Path) -> Iterator[SourceItem]:
        texts = self._transcripts(root)
        for wav in _audio_real(root):
            rel = wav.relative_to(root).as_posix()
            yield SourceItem(
                # `key` là đường dẫn tương đối nên utt_id ổn định giữa các lần nhập:
                # cùng cây vào lại cho cùng id, và ingest vẫn idempotent.
                key=rel,
                audio_path=wav,
                speaker=wav.parent.name,
                text=texts.get(rel, ""),
            )
