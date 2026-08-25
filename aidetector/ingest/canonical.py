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


@register
class CanonicalAdapter(SourceAdapter):
    name = "canonical"
    description = "Cây chuẩn của aidetector (real/<nguồn>/<speaker>/*.wav + metadata.csv)"

    @classmethod
    def probe(cls, root: Path) -> float:
        real = root / "real"
        if not real.is_dir():
            return 0.0
        # Đúng ba tầng real/<nguồn>/<speaker>/<file> mới là cây này; hai tầng thì
        # `FolderAdapter` xử lý hợp hơn, đừng giành.
        sau = [p for p in real.glob("*/*/*") if p.suffix.lower() in AUDIO_EXTENSIONS]
        if not sau:
            return 0.0
        return 0.95 if _metadata(root) else 0.7

    def count_hint(self, root: Path) -> int | None:
        return sum(1 for p in (root / "real").glob("*/*/*")
                   if p.suffix.lower() in AUDIO_EXTENSIONS) or None

    @staticmethod
    def _transcripts(root: Path) -> dict[str, str]:
        """Transcript theo đường dẫn TƯƠNG ĐỐI — khoá duy nhất không đụng nhau.

        Tên file trong cây này chỉ là số thứ tự trong thư mục, nên `0001` xuất hiện ở
        mọi speaker; khoá theo stem như các adapter khác là trộn lẫn transcript giữa
        các giọng.
        """
        path = _metadata(root)
        if path is None:
            return {}
        table: dict[str, str] = {}
        with path.open(newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                if (row.get("path") or "").startswith("real/") and row.get("text"):
                    table[row["path"]] = row["text"]
        log.info("Đọc %d transcript từ %s", len(table), path.name)
        return table

    def iter_items(self, root: Path) -> Iterator[SourceItem]:
        texts = self._transcripts(root)
        real = root / "real"
        for wav in sorted(p for p in real.glob("*/*/*")
                          if p.suffix.lower() in AUDIO_EXTENSIONS):
            rel = wav.relative_to(root).as_posix()
            yield SourceItem(
                # `key` là đường dẫn tương đối nên utt_id ổn định giữa các lần nhập:
                # cùng cây vào lại cho cùng id, và ingest vẫn idempotent.
                key=rel,
                audio_path=wav,
                speaker=wav.parent.name,
                text=texts.get(rel, ""),
            )
