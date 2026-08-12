"""Mozilla Common Voice (bản tải về dạng thư mục: clips/ + *.tsv)."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterator

from ..utils import get_logger
from .base import SourceAdapter, SourceItem, register

log = get_logger("aidetector.ingest.common_voice")

# Thứ tự ưu tiên: validated là tập đã được cộng đồng duyệt.
_TSV_PRIORITY = ("validated.tsv", "train.tsv", "other.tsv", "dev.tsv", "test.tsv")


@register
class CommonVoiceAdapter(SourceAdapter):
    name = "common_voice"
    description = "Mozilla Common Voice (clips/ + validated.tsv)"

    @classmethod
    def probe(cls, root: Path) -> float:
        score = 0.0
        if (root / "clips").is_dir():
            score += 0.5
        if any((root / name).exists() for name in _TSV_PRIORITY):
            score += 0.4
        return min(score, 1.0)

    @staticmethod
    def _pick_tsv(root: Path) -> Path | None:
        for name in _TSV_PRIORITY:
            if (root / name).exists():
                return root / name
        found = sorted(root.glob("*.tsv"))
        return found[0] if found else None

    def iter_items(self, root: Path) -> Iterator[SourceItem]:
        tsv = self._pick_tsv(root)
        if tsv is None:
            log.warning("Không thấy file .tsv nào trong %s", root)
            return
        clips = root / "clips"
        log.info("Common Voice: dùng %s", tsv.name)
        with tsv.open(newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh, delimiter="\t"):
                filename = row.get("path") or row.get("filename") or ""
                if not filename:
                    continue
                audio = clips / filename
                if not audio.exists():          # một số bản đóng gói đã bỏ đuôi mp3
                    alt = list(clips.glob(f"{Path(filename).stem}.*"))
                    if not alt:
                        continue
                    audio = alt[0]
                yield SourceItem(
                    key=filename,
                    audio_path=audio,
                    # client_id là hash ẩn danh của người nói — đủ để chia speaker-disjoint.
                    speaker=(row.get("client_id") or "")[:16] or "cv-unknown",
                    text=(row.get("sentence") or "").strip(),
                    meta={"accent": row.get("accents", ""), "gender": row.get("gender", "")},
                )
