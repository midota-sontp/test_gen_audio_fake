"""Đọc/ghi `corpus/manifest.csv` + các thao tác tra cứu, thống kê, ghi audio."""

from __future__ import annotations

import csv
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np

from ..utils import ensure_dir, get_logger
from .schema import COLUMNS, LABEL_FAKE, LABEL_REAL, Record, relative_audio_path
from .spec import AudioSpec, DEFAULT_SPEC, save_audio

log = get_logger("aidetector.corpus.manifest")

MANIFEST_NAME = "manifest.csv"


class Manifest:
    """Bảng bản ghi corpus, giữ trong bộ nhớ, ghi ra CSV nguyên tử.

    Khoá chính là `utt_id`: thêm lại cùng id sẽ ghi đè chứ không nhân bản, nên
    mọi stage đều chạy lại được mà không sinh rác.
    """

    def __init__(self, root: str | Path, records: Iterable[Record] = ()) -> None:
        self.root = Path(root)
        self._records: dict[str, Record] = {r.utt_id: r for r in records}

    # -------------------------------------------------------------- vào/ra đĩa
    @property
    def csv_path(self) -> Path:
        return self.root / MANIFEST_NAME

    @classmethod
    def load(cls, root: str | Path, required: bool = False) -> "Manifest":
        root = Path(root)
        path = root / MANIFEST_NAME
        if not path.exists():
            if required:
                raise FileNotFoundError(
                    f"Chưa có {path}. Hãy chạy `python -m aidetector ingest ...` trước."
                )
            return cls(root)
        with path.open(newline="", encoding="utf-8") as fh:
            records = [Record.from_row(row) for row in csv.DictReader(fh)]
        log.debug("Đã nạp %d bản ghi từ %s", len(records), path)
        return cls(root, records)

    def save(self) -> Path:
        ensure_dir(self.root)
        tmp = self.csv_path.with_suffix(".csv.tmp")
        with tmp.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(COLUMNS))
            writer.writeheader()
            for rec in self.sorted():
                writer.writerow(rec.to_row())
        os.replace(tmp, self.csv_path)
        log.info("Đã ghi %s (%d bản ghi)", self.csv_path, len(self._records))
        return self.csv_path

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
    def write_audio(
        self, rec: Record, audio: np.ndarray, spec: AudioSpec = DEFAULT_SPEC
    ) -> Record:
        """Ghi mảng audio vào đúng vị trí chuẩn, cập nhật metadata rồi thêm vào manifest."""
        rec.path = relative_audio_path(rec)
        rec.duration = round(len(audio) / spec.sample_rate, 3)
        rec.sample_rate = spec.sample_rate
        rec.channels = spec.channels
        save_audio(self.root / rec.path, audio, spec)
        self.add(rec)
        return rec

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
        if s["by_generator"]:
            lines.append("  Generator : " + ", ".join(f"{k}={v}" for k, v in sorted(s["by_generator"].items())))
        for split, counts in s["by_split"].items():
            lines.append(
                f"  {split:<12}: real={counts.get(LABEL_REAL, 0)} fake={counts.get(LABEL_FAKE, 0)}"
            )
        lines.append(f"  Speaker real: {s['speakers_real']}")
        return "\n".join(lines)
