"""VietMed (leduckhai/VietMed) — có sẵn speaker_name, audio_name (recording), gender, accent."""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

from . import _parquet
from .base import RawItem, download, hf_url

REPO = "leduckhai/VietMed"
FILES = {
    "train": "data/train-00000-of-00001.parquet",
    "dev": "data/dev-00000-of-00001.parquet",
    "test": "data/test-00000-of-00001.parquet",
    "cv": "data/cv-00000-of-00001.parquet",
}
COLUMNS = ["audio", "text", "duration", "utterance_id", "speaker_name",
           "seq_name", "audio_name", "role", "gender", "accent",
           "icd10_code", "rec_condition"]


class VietMedSource:
    name = "vietmed"
    label = 0
    generator = None

    def __init__(self, cache_dir: Path, hf_token: str | None = None):
        self.cache = Path(cache_dir) / self.name
        self.token = hf_token

    def prepare(self) -> None:
        for split, rel in FILES.items():
            download(hf_url(REPO, rel), self.cache / Path(rel).name,
                     self.token, desc=f"vietmed/{split}")

    def units(self) -> list[str]:
        return list(FILES)

    def expected_rows(self) -> int | None:
        try:
            return sum(_parquet.num_rows(self.cache / Path(r).name) for r in FILES.values())
        except Exception:
            return None

    def iter_unit(self, unit: str, start_row: int) -> Iterator[tuple[int, RawItem]]:
        path = self.cache / Path(FILES[unit]).name
        for row_idx, row in _parquet.iter_rows(path, COLUMNS, start_row):
            utt = row["utterance_id"] or f"{unit}_{row_idx}"
            yield row_idx, RawItem(
                item_id=f"vietmed_{utt}",
                source=self.name,
                speaker_id=f"vietmed_{row['speaker_name']}",
                speaker_known=True,
                # nhiều utterance cắt ra từ cùng một bản ghi -> gom theo audio_name
                recording_id=f"vietmed_{row['audio_name']}",
                original_split=unit,
                audio_bytes=row["audio"]["bytes"],
                extra={
                    "text": row["text"],
                    "gender": row["gender"],
                    "accent": row["accent"],
                    "role": row["role"],
                    "icd10_code": row["icd10_code"],
                    "rec_condition": row["rec_condition"],
                    "seq_name": row["seq_name"],
                    "source_duration": row["duration"],
                },
            )
