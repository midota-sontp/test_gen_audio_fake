"""Common Voice Corpus 20 (tiếng Việt) — mirror hataphu/common-voice-corpus-20.

Lưu ý: mirror này CHỈ có (audio, sentence), không có client_id. Theo quyết định của
user, mỗi clip được coi là một speaker riêng và đánh dấu speaker_known = False.
File mp3 gốc là 32 kHz nên bắt buộc resample về 16 kHz.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

from . import _parquet
from .base import RawItem, download, hf_url

REPO = "hataphu/common-voice-corpus-20"
FILES = {
    "train": "data/train-00000-of-00001.parquet",
    "validation": "data/validation-00000-of-00001.parquet",
    "test": "data/test-00000-of-00001.parquet",
}


class CommonVoiceSource:
    name = "common_voice"
    label = 0
    generator = None

    def __init__(self, cache_dir: Path, hf_token: str | None = None):
        self.cache = Path(cache_dir) / self.name
        self.token = hf_token

    def prepare(self) -> None:
        for split, rel in FILES.items():
            download(hf_url(REPO, rel), self.cache / Path(rel).name,
                     self.token, desc=f"common_voice/{split}")

    def units(self) -> list[str]:
        return list(FILES)

    def expected_rows(self) -> int | None:
        try:
            return sum(_parquet.num_rows(self.cache / Path(r).name) for r in FILES.values())
        except Exception:
            return None

    def iter_unit(self, unit: str, start_row: int) -> Iterator[tuple[int, RawItem]]:
        path = self.cache / Path(FILES[unit]).name
        for row_idx, row in _parquet.iter_rows(path, ["audio", "sentence"], start_row):
            audio = row["audio"]
            stem = Path(audio["path"] or f"{unit}_{row_idx}").stem
            # common_voice_vi_23901119 -> 23901119 (tránh lặp tiền tố trong id)
            clip = stem[len("common_voice_vi_"):] if stem.startswith("common_voice_vi_") else stem
            yield row_idx, RawItem(
                item_id=f"common_voice_{clip}",
                source=self.name,
                speaker_id=f"cv20_unknown_{clip}",
                speaker_known=False,
                recording_id=f"cv20_{clip}",
                original_split=unit,
                audio_bytes=audio["bytes"],
                extra={"text": row["sentence"], "clip_id": stem},
            )
