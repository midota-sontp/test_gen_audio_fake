"""Adapter cho dataset audio trên HuggingFace Hub.

Không dò theo thư mục — dùng qua CLI: `ingest --hf AILAB-VNUHCM/vivos`.
Tên cột (audio/text/speaker) được tự dò theo các tên thường gặp.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np

from ..utils import get_logger
from .base import SourceAdapter, SourceItem, register

log = get_logger("aidetector.ingest.hf")

_AUDIO_KEYS = ("audio", "wav", "speech", "file", "path")
_TEXT_KEYS = ("sentence", "text", "transcription", "transcript", "normalized_text")
_SPEAKER_KEYS = ("speaker", "speaker_id", "client_id", "spk", "spk_id", "speaker_name")


def _first_key(columns, candidates) -> str | None:
    lowered = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand in lowered:
            return lowered[cand]
    return None


@register
class HuggingFaceAdapter(SourceAdapter):
    name = "hf"
    description = "Dataset audio trên HuggingFace Hub (dùng --hf <repo_id>)"

    def __init__(self, repo_id: str, split: str = "train", config: str | None = None,
                 streaming: bool = False, token: str | None = None) -> None:
        self.repo_id = repo_id
        self.split = split
        self.config = config
        self.streaming = streaming
        self.token = token

    @classmethod
    def probe(cls, root: Path) -> float:
        return 0.0  # chỉ gọi tường minh

    def iter_items(self, root: Path | None = None) -> Iterator[SourceItem]:
        try:
            from datasets import load_dataset
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "Cần cài `datasets` để ingest từ HuggingFace: pip install datasets"
            ) from exc

        log.info("Tải %s (split=%s, config=%s)", self.repo_id, self.split, self.config)
        ds = load_dataset(
            self.repo_id, self.config, split=self.split,
            streaming=self.streaming, token=self.token,
        )
        columns = list(getattr(ds, "column_names", None) or [])
        audio_key = _first_key(columns, _AUDIO_KEYS)
        text_key = _first_key(columns, _TEXT_KEYS)
        speaker_key = _first_key(columns, _SPEAKER_KEYS)
        if audio_key is None:
            raise RuntimeError(f"Không tìm thấy cột audio trong {columns}")
        log.info("Cột: audio=%s text=%s speaker=%s", audio_key, text_key, speaker_key)

        for i, row in enumerate(ds):
            cell = row[audio_key]
            if isinstance(cell, dict) and "array" in cell:
                audio = np.asarray(cell["array"], dtype=np.float32)
                item = SourceItem(
                    key=str(cell.get("path") or f"{self.repo_id}#{i}"),
                    audio=audio,
                    sample_rate=int(cell["sampling_rate"]),
                )
            elif isinstance(cell, (str, Path)):
                item = SourceItem(key=str(cell), audio_path=Path(cell))
            else:
                continue
            item.text = str(row.get(text_key, "") or "") if text_key else ""
            item.speaker = str(row.get(speaker_key, "") or "") if speaker_key else "hf-unknown"
            yield item
