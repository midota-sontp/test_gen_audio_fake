"""Tầng ingest — đưa MỌI nguồn dữ liệu về chuẩn corpus.

Thêm một nguồn mới = thêm một file trong `aidetector/ingest/` với một lớp con của
`SourceAdapter` được đánh dấu `@register`. `probe()` cho phép CLI tự nhận diện
loại dataset khi người dùng chỉ đưa vào một thư mục.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np

from ..utils import get_logger

log = get_logger("aidetector.ingest")


@dataclass
class SourceItem:
    """Một utterance thô do adapter trả về (chưa chuẩn hoá)."""

    key: str                                   # id duy nhất trong nguồn (thường là đường dẫn tương đối)
    audio_path: Path | None = None             # dùng path HOẶC audio in-memory
    audio: np.ndarray | None = None
    sample_rate: int | None = None             # bắt buộc nếu dùng `audio`
    speaker: str = ""
    text: str = ""
    language: str = "vi"
    split_hint: str = ""                       # nguồn đã chia sẵn train/test thì ghi vào đây
    meta: dict = field(default_factory=dict)


class SourceAdapter:
    """Lớp cơ sở cho mọi adapter dataset."""

    name: str = "base"
    description: str = ""
    #: nhãn mà adapter này sinh ra — hầu hết là "real"
    label: str = "real"

    @classmethod
    def probe(cls, root: Path) -> float:
        """Độ tự tin (0..1) rằng `root` là dataset loại này."""
        return 0.0

    def iter_items(self, root: Path) -> Iterator[SourceItem]:
        raise NotImplementedError

    def count_hint(self, root: Path) -> int | None:
        """Số lượng ước tính, chỉ để hiển thị tiến độ."""
        return None


_REGISTRY: dict[str, type[SourceAdapter]] = {}


def register(cls: type[SourceAdapter]) -> type[SourceAdapter]:
    if cls.name in _REGISTRY:
        raise ValueError(f"Adapter trùng tên: {cls.name}")
    _REGISTRY[cls.name] = cls
    return cls


def get_adapter(name: str) -> type[SourceAdapter]:
    if name not in _REGISTRY:
        raise KeyError(f"Không có adapter {name!r}. Hiện có: {', '.join(sorted(_REGISTRY))}")
    return _REGISTRY[name]


def available_adapters() -> dict[str, type[SourceAdapter]]:
    return dict(_REGISTRY)


def detect_adapter(root: Path) -> tuple[type[SourceAdapter], float]:
    """Chọn adapter khớp nhất với thư mục `root`."""
    scores = sorted(
        ((cls.probe(root), cls) for cls in _REGISTRY.values()),
        key=lambda pair: pair[0],
        reverse=True,
    )
    if not scores or scores[0][0] <= 0:
        raise ValueError(
            f"Không nhận diện được dataset tại {root}. "
            f"Chỉ định thủ công bằng --adapter <{'|'.join(sorted(_REGISTRY))}>"
        )
    score, cls = scores[0]
    others = ", ".join(f"{c.name}={s:.2f}" for s, c in scores[1:4])
    log.info("Nhận diện dataset: %s (điểm %.2f)%s", cls.name, score, f" · khác: {others}" if others else "")
    return cls, score
