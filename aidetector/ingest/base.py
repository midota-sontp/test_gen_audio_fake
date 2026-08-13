"""Tầng ingest — đưa MỌI nguồn dữ liệu về chuẩn corpus.

Thêm một nguồn mới = thêm một file trong `aidetector/ingest/` với một lớp con của
`SourceAdapter` được đánh dấu `@register`. `probe()` cho phép CLI tự nhận diện
loại dataset khi người dùng chỉ đưa vào một thư mục.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np

from ..corpus.spec import AUDIO_EXTENSIONS
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


#: Số tầng thư mục con được dò khi thư mục gốc không khớp adapter chuyên biệt nào.
#: Cần vì Kaggle/Zenodo hay bọc thêm vài tầng: <slug>/archive/vivos/train/waves/...
#: Dò sâu vẫn rẻ nhờ rào "bỏ qua thư mục có quá nhiều con" bên dưới.
MAX_PROBE_DEPTH = 4


def _candidate_roots(root: Path, max_depth: int) -> list[Path]:
    """`root` và các thư mục con của nó, nông trước sâu sau."""
    candidates = [root]
    frontier = [root]
    for _ in range(max_depth):
        nxt: list[Path] = []
        for parent in frontier:
            try:
                children = sorted(
                    p for p in parent.iterdir() if p.is_dir() and not p.name.startswith(".")
                )
            except (OSError, PermissionError):
                continue
            # Thư mục chứa quá nhiều thư mục con thường là kho speaker/clip, không
            # phải lớp bọc — chui vào đó chỉ tốn thời gian.
            if len(children) > 24:
                continue
            nxt.extend(children)
        candidates.extend(nxt)
        frontier = nxt
    return candidates


def _best_adapter_at(root: Path) -> tuple[float, type[SourceAdapter]] | None:
    scores = sorted(
        ((cls.probe(root), cls) for cls in _REGISTRY.values()),
        key=lambda pair: (pair[0], pair[1].name),
        reverse=True,
    )
    return scores[0] if scores and scores[0][0] > 0 else None


_ARCHIVE_SUFFIXES = {".zip", ".tar", ".gz", ".tgz", ".bz2", ".xz", ".7z", ".rar"}
_TABULAR_SUFFIXES = {".parquet", ".arrow", ".jsonl", ".json", ".csv", ".tsv"}


def describe_directory(root: Path, max_files: int = 20_000) -> str:
    """Mô tả thứ có thật trong thư mục, để lỗi "không nhận diện được" còn hành động được.

    Không có dòng này thì người dùng chỉ biết "hỏng" mà không biết vì sao: thư mục
    rỗng? sai đường dẫn? dataset còn nằm trong file nén? định dạng lạ?
    """
    from collections import Counter

    suffixes: Counter[str] = Counter()
    samples: list[str] = []
    total = 0
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]
        for name in filenames:
            if name.startswith("."):
                continue
            total += 1
            suffixes[Path(name).suffix.lower() or "(không đuôi)"] += 1
            if len(samples) < 5:
                samples.append(str(Path(dirpath, name).relative_to(root)))
            if total >= max_files:
                break
        if total >= max_files:
            break

    if total == 0:
        return "  Thư mục rỗng (không có file nào) — kiểm tra lại đường dẫn."

    lines = [f"  Có {total} file nhưng không file audio nào được nhận ra."]
    lines.append("  Đuôi file gặp phải: "
                 + ", ".join(f"{ext}×{n}" for ext, n in suffixes.most_common(8)))
    lines.append("  Ví dụ: " + ", ".join(samples))

    present = set(suffixes)
    if present & _ARCHIVE_SUFFIXES:
        lines.append("  → Dataset có vẻ vẫn nằm trong file nén; giải nén ra thư mục rồi ingest lại.")
    if present & _TABULAR_SUFFIXES:
        lines.append("  → Trông như dataset dạng bảng (parquet/arrow của HuggingFace). "
                     "Dùng `ingest --hf <repo_id>` thay vì đọc theo thư mục.")
    known = ", ".join(sorted(AUDIO_EXTENSIONS))
    lines.append(f"  → Các đuôi audio được hỗ trợ: {known}")
    return "\n".join(lines)


def detect_adapter(
    root: Path, max_depth: int = MAX_PROBE_DEPTH
) -> tuple[type[SourceAdapter], float, Path]:
    """Chọn adapter khớp nhất, dò cả các thư mục con.

    Trả về `(adapter, điểm, thư mục thực sự dùng)` — thư mục trả về có thể nằm sâu
    hơn `root` khi dataset bị bọc thêm tầng.
    """
    best: tuple[float, type[SourceAdapter], Path] | None = None
    for candidate in _candidate_roots(root, max_depth):
        found = _best_adapter_at(candidate)
        if found is None:
            continue
        score, cls = found
        # Nông hơn thì thắng khi điểm bằng nhau: ưu tiên bao trọn dataset.
        depth = len(candidate.relative_to(root).parts)
        if best is None or (score, -depth) > (best[0], -len(best[2].relative_to(root).parts)):
            best = (score, cls, candidate)

    if best is None:
        raise ValueError(
            f"Không nhận diện được dataset tại {root} (đã dò tới {max_depth} tầng con).\n"
            + describe_directory(root)
            + f"\nHoặc chỉ định thủ công: --adapter <{'|'.join(sorted(_REGISTRY))}>"
        )

    score, cls, effective = best
    where = "" if effective == root else f" tại {effective.relative_to(root)}/"
    log.info("Nhận diện dataset: %s (điểm %.2f)%s", cls.name, score, where)
    if effective != root:
        log.info("Bỏ qua %d tầng thư mục bọc ngoài — dùng %s",
                 len(effective.relative_to(root).parts), effective)
    return cls, score, effective
