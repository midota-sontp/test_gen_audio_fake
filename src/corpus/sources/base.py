"""Giao diện chung cho mọi nguồn dữ liệu. Thêm nguồn mới = thêm 1 file ở đây."""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Protocol

import requests

CHUNK = 1 << 20


@dataclass
class RawItem:
    item_id: str                 # id ổn định, tái chạy phải ra đúng id này
    source: str
    speaker_id: str
    recording_id: str
    original_split: str
    audio_bytes: bytes
    speaker_known: bool = True
    label: int = 0               # 0 = REAL, 1 = FAKE
    generator: str | None = None # bắt buộc có giá trị khi label = 1
    extra: dict = field(default_factory=dict)


class Source(Protocol):
    name: str
    label: int
    generator: str | None

    def prepare(self) -> None: ...
    def units(self) -> list[str]: ...
    def iter_unit(self, unit: str, start_row: int) -> Iterator[tuple[int, RawItem]]: ...
    def expected_rows(self) -> int | None: ...


def download(url: str, dest: Path, token: str | None = None, desc: str = "") -> Path:
    """Tải file có resume bằng HTTP Range; bỏ qua nếu đã đủ dung lượng."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    headers = {"Authorization": f"Bearer {token}"} if token else {}

    head = requests.head(url, headers=headers, allow_redirects=True, timeout=60)
    total = int(head.headers.get("x-linked-size") or head.headers.get("content-length") or 0)
    if dest.exists() and total and dest.stat().st_size == total:
        return dest

    pos = dest.stat().st_size if dest.exists() else 0
    if pos and total and pos > total:
        dest.unlink()
        pos = 0
    mode = "ab" if pos else "wb"
    if pos:
        headers["Range"] = f"bytes={pos}-"

    with requests.get(url, headers=headers, stream=True, timeout=120) as r:
        if pos and r.status_code == 200:      # server bỏ qua Range -> tải lại từ đầu
            pos, mode = 0, "wb"
        r.raise_for_status()
        last = time.time()
        with open(dest, mode) as f:
            for chunk in r.iter_content(CHUNK):
                f.write(chunk)
                pos += len(chunk)
                if time.time() - last > 5:
                    pct = f"{pos / total * 100:5.1f}%" if total else f"{pos / 1e6:.0f} MB"
                    print(f"  [tải] {desc or dest.name} {pct}", flush=True)
                    last = time.time()
            f.flush()
            os.fsync(f.fileno())
    return dest


def hf_url(repo: str, path: str) -> str:
    return f"https://huggingface.co/datasets/{repo}/resolve/main/{path}"
