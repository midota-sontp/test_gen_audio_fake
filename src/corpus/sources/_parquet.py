"""Duyệt parquet theo row-group, bỏ qua nguyên nhóm khi resume."""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

import pyarrow.parquet as pq


def iter_rows(path: Path, columns: list[str], start_row: int = 0) -> Iterator[tuple[int, dict]]:
    pf = pq.ParquetFile(path)
    offset = 0
    for g in range(pf.num_row_groups):
        n = pf.metadata.row_group(g).num_rows
        if offset + n <= start_row:
            offset += n
            continue
        table = pf.read_row_group(g, columns=columns)
        cols = {c: table.column(c).to_pylist() for c in columns}
        for i in range(n):
            row_idx = offset + i
            if row_idx < start_row:
                continue
            yield row_idx, {c: cols[c][i] for c in columns}
        offset += n


def num_rows(path: Path) -> int:
    return pq.ParquetFile(path).metadata.num_rows
