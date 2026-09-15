"""Ghi audio + metadata vào shard tar để đẩy lên Kaggle.

Vì sao là tar chứ không phải file rời: corpus ~30k file WAV nhỏ. Kaggle upload
từng file rất chậm và hay lỗi, còn ổ đĩa máy build thì không phải lúc nào cũng dư.
Mỗi shard là một tar không nén (WAV PCM16 nén gzip gần như vô ích) kèm một file
metadata JSONL cùng tên.

Bền vững khi bị kill: mỗi checkpoint ghi lại tar_offset/meta_offset vào sqlite.
Lần chạy sau cắt (truncate) hai file về đúng offset đó rồi ghi tiếp -> không bao
giờ có member tar dở dang hay dòng JSONL cụt.
"""
from __future__ import annotations

import json
import os
import tarfile
import time
from io import BytesIO
from pathlib import Path

TAR_EOF = b"\0" * 1024        # khối kết thúc tar; lần append sau ghi đè lên


class DistWriter:
    def __init__(self, dist_dir: Path, state, shard_max_bytes: int, prefix: str = "real"):
        self.dist = Path(dist_dir)
        self.audio_dir = self.dist / "audio"
        self.meta_dir = self.dist / "metadata"
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        self.meta_dir.mkdir(parents=True, exist_ok=True)
        self.state = state
        self.max_bytes = shard_max_bytes
        self.prefix = prefix
        self.shard = None
        self.tar_fh = None
        self.tf = None
        self.meta_fh = None
        self.n_files = 0
        self.nbytes = 0
        self._open_or_resume()

    # ---------- vòng đời shard ----------
    def _paths(self, shard: str) -> tuple[Path, Path]:
        return self.audio_dir / f"{shard}.tar", self.meta_dir / f"{shard}.jsonl"

    def _open_or_resume(self) -> None:
        row = self.state.open_shard()
        if row is None:
            self._create_shard()
            return
        self.shard = row["name"]
        tar_path, meta_path = self._paths(self.shard)
        self.n_files = row["n_files"] or 0
        self.nbytes = row["bytes"] or 0
        self._attach(tar_path, meta_path, row["tar_offset"] or 0, row["meta_offset"] or 0)

    def _create_shard(self) -> None:
        idx = self.state.next_shard_index()
        self.shard = f"{self.prefix}_shard_{idx:04d}"
        self.state.begin()
        self.state.create_shard(self.shard)
        self.state.commit()
        tar_path, meta_path = self._paths(self.shard)
        self.n_files = 0
        self.nbytes = 0
        self._attach(tar_path, meta_path, 0, 0)

    def _attach(self, tar_path: Path, meta_path: Path, tar_off: int, meta_off: int) -> None:
        """Mở file ở đúng offset đã checkpoint, cắt bỏ phần ghi dở sau đó."""
        for path, off in ((tar_path, tar_off), (meta_path, meta_off)):
            if not path.exists():
                path.touch()
            with open(path, "r+b") as f:
                f.truncate(off)
        self.tar_fh = open(tar_path, "r+b")
        self.tar_fh.seek(tar_off)
        self.tf = tarfile.open(fileobj=self.tar_fh, mode="w", format=tarfile.PAX_FORMAT)
        self.meta_fh = open(meta_path, "r+b")
        self.meta_fh.seek(meta_off)

    # ---------- ghi ----------
    def add(self, arcname: str, payload: bytes, meta: dict) -> str:
        info = tarfile.TarInfo(arcname)
        info.size = len(payload)
        info.mtime = int(time.time())
        info.mode = 0o644
        self.tf.addfile(info, BytesIO(payload))
        self.meta_fh.write((json.dumps(meta, ensure_ascii=False) + "\n").encode("utf-8"))
        self.n_files += 1
        self.nbytes += len(payload)
        return self.shard

    def flush(self) -> tuple[int, int]:
        """fsync cả hai file, đóng dấu EOF tar rồi lùi con trỏ về lại chỗ cũ.

        Nhờ khối EOF này, shard đang mở vẫn là một tar hợp lệ ở MỌI thời điểm
        checkpoint — đẩy lên Kaggle giữa chừng vẫn đọc được.
        """
        tar_off = self.tar_fh.tell()
        self.tar_fh.write(TAR_EOF)
        self.tar_fh.flush()
        os.fsync(self.tar_fh.fileno())
        self.tar_fh.seek(tar_off)
        self.meta_fh.flush()
        os.fsync(self.meta_fh.fileno())
        return tar_off, self.meta_fh.tell()

    def should_rotate(self) -> bool:
        return self.nbytes >= self.max_bytes

    def close_shard(self) -> str:
        """Đóng shard hiện tại (ghi EOF thật) và mở shard mới."""
        closed = self.shard
        self.tf.close()                      # tarfile tự ghi 2 khối EOF
        self.tar_fh.flush()
        os.fsync(self.tar_fh.fileno())
        tar_off = self.tar_fh.tell()
        self.tar_fh.close()
        self.meta_fh.flush()
        os.fsync(self.meta_fh.fileno())
        meta_off = self.meta_fh.tell()
        self.meta_fh.close()
        from .state import utcnow
        self.state.begin()
        self.state.update_shard(closed, status="closed", n_files=self.n_files,
                                bytes=self.nbytes, tar_offset=tar_off,
                                meta_offset=meta_off, closed_at=utcnow())
        self.state.commit()
        self._create_shard()
        return closed

    def close(self) -> None:
        for fh in (self.tar_fh, self.meta_fh):
            try:
                fh.close()
            except Exception:
                pass
