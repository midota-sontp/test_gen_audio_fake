"""Đồng bộ shard lên Kaggle Dataset.

GIỚI HẠN CẦN BIẾT: Kaggle không có upload delta. Mỗi `datasets version` đẩy lại
TOÀN BỘ thư mục và mất vài phút để tạo version. Đẩy thật sự mỗi 10 audio là bất
khả thi. Cách làm ở đây:

  * checkpoint 10 audio  -> ghi xuống đĩa (luôn có), cập nhật progress.json
  * dataset "index"      -> vài KB, đẩy được thường xuyên (mặc định >= 60s/lần)
  * dataset theo shard   -> mỗi shard là một dataset riêng `<slug>-pNNNN`, nên
                            mỗi lần đẩy chỉ tốn đúng shard đang mở, không phải
                            toàn bộ corpus đã tích luỹ.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path

LICENSE = "other"      # nguồn gốc mỗi dataset có license riêng, xem README
CMD_TIMEOUT = 3600     # kaggle CLI treo (mạng chết, hỏi credential) không được kéo job treo theo


class KaggleError(RuntimeError):
    pass


def _run(args: list[str], timeout: float = CMD_TIMEOUT) -> str:
    try:
        proc = subprocess.run(args, capture_output=True, text=True,
                              stdin=subprocess.DEVNULL, timeout=timeout)
    except subprocess.TimeoutExpired:
        raise KaggleError(f"quá {timeout:.0f}s không phản hồi: {' '.join(args)}")
    out = (proc.stdout or "") + (proc.stderr or "")
    if proc.returncode != 0:
        raise KaggleError(" ".join(args) + "\n" + out.strip()[:1500])
    return out


def _link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)          # hardlink: không tốn thêm dung lượng
    except OSError:
        shutil.copy2(src, dst)


class KaggleSync:
    def __init__(self, owner: str, slug_base: str, stage_dir: Path,
                 public: bool = False, enabled: bool = True):
        self.owner = owner
        self.slug_base = slug_base
        self.stage = Path(stage_dir)
        self.stage.mkdir(parents=True, exist_ok=True)
        self.public = public
        self.enabled = enabled
        self._created: set[str] = set()

    # ---------- hạ tầng ----------
    def check(self) -> None:
        if not self.enabled:
            return
        if not shutil.which("kaggle"):
            raise KaggleError("Không thấy lệnh `kaggle`. Cài `pip install kaggle`.")
        if not (os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY")) \
                and not Path("~/.kaggle/kaggle.json").expanduser().exists():
            raise KaggleError("Thiếu credential Kaggle: đặt KAGGLE_USERNAME/KAGGLE_KEY "
                              "hoặc mount ~/.kaggle/kaggle.json.")

    def _dataset_exists(self, slug: str) -> bool:
        try:
            out = _run(["kaggle", "datasets", "list", "-m", "-s", slug], timeout=120)
        except KaggleError:
            return False
        return f"{self.owner}/{slug}" in out

    def _write_metadata(self, d: Path, slug: str, title: str) -> None:
        (d / "dataset-metadata.json").write_text(json.dumps({
            "title": title[:50],
            "id": f"{self.owner}/{slug}",
            "licenses": [{"name": LICENSE}],
        }, ensure_ascii=False, indent=2), encoding="utf-8")

    def _push_dir(self, d: Path, slug: str, title: str, message: str) -> str:
        self._write_metadata(d, slug, title)
        if slug in self._created or self._dataset_exists(slug):
            self._created.add(slug)
            _run(["kaggle", "datasets", "version", "-p", str(d), "-m", message,
                  "-r", "skip", "-q"])
            return "version"
        _run(["kaggle", "datasets", "create", "-p", str(d), "-q"]
             + ([] if self.public else ["-u"]))
        self._created.add(slug)
        return "create"

    # ---------- nội dung ----------
    def push_index(self, progress_path: Path, manifest: dict, message: str) -> str:
        """Đẩy file tiến độ (vài KB) — đây là thứ đồng bộ được với nhịp dày."""
        if not self.enabled:
            return "disabled"
        d = self.stage / "index"
        d.mkdir(parents=True, exist_ok=True)
        if progress_path.exists():
            shutil.copy2(progress_path, d / "progress.json")
        (d / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        slug = f"{self.slug_base}-index"
        return self._push_dir(d, slug, f"{self.slug_base} index", message)

    def push_shard(self, shard: str, tar_path: Path, meta_path: Path,
                   message: str) -> str:
        """Mỗi shard một dataset riêng -> lần đẩy nào cũng chỉ tốn 1 shard."""
        if not self.enabled:
            return "disabled"
        d = self.stage / shard
        d.mkdir(parents=True, exist_ok=True)
        _link_or_copy(tar_path, d / tar_path.name)
        _link_or_copy(meta_path, d / meta_path.name)
        slug = f"{self.slug_base}-{shard.split('_')[-1]}"
        return self._push_dir(d, slug, f"{self.slug_base} {shard}", message)


class PushThrottle:
    """Chặn gọi Kaggle quá dày; shard đang mở đẩy thưa hơn file index."""

    def __init__(self, index_every: float, shard_every: float):
        self.index_every = index_every
        self.shard_every = shard_every
        self._last = {"index": 0.0, "shard": 0.0}

    def due(self, kind: str, force: bool = False) -> bool:
        every = self.index_every if kind == "index" else self.shard_every
        if every <= 0 and not force:
            return False
        if force or time.time() - self._last[kind] >= every:
            self._last[kind] = time.time()
            return True
        return False
