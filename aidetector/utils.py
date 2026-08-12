"""Tiện ích dùng chung: log, seed, device, thời gian, hash ổn định."""

from __future__ import annotations

import hashlib
import logging
import os
import random
import sys
import time
from contextlib import contextmanager
from pathlib import Path

_LEVEL_COLORS = {
    "DEBUG": "\033[38;5;244m",
    "INFO": "\033[38;5;39m",
    "WARNING": "\033[38;5;214m",
    "ERROR": "\033[38;5;196m",
    "CRITICAL": "\033[48;5;196;38;5;231m",
}
_RESET = "\033[0m"


class _Formatter(logging.Formatter):
    def __init__(self, color: bool) -> None:
        super().__init__("%(asctime)s %(levelname)-7s %(name)s | %(message)s", "%H:%M:%S")
        self.color = color

    def format(self, record: logging.LogRecord) -> str:  # noqa: A003
        text = super().format(record)
        if not self.color:
            return text
        return f"{_LEVEL_COLORS.get(record.levelname, '')}{text}{_RESET}"


def setup_logging(level: str = "INFO") -> None:
    """Cấu hình logger gốc một lần cho toàn CLI."""
    root = logging.getLogger()
    if root.handlers:
        root.setLevel(level.upper())
        return
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(_Formatter(color=sys.stderr.isatty()))
    root.addHandler(handler)
    root.setLevel(level.upper())
    # Bớt ồn từ thư viện bên thứ ba.
    for noisy in ("urllib3", "filelock", "huggingface_hub", "numba", "matplotlib"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def set_seed(seed: int) -> None:
    """Cố định seed cho random / numpy / torch (nếu có)."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:  # pragma: no cover
        pass
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:  # pragma: no cover
        pass


def resolve_device(spec: str = "auto") -> str:
    """`auto` -> cuda > mps > cpu. Các giá trị khác được trả nguyên vẹn."""
    if spec and spec != "auto":
        return spec
    try:
        import torch
    except ImportError:  # pragma: no cover
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def stable_id(*parts: object, length: int = 12) -> str:
    """Hash ngắn, ổn định giữa các lần chạy (khác với hash() của Python)."""
    raw = "\x1f".join(str(p) for p in parts).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:length]


def stable_rand(*parts: object) -> random.Random:
    """RNG tất định suy ra từ khoá — cùng khoá luôn cho cùng chuỗi số."""
    return random.Random(int(hashlib.sha1("\x1f".join(str(p) for p in parts).encode()).hexdigest()[:16], 16))


def slugify(text: str, max_len: int = 48) -> str:
    keep = [c if (c.isalnum() or c in "-_.") else "-" for c in str(text)]
    out = "".join(keep).strip("-").lower()
    while "--" in out:
        out = out.replace("--", "-")
    return (out or "unknown")[:max_len]


def human_time(seconds: float) -> str:
    seconds = int(max(0, seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


@contextmanager
def timed(label: str, logger: logging.Logger | None = None):
    log = logger or get_logger("aidetector")
    start = time.time()
    log.info("▶ %s", label)
    try:
        yield
    finally:
        log.info("✔ %s — %s", label, human_time(time.time() - start))


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def progress(iterable, total: int | None = None, label: str = "", every: int = 50):
    """Thanh tiến độ tối giản, không cần phụ thuộc ngoài (tqdm là tuỳ chọn)."""
    try:
        from tqdm import tqdm

        return tqdm(iterable, total=total, desc=label, unit="it", dynamic_ncols=True)
    except ImportError:
        pass

    def _gen():
        log = get_logger("aidetector.progress")
        start = time.time()
        n = total
        if n is None:
            try:
                n = len(iterable)  # type: ignore[arg-type]
            except TypeError:
                n = None
        for i, item in enumerate(iterable, 1):
            yield item
            if i % every == 0 or i == n:
                done = time.time() - start
                eta = (done / i) * (n - i) if n else 0
                pct = f"{100 * i / n:5.1f}%" if n else ""
                log.info("%s %s (%d/%s) ETA %s", label, pct, i, n or "?", human_time(eta))

    return _gen()
