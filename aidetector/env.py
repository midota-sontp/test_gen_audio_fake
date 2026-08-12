"""Nhận diện môi trường chạy (local / Kaggle / Colab) và đường dẫn tương ứng.

Kaggle và Colab có ràng buộc riêng — thư mục input chỉ đọc, thư mục làm việc bị
giới hạn dung lượng, phiên làm việc hết hạn sau vài giờ — nên pipeline cần biết
mình đang ở đâu để chọn đường dẫn và cảnh báo cho đúng.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from .utils import get_logger

log = get_logger("aidetector.env")

LOCAL = "local"
KAGGLE = "kaggle"
COLAB = "colab"


@dataclass(frozen=True)
class Platform:
    name: str
    work_dir: Path            # nơi ghi corpus/features/checkpoints
    input_dir: Path | None    # thư mục dataset chỉ-đọc (Kaggle Datasets)
    tmp_dir: Path
    #: dung lượng khuyến nghị tối đa cho work_dir (GB), None = không giới hạn rõ ràng
    disk_budget_gb: float | None = None
    #: phiên làm việc tự ngắt sau bao nhiêu giờ, None = không giới hạn
    session_hours: float | None = None

    @property
    def is_ephemeral(self) -> bool:
        """True nếu dữ liệu mất khi phiên kết thúc ⇒ cần export ra ngoài."""
        return self.name in (KAGGLE, COLAB)

    def describe(self) -> str:
        parts = [self.name, f"work={self.work_dir}"]
        if self.input_dir:
            parts.append(f"input={self.input_dir}")
        if self.disk_budget_gb:
            parts.append(f"đĩa≈{self.disk_budget_gb:g}GB")
        if self.session_hours:
            parts.append(f"phiên≈{self.session_hours:g}h")
        return " · ".join(parts)


def detect_platform() -> Platform:
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or Path("/kaggle/working").is_dir():
        return Platform(
            name=KAGGLE,
            work_dir=Path("/kaggle/working"),
            input_dir=Path("/kaggle/input"),
            tmp_dir=Path("/kaggle/temp"),
            disk_budget_gb=20.0,
            session_hours=9.0,
        )
    if "COLAB_GPU" in os.environ or Path("/content").is_dir():
        return Platform(
            name=COLAB,
            work_dir=Path("/content"),
            input_dir=None,
            tmp_dir=Path("/tmp"),
            disk_budget_gb=None,
            session_hours=12.0,
        )
    return Platform(name=LOCAL, work_dir=Path.cwd(), input_dir=None, tmp_dir=Path("/tmp"))


def free_space_gb(path: str | Path) -> float:
    path = Path(path)
    while not path.exists() and path != path.parent:
        path = path.parent
    return shutil.disk_usage(path).free / 1024**3


def find_kaggle_datasets(pattern: str = "") -> list[Path]:
    """Liệt kê các dataset đang được mount vào `/kaggle/input`."""
    root = Path("/kaggle/input")
    if not root.is_dir():
        return []
    return sorted(p for p in root.iterdir() if p.is_dir() and (not pattern or pattern in p.name))


def warn_if_constrained(work_dir: str | Path) -> None:
    """Cảnh báo sớm về những giới hạn hay làm hỏng một phiên Kaggle/Colab."""
    platform = detect_platform()
    if platform.name == LOCAL:
        return
    free = free_space_gb(work_dir)
    log.info("Môi trường: %s · còn trống %.1f GB", platform.describe(), free)
    if free < 5:
        log.warning(
            "Chỉ còn %.1f GB trống — corpus + cache đặc trưng dễ làm đầy đĩa. "
            "Giảm số mẫu (--limit / --count) hoặc ghi features vào %s.",
            free, platform.tmp_dir,
        )
    if platform.is_ephemeral:
        log.warning(
            "Dữ liệu trong %s sẽ MẤT khi phiên kết thúc. Nhớ lưu %s ra Kaggle Dataset "
            "hoặc tải về trước khi hết phiên (~%s giờ).",
            platform.work_dir, "corpus/ + checkpoints/", platform.session_hours,
        )
