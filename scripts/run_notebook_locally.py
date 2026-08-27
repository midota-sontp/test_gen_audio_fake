#!/usr/bin/env python3
"""Chạy trọn hai notebook Kaggle ngay trên máy, trong một môi trường giả lập.

Nhiều lỗi của notebook chỉ lộ ra khi chạy thật: một stage im lặng nuốt lỗi, một ô
đọc phải dữ liệu cũ, một engine tuỳ chọn làm dừng cả phiên. Kiểm tra cú pháp không
bắt được những thứ đó. Script này thay `/kaggle/working` và `/kaggle/input` bằng
thư mục sandbox rồi exec tuần tự mọi ô code trong CÙNG một namespace, đúng như
Jupyter — nên không cần tài khoản Kaggle vẫn thử được toàn bộ.

    python scripts/run_notebook_locally.py <thư-mục-sandbox> <thư-mục-VIVOS> [dataset|train]

Không nêu tên file thì chạy CẢ HAI, dataset trước rồi train — mỗi file một namespace
riêng đúng như hai phiên Kaggle, nhưng dùng chung `/kaggle/working` giả, nên corpus vừa
sinh chính là đầu vào của lượt train (ô A1b nhận ra và không bung đè lên).

Ô cài đặt (pip/apt) bị bỏ qua: môi trường chạy phải cài sẵn thư viện.
Thoát 0 nếu mọi ô chạy xong, khác 0 nếu có ô nào dừng hoặc ném ngoại lệ.
"""

from __future__ import annotations

import json
import shutil
import sys
import traceback
from pathlib import Path

NOTEBOOKS = {
    "dataset": Path("notebooks/aidetector_dataset.ipynb"),
    "train": Path("notebooks/aidetector_train.ipynb"),
}


def build_fake_kaggle(sandbox: Path, vivos: Path) -> tuple[Path, Path]:
    """Dựng /kaggle giả: một dataset rỗng gây nhiễu + một dataset thật bọc thêm tầng."""
    shutil.rmtree(sandbox, ignore_errors=True)
    work, mounts = sandbox / "working", sandbox / "input"
    work.mkdir(parents=True)
    (mounts / "datasets" / "kynthesis").mkdir(parents=True)
    shutil.copytree(vivos, mounts / "vivos-vietnamese" / "vivos")
    return work, mounts


def prepare_cell(source: str, work: Path, mounts: Path) -> tuple[str, str]:
    """Trả (mã sẵn sàng exec, lý do bỏ qua)."""
    if "pip install" in source or "apt-get" in source:
        return "", "ô cài đặt thư viện"
    code = "\n".join(l for l in source.splitlines() if not l.lstrip().startswith("!"))
    code = code.replace("/kaggle/working", str(work)).replace("/kaggle/input", str(mounts))
    return (code, "") if code.strip() else ("", "ô rỗng sau khi bỏ lệnh shell")


def run_notebook(path: Path, work: Path, mounts: Path) -> int:
    """Exec tuần tự mọi ô code của một file trong CÙNG một namespace, như Jupyter."""
    notebook = json.loads(path.read_text(encoding="utf-8"))
    cells = [c for c in notebook["cells"] if c["cell_type"] == "code"]
    print(f"\n{'#' * 78}\n# {path}: {len(notebook['cells'])} ô ({len(cells)} ô code)\n")

    namespace: dict = {"__name__": "__main__"}
    for index, cell in enumerate(cells):
        source = "".join(cell["source"])
        label = next((l for l in source.splitlines() if l.strip()), "")[:70]
        code, skip = prepare_cell(source, work, mounts)
        print(f"{'=' * 78}\n▶ ô {index}: {label}")
        if skip:
            print(f"  ⤼ bỏ qua ({skip})")
            continue
        try:
            exec(compile(code, f"cell{index}", "exec"), namespace)
        except SystemExit as exc:
            print(f"\n✖ NOTEBOOK DỪNG ở ô {index}: {exc}")
            return 1
        except Exception:  # noqa: BLE001 — báo lại nguyên trạng cho người chạy
            print(f"\n✖ NGOẠI LỆ ở ô {index}:")
            traceback.print_exc(file=sys.stdout)
            return 1

        # Config nhúng trong payload vẫn trỏ /kaggle/working — sửa ngay sau khi bung.
        if "_PAYLOAD" in source:
            config = Path(namespace["WORK"]) / "configs" / "kaggle.yaml"
            config.write_text(
                config.read_text(encoding="utf-8").replace("/kaggle/working", str(work)),
                encoding="utf-8",
            )
            print(f"  [harness] đã trỏ {config.name} sang sandbox")
    return 0


def main(argv: list[str]) -> int:
    if len(argv) not in (3, 4):
        print(__doc__)
        return 2
    sandbox, vivos = Path(argv[1]).resolve(), Path(argv[2]).resolve()
    chon = argv[3] if len(argv) == 4 else None
    if chon is not None and chon not in NOTEBOOKS:
        print(f"Tên file phải là một trong {sorted(NOTEBOOKS)}, không phải {chon!r}")
        return 2
    if not vivos.is_dir():
        print(f"Không thấy thư mục VIVOS: {vivos}")
        return 2

    work, mounts = build_fake_kaggle(sandbox, vivos)

    import matplotlib

    matplotlib.use("Agg")
    import IPython.display as display_mod

    display_mod.display = lambda *a, **k: print("   [hiển thị]", *a)

    for ten in ([chon] if chon else ["dataset", "train"]):
        if run_notebook(NOTEBOOKS[ten], work, mounts):
            return 1

    print(f"\n{'=' * 78}\n✔ TOÀN BỘ NOTEBOOK CHẠY XONG, KHÔNG LỖI")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
