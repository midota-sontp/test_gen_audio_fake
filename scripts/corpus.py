#!/usr/bin/env python3
"""Một cửa cho mọi việc. Thay cho 9 service docker-compose gần như giống hệt nhau.

    corpus real      dựng phần REAL từ 3 nguồn
    corpus fake      dựng phần FAKE, số lượng bằng đúng REAL
    corpus export    bung shard thành cây dataset (file WAV rời + metadata)
    corpus split     chia train/validation/test speaker-disjoint
    corpus audit     mở lại từng WAV trong shard, đo lại từ đầu
    corpus push      đẩy cây dataset lên Kaggle (một dataset)
    corpus monitor   xem tiến độ trong terminal
    corpus dashboard chạy UI web

Tham số phía sau tên lệnh được chuyển thẳng cho script tương ứng:

    corpus split --dry-run
    corpus real --sources vivos --limit 50
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "src"))

# tên lệnh -> (module, mô tả, tham số mặc định nếu người dùng không đưa)
COMMANDS = {
    "real":      ("build_corpus",   "dựng phần REAL từ 3 nguồn"),
    "fake":      ("build_fake",     "dựng phần FAKE, cân bằng với REAL"),
    "export":    ("export_dataset", "bung shard thành cây dataset"),
    "split":     ("make_splits",    "chia train/validation/test"),
    "audit":     ("audit_shards",   "kiểm tra lại audio đã ghi"),
    "push":      ("push_dataset",   "đẩy dataset lên Kaggle"),
    "monitor":   ("monitor",        "xem tiến độ trong terminal"),
    "dashboard": ("serve_monitor",  "chạy UI web"),
}


def usage(code: int = 0) -> int:
    print(__doc__.strip())
    print("\nLệnh:")
    for name, (_, desc) in COMMANDS.items():
        print(f"  {name:<11} {desc}")
    print("\nThêm --help sau tên lệnh để xem tham số của lệnh đó.")
    return code


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help", "help"):
        return usage()
    name, rest = argv[0], argv[1:]
    if name not in COMMANDS:
        print(f"Lệnh không tồn tại: {name!r}\n", file=sys.stderr)
        return usage(2)

    module_name = COMMANDS[name][0]
    module = __import__(module_name)
    sys.argv = [f"corpus {name}"] + rest
    return module.main() or 0


if __name__ == "__main__":
    raise SystemExit(main())
