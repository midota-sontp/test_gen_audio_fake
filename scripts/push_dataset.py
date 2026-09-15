#!/usr/bin/env python3
"""Đẩy cây dataset cuối lên Kaggle thành MỘT dataset duy nhất.

    vietnamese-audio-deepfake-demo/
    ├── audio/real/{common_voice,vietmed,vivos}/*.wav
    ├── audio/fake/vieneu_tts/*.wav
    ├── metadata.csv
    ├── metadata.parquet
    └── README.md

Kaggle CLI bỏ qua thư mục con trừ khi có `--dir-mode`. Với `zip`, mỗi thư mục con
được nén thành một file rồi Kaggle bung ra lại phía server — nhờ vậy cây thư mục
`audio/real/...` được giữ nguyên, thay vì phải upload 50 nghìn file WAV rời.

    python scripts/push_dataset.py --dataset /data/dataset/vietnamese-audio-deepfake-demo
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from corpus.kaggle_sync import KaggleError, KaggleSync, _run   # noqa: E402


def human(n: int) -> str:
    return f"{n / 1e9:.2f} GB" if n >= 1e9 else f"{n / 1e6:.1f} MB"


def main() -> int:
    p = argparse.ArgumentParser(description="Đẩy dataset lên Kaggle (một dataset)")
    p.add_argument("--dataset", default="/data/dataset/vietnamese-audio-deepfake-demo",
                   help="thư mục do export_dataset.py sinh ra")
    p.add_argument("--owner", default=os.getenv("KAGGLE_USERNAME"))
    p.add_argument("--slug", default=os.getenv("KAGGLE_DATASET_SLUG",
                                               "vietnamese-audio-deepfake-demo"))
    p.add_argument("--title", default="Vietnamese Audio Deepfake Dataset — Demo v1")
    p.add_argument("--public", action="store_true",
                   help="CÔNG KHAI. Mặc định là private — kiểm quyền redistribution trước.")
    p.add_argument("--dir-mode", default="zip", choices=["zip", "tar"],
                   help="cách gói thư mục con (zip: nén, tar: không nén, nhanh hơn)")
    p.add_argument("--message", default=None, help="ghi chú cho version mới")
    a = p.parse_args()

    d = Path(a.dataset)
    if not a.owner:
        print("LỖI: thiếu --owner / KAGGLE_USERNAME", file=sys.stderr)
        return 2
    if not (d / "metadata.csv").exists():
        print(f"LỖI: {d} chưa có metadata.csv — chạy export_dataset.py trước.",
              file=sys.stderr)
        return 2

    files = [f for f in d.rglob("*")
             if f.is_file() and f.name != "dataset-metadata.json"]
    total = sum(f.stat().st_size for f in files)
    n_wav = sum(1 for f in files if f.suffix == ".wav")
    print(f"Thư mục : {d}")
    print(f"Nội dung: {len(files):,} file ({n_wav:,} WAV), {human(total)}")
    print(f"Đích    : {a.owner}/{a.slug}  [{'CÔNG KHAI' if a.public else 'private'}]")

    k = KaggleSync(a.owner, a.slug, d.parent / ".stage", public=a.public)
    try:
        k.check()
    except KaggleError as e:
        print(f"\nLỖI XÁC THỰC KAGGLE\n{e}\n", file=sys.stderr)
        return 2

    meta = d / "dataset-metadata.json"
    meta.write_text(json.dumps({
        "title": a.title[:50],
        "id": f"{a.owner}/{a.slug}",
        "licenses": [{"name": "other"}],
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    exists = k._dataset_exists(a.slug)
    msg = a.message or f"{len(files):,} file, {human(total)}"
    print(f"\n{'Cập nhật version mới' if exists else 'Tạo dataset mới'} — "
          f"đang gói và upload {human(total)}, sẽ mất khá lâu...\n", flush=True)
    try:
        if exists:
            _run(["kaggle", "datasets", "version", "-p", str(d), "-m", msg,
                  "-r", a.dir_mode, "-q"], timeout=14400)
        else:
            _run(["kaggle", "datasets", "create", "-p", str(d),
                  "-r", a.dir_mode, "-q"] + (["-u"] if a.public else []),
                 timeout=14400)
    except KaggleError as e:
        print(f"\nĐẨY HỎNG\n{e}", file=sys.stderr)
        return 1
    print(f"Xong: https://www.kaggle.com/datasets/{a.owner}/{a.slug}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
