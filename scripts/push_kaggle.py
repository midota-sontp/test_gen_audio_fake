#!/usr/bin/env python3
"""Đẩy thủ công shard + file tiến độ lên Kaggle (dùng khi job đã dừng hoặc push lỗi).

  python scripts/push_kaggle.py --out /data/out --owner <user> --slug vi-real-audio-demo-v1
  python scripts/push_kaggle.py --out /data/out ... --only real_shard_0003
  python scripts/push_kaggle.py --out /data/out ... --retry-failed   # chỉ shard chưa đẩy
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from corpus.kaggle_sync import KaggleError, KaggleSync   # noqa: E402
from corpus.state import State, utcnow                   # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description="Đẩy shard corpus lên Kaggle Dataset")
    p.add_argument("--out", default="/data/out")
    p.add_argument("--owner", default=os.getenv("KAGGLE_USERNAME"))
    p.add_argument("--slug", default=os.getenv("KAGGLE_SLUG", "vi-real-audio-demo-v1"))
    p.add_argument("--public", action="store_true")
    p.add_argument("--only", nargs="*", help="chỉ đẩy các shard này")
    p.add_argument("--retry-failed", action="store_true", help="chỉ shard chưa từng đẩy")
    p.add_argument("--index-only", action="store_true")
    a = p.parse_args()
    if not a.owner:
        print("LỖI: thiếu --owner / KAGGLE_USERNAME", file=sys.stderr)
        return 2

    out = Path(a.out)
    st = State(out / "state" / "corpus.sqlite")
    k = KaggleSync(a.owner, a.slug, out / "stage", public=a.public)
    try:
        k.check()
    except KaggleError as e:
        print(f"\nLỖI XÁC THỰC KAGGLE\n{e}\n", file=sys.stderr)
        return 2

    rc = 0
    if not a.index_only:
        for s in st.shards():
            if a.only and s["name"] not in a.only:
                continue
            if a.retry_failed and s["pushed_at"]:
                continue
            tar = out / "dist" / "audio" / f"{s['name']}.tar"
            meta = out / "dist" / "metadata" / f"{s['name']}.jsonl"
            if not tar.exists() or not s["n_files"]:
                print(f"  bỏ qua {s['name']}: chưa có dữ liệu")
                continue
            try:
                action = k.push_shard(s["name"], tar, meta,
                                      f"{s['name']} @ {utcnow()} ({s['n_files']} files)")
                st.begin()
                st.update_shard(s["name"], pushed_at=utcnow(),
                                push_target=f"{a.owner}/{a.slug}-{s['name'].split('_')[-1]}")
                st.commit()
                print(f"  ✓ {action} {s['name']} ({(s['bytes'] or 0) / 1e6:.1f} MB)")
            except KaggleError as e:
                st.rollback()
                print(f"  ✗ {s['name']}: {e}", file=sys.stderr)
                rc = 1

    stats = st.stats()
    manifest = {
        "slug_base": a.slug, "owner": a.owner,
        "total_accepted": stats["total_accepted"], "total_hours": stats["total_hours"],
        "shards": [{"name": s["name"], "status": s["status"], "n_files": s["n_files"],
                    "bytes": s["bytes"], "dataset": s["push_target"],
                    "pushed_at": s["pushed_at"]} for s in stats["shards"]],
    }
    try:
        k.push_index(out / "state" / "progress.json", manifest, f"index @ {utcnow()}")
        print("  ✓ index")
    except KaggleError as e:
        print(f"  ✗ index: {e}", file=sys.stderr)
        rc = 1
    st.close()
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
