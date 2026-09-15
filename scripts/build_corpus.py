#!/usr/bin/env python3
"""Dựng corpus REAL tiếng Việt đạt chuẩn Demo v1.

Ví dụ:
  python scripts/build_corpus.py --out /data/out --cache /data/cache
  python scripts/build_corpus.py --out /data/out --sources vivos --limit 50 --no-kaggle
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from corpus.builder import BuildConfig, Builder          # noqa: E402
from corpus.kaggle_sync import KaggleSync                # noqa: E402
from corpus.sources.registry import REAL_SOURCES, REGISTRY  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Dựng corpus audio REAL tiếng Việt (Demo v1)")
    g = p.add_argument_group("đường dẫn")
    g.add_argument("--out", default=os.getenv("CORPUS_OUT", "/data/out"),
                   help="thư mục output (dist/, state/)")
    g.add_argument("--cache", default=os.getenv("CORPUS_CACHE", "/data/cache"),
                   help="thư mục cache dữ liệu nguồn đã tải")

    g = p.add_argument_group("nguồn")
    g.add_argument("--sources", nargs="+", default=REAL_SOURCES,
                   choices=list(REGISTRY), help="các nguồn cần xử lý")
    g.add_argument("--limit", type=int, default=None,
                   help="giới hạn số audio mỗi nguồn (để chạy thử)")
    g.add_argument("--hf-token", default=os.getenv("HF_TOKEN"))

    g = p.add_argument_group("ngưỡng lọc")
    g.add_argument("--min-duration", type=float, default=2.0)
    g.add_argument("--max-duration", type=float, default=10.0)
    g.add_argument("--min-speech-ratio", type=float, default=0.50)
    g.add_argument("--max-clipping-ratio", type=float, default=0.01)
    g.add_argument("--vad-aggressiveness", type=int, default=2, choices=[0, 1, 2, 3])
    g.add_argument("--rms-dbfs", type=float, default=None,
                   help="bật chuẩn hoá RMS (vd -23). Mặc định tắt theo spec Demo v1.")

    g = p.add_argument_group("checkpoint & shard")
    g.add_argument("--checkpoint-every", type=int, default=10,
                   help="số audio NHẬN được giữa hai lần chốt (mặc định 10)")
    g.add_argument("--checkpoint-max-processed", type=int, default=250,
                   help="chốt cả khi bị loại liên tục, để con trỏ không đứng yên")
    g.add_argument("--checkpoint-max-seconds", type=float, default=60.0)
    g.add_argument("--shard-max-mb", type=int, default=256)
    g.add_argument("--max-seconds", type=float, default=None,
                   help="ngân sách thời gian cho lần chạy này, dừng sạch khi hết")

    g = p.add_argument_group("Kaggle")
    g.add_argument("--kaggle-owner", default=os.getenv("KAGGLE_USERNAME"))
    g.add_argument("--kaggle-slug", default=os.getenv("KAGGLE_SLUG", "vi-real-audio-demo-v1"))
    g.add_argument("--kaggle-public", action="store_true")
    g.add_argument("--no-kaggle", action="store_true", help="chỉ ghi xuống đĩa")
    g.add_argument("--push-index-every", type=float, default=120.0,
                   help="giây giữa hai lần đẩy file tiến độ (nhẹ, vài KB)")
    g.add_argument("--push-shard-every", type=float, default=900.0,
                   help="giây giữa hai lần đẩy shard đang mở (nặng, tới vài trăm MB)")
    return p.parse_args(argv)


def main(argv=None) -> int:
    a = parse_args(argv)
    out, cache = Path(a.out), Path(a.cache)
    out.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)

    kaggle = None
    if not a.no_kaggle:
        if not a.kaggle_owner:
            print("LỖI: thiếu --kaggle-owner (hoặc biến KAGGLE_USERNAME). "
                  "Dùng --no-kaggle nếu chỉ muốn ghi xuống đĩa.", file=sys.stderr)
            return 2
        kaggle = KaggleSync(a.kaggle_owner, a.kaggle_slug, out / "stage",
                            public=a.kaggle_public)
        kaggle.check()

    cfg = BuildConfig(
        out_dir=out, cache_dir=cache, sources=a.sources,
        min_duration=a.min_duration, max_duration=a.max_duration,
        min_speech_ratio=a.min_speech_ratio, max_clipping_ratio=a.max_clipping_ratio,
        vad_aggressiveness=a.vad_aggressiveness, rms_dbfs=a.rms_dbfs,
        checkpoint_every=a.checkpoint_every,
        checkpoint_max_processed=a.checkpoint_max_processed,
        checkpoint_max_seconds=a.checkpoint_max_seconds,
        shard_max_bytes=a.shard_max_mb * 1024 * 1024,
        limit_per_source=a.limit, max_seconds=a.max_seconds,
        hf_token=a.hf_token, kaggle=kaggle,
        push_index_every=a.push_index_every, push_shard_every=a.push_shard_every,
    )

    print(f"Output : {out}")
    print(f"Cache  : {cache}")
    print(f"Nguồn  : {', '.join(a.sources)}")
    print(f"Lọc    : {a.min_duration}-{a.max_duration}s | speech>={a.min_speech_ratio} "
          f"| clipping<{a.max_clipping_ratio}")
    print(f"Kaggle : {'tắt' if kaggle is None else f'{a.kaggle_owner}/{a.kaggle_slug}-*'}")

    stats = Builder(cfg).run()

    print("\n================ KẾT QUẢ ================")
    print(f"Nhận  : {stats['total_accepted']:,}  ({stats['total_hours']} giờ, "
          f"{stats['total_bytes'] / 1e9:.2f} GB)")
    print(f"Loại  : {stats['total_rejected']:,}")
    print(f"Speaker: {stats['speakers']:,}")
    for src, n in sorted(stats["accepted_by_source"].items()):
        print(f"  {src:<14} nhận {n:>7,}")
        for reason, m in sorted(stats["rejected_by_source"].get(src, {}).items(),
                                key=lambda kv: -kv[1]):
            print(f"      loại {reason:<26} {m:>7,}")
    print("Phân bố duration:", stats["duration_buckets"])
    print(f"progress.json: {out / 'state' / 'progress.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
