#!/usr/bin/env python3
"""Dựng phần FAKE từ VieNeu-TTS-140h, số lượng bằng đúng số REAL đã có.

Ba pha, pha nào cũng resume được:

  1. index      quét toàn bộ VieNeu, chạy đúng bộ lọc chất lượng của REAL,
                ghi sổ ứng viên (chưa ghi audio — chưa biết cần bao nhiêu).
  2. select     N = số REAL đã nhận; chọn N ứng viên, bám phân bố duration của
                REAL và chia đều cho speaker.
  3. write      quét lại, chỉ ghi đúng N audio đã chọn (item khác bỏ qua trước
                khi decode nên pha này nhanh hơn pha 1 nhiều).

    export HF_TOKEN=hf_...            # dataset gated, phải accept terms trước
    python scripts/build_fake.py --out /data/out --cache /data/cache --probe
    python scripts/build_fake.py --out /data/out --cache /data/cache
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from corpus.builder import BuildConfig, Builder          # noqa: E402
from corpus.kaggle_sync import KaggleSync                # noqa: E402
from corpus.selector import select                       # noqa: E402
from corpus.sources.base import AccessError             # noqa: E402
from corpus.sources.vieneu import VieNeuSource           # noqa: E402
from corpus.state import State, write_progress_json      # noqa: E402

SOURCE = "vieneu_tts"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Dựng FAKE cân bằng với REAL")
    p.add_argument("--out", default=os.getenv("CORPUS_OUT", "/data/out"))
    p.add_argument("--cache", default=os.getenv("CORPUS_CACHE", "/data/cache"))
    p.add_argument("--hf-token", default=os.getenv("HF_TOKEN"))
    p.add_argument("--phase", default="all",
                   choices=["all", "index", "select", "write"])
    p.add_argument("--probe", action="store_true",
                   help="tải 1 file arrow, in schema rồi thoát")
    p.add_argument("--target", type=int, default=None,
                   help="ép số FAKE (mặc định = số REAL đã nhận)")
    p.add_argument("--seed", type=int, default=20260915)
    p.add_argument("--limit", type=int, default=None, help="giới hạn để chạy thử")
    p.add_argument("--no-keep-cache", action="store_true",
                   help="xoá từng file arrow sau khi xử lý (đĩa ~500 MB thay vì 24 GB, "
                        "đổi lại phải tải lại ở pha write)")
    # cột có thể ép nếu dò sai
    p.add_argument("--audio-column")
    p.add_argument("--speaker-column")
    p.add_argument("--text-column")
    p.add_argument("--id-column")
    # ngưỡng lọc: mặc định giống hệt REAL
    p.add_argument("--min-duration", type=float, default=2.0)
    p.add_argument("--max-duration", type=float, default=10.0)
    p.add_argument("--min-speech-ratio", type=float, default=0.50)
    p.add_argument("--max-clipping-ratio", type=float, default=0.01)
    p.add_argument("--vad-aggressiveness", type=int, default=2, choices=[0, 1, 2, 3])
    p.add_argument("--checkpoint-every", type=int, default=10)
    p.add_argument("--shard-max-mb", type=int, default=256)
    p.add_argument("--max-seconds", type=float, default=None)
    # Kaggle
    p.add_argument("--kaggle-owner", default=os.getenv("KAGGLE_USERNAME"))
    p.add_argument("--kaggle-slug", default=os.getenv("KAGGLE_SLUG", "vi-real-audio-demo-v1"))
    p.add_argument("--kaggle-public", action="store_true")
    p.add_argument("--no-kaggle", action="store_true")
    p.add_argument("--push-index-every", type=float, default=120.0)
    p.add_argument("--push-shard-every", type=float, default=900.0)
    return p.parse_args(argv)


def make_cfg(a, out: Path, cache: Path, kaggle, *, index_only: bool,
             only_ids=None, prefix: str) -> BuildConfig:
    return BuildConfig(
        out_dir=out, cache_dir=cache, sources=[SOURCE],
        min_duration=a.min_duration, max_duration=a.max_duration,
        min_speech_ratio=a.min_speech_ratio, max_clipping_ratio=a.max_clipping_ratio,
        vad_aggressiveness=a.vad_aggressiveness,
        checkpoint_every=a.checkpoint_every,
        shard_max_bytes=a.shard_max_mb * 1024 * 1024,
        limit_per_source=a.limit, max_seconds=a.max_seconds,
        hf_token=a.hf_token, kaggle=kaggle,
        push_index_every=a.push_index_every, push_shard_every=a.push_shard_every,
        index_only=index_only, only_ids=only_ids, cursor_prefix=prefix,
        source_kwargs={SOURCE: {
            "keep_cache": not a.no_keep_cache,
            "columns": {"audio": a.audio_column, "speaker": a.speaker_column,
                        "text": a.text_column, "id": a.id_column},
        }},
    )


def do_select(out: Path, target: int | None, seed: int) -> dict:
    st = State(out / "state" / "corpus.sqlite")
    n_real = st.count_items(label=0)
    n = target if target is not None else n_real
    cands = st.candidates(SOURCE)
    print(f"REAL đã nhận : {n_real:,}")
    print(f"Ứng viên FAKE: {len(cands):,}")
    print(f"Cần chọn     : {n:,}")
    if not cands:
        st.close()
        raise SystemExit("Chưa có ứng viên nào — chạy pha index trước.")
    ids, report = select(cands, st.bucket_counts(label=0), n, seed=seed)
    st.begin()
    st.set_selected(ids, SOURCE)
    st.set_meta("fake_selection", report)
    st.commit()
    st.close()
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["short_by"]:
        print(f"CẢNH BÁO: thiếu {report['short_by']:,} mẫu — không đủ ứng viên FAKE. "
              f"FAKE sẽ ít hơn REAL, cân lại bằng cách giảm REAL hoặc thêm generator.",
              file=sys.stderr)
    return report


def main(argv=None) -> int:
    a = parse_args(argv)
    out, cache = Path(a.out), Path(a.cache)
    out.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)

    # Kiểm quyền TRƯỚC khi vào pha 1: hỏng ở đây thì hỏng ngay, không để job chạy
    # được một lúc rồi mới ngã sấp mặt lúc bắt đầu tải 24 GB.
    try:
        src = VieNeuSource(cache, a.hf_token, keep_cache=not a.no_keep_cache,
                           columns={"audio": a.audio_column, "speaker": a.speaker_column,
                                    "text": a.text_column, "id": a.id_column})
        pre = src.preflight()
    except AccessError as e:
        print(f"\nLỖI QUYỀN TRUY CẬP VieNeu-TTS-140h\n{e}\n", file=sys.stderr)
        return 2
    print(f"HuggingFace: đăng nhập '{pre['user']}', đã có quyền đọc "
          f"{SOURCE} ({pre['first_file_bytes'] / 1e6:.0f} MB/file × 49)")

    if a.probe:
        info = src.probe()
        print("Các cột trong file arrow:")
        for name, typ in info["fields"]:
            print(f"  {name:<24} {typ}")
        print("\nDò được:")
        for role, col in info["detected"].items():
            print(f"  {role:<10} -> {col}")
        if not info["detected"].get("speaker"):
            print("\nCẢNH BÁO: không tìm thấy cột speaker. Chọn mẫu cân bằng theo "
                  "speaker sẽ không hoạt động — ép bằng --speaker-column.", file=sys.stderr)
        return 0

    kaggle = None
    if not a.no_kaggle:
        if not a.kaggle_owner:
            print("LỖI: thiếu --kaggle-owner / KAGGLE_USERNAME (hoặc dùng --no-kaggle).",
                  file=sys.stderr)
            return 2
        kaggle = KaggleSync(a.kaggle_owner, a.kaggle_slug, out / "stage",
                            public=a.kaggle_public)
        kaggle.check()

    if a.phase in ("all", "index"):
        print("\n########## PHA 1/3 — INDEX ứng viên FAKE ##########")
        cfg = make_cfg(a, out, cache, None, index_only=True, prefix="index:")
        s = Builder(cfg).run()
        c = s["candidates"].get(SOURCE, {})
        print(f"Ứng viên: {c.get('total', 0):,} | loại: "
              f"{sum(s['rejected_by_source'].get(SOURCE, {}).values()):,}")
        for reason, n in sorted(s["rejected_by_source"].get(SOURCE, {}).items(),
                                key=lambda kv: -kv[1]):
            print(f"    {reason:<28} {n:>8,}")

    if a.phase in ("all", "select"):
        print("\n########## PHA 2/3 — CHỌN MẪU CÂN BẰNG ##########")
        do_select(out, a.target, a.seed)

    if a.phase in ("all", "write"):
        print("\n########## PHA 3/3 — GHI AUDIO FAKE ##########")
        st = State(out / "state" / "corpus.sqlite")
        ids = st.selected_ids(SOURCE)
        st.close()
        if not ids:
            print("Chưa chọn mẫu nào — chạy pha select trước.", file=sys.stderr)
            return 1
        print(f"Sẽ ghi {len(ids):,} audio FAKE.")
        cfg = make_cfg(a, out, cache, kaggle, index_only=False,
                       only_ids=ids, prefix="write:")
        s = Builder(cfg).run()
        real, fake = s["accepted_by_label"]["real"], s["accepted_by_label"]["fake"]
        print("\n================ KẾT QUẢ ================")
        print(f"REAL: {real:,} | FAKE: {fake:,} | tổng {real + fake:,}")
        print(f"Cân bằng: {'ĐẠT' if real == fake else f'LỆCH {abs(real - fake):,}'}")
        print("Phân bố duration:", s["duration_buckets"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
