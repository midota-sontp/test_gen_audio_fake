#!/usr/bin/env python3
"""Theo dõi tiến độ job dựng corpus. Đọc progress.json / sqlite, không đụng vào job.

  python scripts/monitor.py --out /data/out            # xem một lần
  python scripts/monitor.py --out /data/out --watch 10 # tự làm mới mỗi 10 giây
  python scripts/monitor.py --out /data/out --json     # cho script khác đọc
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

EXPECTED = {"common_voice": 8963, "vietmed": 9207, "vivos": 12420, "vieneu_tts": 73882}
REASON_VI = {
    "decode_failed": "lỗi giải mã",
    "empty_audio": "audio rỗng",
    "sample_rate_invalid": "sample rate không hợp lệ",
    "duration_out_of_range": "duration ngoài 2-10s",
    "speech_ratio_too_low": "speech < 50%",
    "clipping_too_high": "clipping >= 1%",
    "duplicate_raw_sha256": "trùng (hash gốc)",
    "duplicate_normalized_sha256": "trùng (hash sau chuẩn hoá)",
    "normalize_failed": "lỗi chuẩn hoá",
}


def load(out: Path) -> dict:
    """Ưu tiên đọc thẳng sqlite (mới nhất), lùi về progress.json nếu DB đang khoá."""
    db = out / "state" / "corpus.sqlite"
    if db.exists():
        try:
            from corpus.state import State
            st = State(db, readonly=True)
            data = st.stats()
            st.close()
            data["_from"] = "sqlite"
            prog = out / "state" / "progress.json"
            if prog.exists():
                p = json.loads(prog.read_text())
                for k in ("updated_at", "status", "elapsed_sec",
                          "processed_this_run", "rate_per_sec", "config"):
                    data.setdefault(k, p.get(k))
            return data
        except Exception as e:
            print(f"(không đọc được sqlite: {e})", file=sys.stderr)
    prog = out / "state" / "progress.json"
    if not prog.exists():
        raise SystemExit(f"Chưa có tiến độ nào tại {out}")
    data = json.loads(prog.read_text())
    data["_from"] = "progress.json"
    return data


def bar(done: int, total: int, width: int = 34) -> str:
    if not total:
        return "-" * width
    frac = min(done / total, 1.0)
    n = int(frac * width)
    return "█" * n + "░" * (width - n) + f" {frac * 100:5.1f}%"


def render(d: dict) -> str:
    L = []
    status = {"running": "ĐANG CHẠY", "stopped": "ĐÃ DỪNG", "done": "HOÀN TẤT"}.get(
        d.get("status"), d.get("status") or "?")
    L.append("=" * 72)
    L.append(f" CORPUS REAL — {status}     cập nhật: {d.get('updated_at', '?')}")
    L.append("=" * 72)

    acc = d.get("accepted_by_source", {})
    rej = d.get("rejected_by_source", {})
    skp = d.get("skipped_by_source", {})
    cursors = d.get("cursors", [])
    by_src: dict[str, list] = {}
    for c in cursors:
        by_src.setdefault(c["source"], []).append(c)

    L.append("\n── Tiến độ quét theo nguồn " + "─" * 45)
    for src in sorted(set(list(by_src) + list(acc) + list(EXPECTED))):
        seen = sum(c["next_row"] for c in by_src.get(src, []))
        total = EXPECTED.get(src, 0)
        units_done = sum(1 for c in by_src.get(src, []) if c["unit_done"])
        units = len(by_src.get(src, []))
        a = acc.get(src, 0)
        r = sum(rej.get(src, {}).values())
        L.append(f"  {src:<13} {bar(seen, total)}  {seen:>6,}/{total:,}"
                 f"  (unit {units_done}/{units})")
        k = sum(skp.get(src, {}).values())
        keep = f"{a / (a + r) * 100:.1f}%" if (a + r) else "—"
        L.append(f"  {'':<13} nhận {a:>6,} | loại {r:>6,} | tỷ lệ giữ {keep}"
                 + (f" | bỏ qua {k:,} trùng id" if k else ""))
        for reason, n in sorted(rej.get(src, {}).items(), key=lambda kv: -kv[1])[:4]:
            L.append(f"  {'':<15} · {REASON_VI.get(reason, reason):<28} {n:>6,}")

    L.append("\n── Tổng " + "─" * 63)
    tb = d.get("total_bytes", 0)
    L.append(f"  Nhận: {d.get('total_accepted', 0):>7,} audio"
             f" | {d.get('total_hours', 0):>8.2f} giờ"
             f" | {tb / 1e9:>5.2f} GB"
             f" | {d.get('speakers', 0):,} speaker")
    L.append(f"  Loại: {d.get('total_rejected', 0):>7,}")
    if d.get("rate_per_sec"):
        L.append(f"  Tốc độ: {d['rate_per_sec']} audio/giây"
                 f" | đã chạy {d.get('elapsed_sec', 0) / 60:.1f} phút")
        remain = sum(EXPECTED.values()) - sum(c["next_row"] for c in cursors)
        if remain > 0:
            L.append(f"  Còn ~{remain:,} audio → ETA ~"
                     f"{remain / d['rate_per_sec'] / 60:.0f} phút")

    buckets = d.get("duration_buckets", {})
    tot = sum(buckets.values()) or 1
    L.append("\n── Phân bố duration " + "─" * 51)
    for k in ("2-4s", "4-6s", "6-8s", "8-10s"):
        n = buckets.get(k, 0)
        L.append(f"  {k:<6} {n:>7,}  {n / tot * 100:5.1f}%  {'▇' * int(n / tot * 40)}")

    shards = d.get("shards", [])
    L.append("\n── Shard / đồng bộ Kaggle " + "─" * 45)
    if not shards:
        L.append("  (chưa có shard)")
    pushed = sum(1 for s in shards if s.get("pushed_at"))
    if len(shards) > 12:
        L.append(f"  ({len(shards)} shard, đã đẩy {pushed} — hiện 12 cái mới nhất)")
        shards = shards[-12:]
    for s in shards:
        pushed = s.get("pushed_at") or "chưa đẩy"
        tgt = s.get("push_target") or "-"
        L.append(f"  {s['name']:<18} {s['status']:<7} {s['n_files']:>6,} file"
                 f" {(s['bytes'] or 0) / 1e6:>8.1f} MB  → {tgt}  [{pushed}]")
    return "\n".join(L)


def main() -> int:
    p = argparse.ArgumentParser(description="Theo dõi tiến độ dựng corpus")
    p.add_argument("--out", default="/data/out")
    p.add_argument("--watch", type=float, default=0, help="giây giữa hai lần làm mới")
    p.add_argument("--json", action="store_true")
    a = p.parse_args()
    out = Path(a.out)
    while True:
        d = load(out)
        if a.json:
            print(json.dumps(d, ensure_ascii=False, indent=2, default=str))
        else:
            print(("\033[2J\033[H" if a.watch else "") + render(d), flush=True)
        if not a.watch:
            return 0
        time.sleep(a.watch)


if __name__ == "__main__":
    raise SystemExit(main())
