#!/usr/bin/env python3
"""Chia train / validation / test cho dataset đã export (spec Demo v1 §10).

Ràng buộc: **speaker-disjoint VÀ recording-disjoint**.

Hai ràng buộc này không suy ra nhau. VietMed là hội thoại: mỗi recording
(`audio_name`) chứa NHIỀU speaker — `VietMed_011` gồm `_a`, `_b`, `_c`. Chia theo
speaker sẽ xé một recording ra hai split, chia theo recording sẽ để một speaker
xuất hiện ở hai split. Nên đơn vị chia không phải speaker, cũng không phải
recording, mà là **cụm liên thông** của đồ thị speaker ↔ recording.

Chia riêng cho từng (label, source) để mỗi split vừa cân REAL/FAKE vừa có đủ
mọi nguồn.

    python scripts/make_splits.py --dataset /data/dataset/vietnamese-audio-deepfake-demo
    python scripts/make_splits.py --dataset ... --dry-run
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

SPLITS = ("train", "validation", "test")
BUCKETS = ("2-4s", "4-6s", "6-8s", "8-10s")


class Union:
    def __init__(self):
        self.p: dict = {}

    def find(self, x):
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)   # gộp xác định, không phụ thuộc thứ tự


def bucket(d: float) -> str:
    return "2-4s" if d < 4 else "4-6s" if d < 6 else "6-8s" if d < 8 else "8-10s"


def build_groups(rows: list[dict]) -> dict:
    """Cụm liên thông speaker ↔ recording. Trả {khoá cụm: [chỉ số dòng]}."""
    u = Union()
    for r in rows:
        u.union(("S", r["speaker_id"]), ("R", r["recording_id"]))
    groups: dict = defaultdict(list)
    for i, r in enumerate(rows):
        groups[u.find(("S", r["speaker_id"]))].append(i)
    return groups


def assign(rows: list[dict], groups: dict, ratios: dict[str, float]) -> dict[int, str]:
    """Xếp cụm vào split theo kiểu lấp chỗ thiếu nhất, cụm to xếp trước."""
    by_part: dict = defaultdict(list)
    for key, idxs in groups.items():
        r = rows[idxs[0]]
        by_part[(r["label"], r["source"])].append((key, idxs))

    out: dict[int, str] = {}
    for part in sorted(by_part, key=lambda p: (p[0], p[1])):
        items = by_part[part]
        total = sum(len(i) for _, i in items)
        target = {s: total * ratios[s] for s in SPLITS}
        cur = {s: 0 for s in SPLITS}
        # cụm lớn xếp trước để phần dư nhỏ dần; khoá cụm làm tie-break cho xác định
        for key, idxs in sorted(items, key=lambda kv: (-len(kv[1]), str(kv[0]))):
            s = max(SPLITS, key=lambda s: (target[s] - cur[s], -SPLITS.index(s)))
            cur[s] += len(idxs)
            for i in idxs:
                out[i] = s
    return out


def verify(rows: list[dict]) -> list[str]:
    """Không tin kết quả: kiểm lại speaker và recording có rò qua split không."""
    problems = []
    for field in ("speaker_id", "recording_id"):
        seen: dict = defaultdict(set)
        for r in rows:
            seen[r[field]].add(r["split"])
        leaked = {k: sorted(v) for k, v in seen.items() if len(v) > 1}
        for k, v in list(leaked.items())[:5]:
            problems.append(f"{field} {k!r} nằm ở {v}")
        if leaked:
            problems.append(f"→ tổng {len(leaked)} {field} bị rò qua nhiều split")
    if any(r.get("split") not in SPLITS for r in rows):
        problems.append("có dòng chưa được gán split")
    return problems


def warn_coarse(rows: list[dict], groups: dict, ratios: dict) -> list[str]:
    """Cụm quá to thì 70/15/15 là bất khả thi — phải nói ra chứ không im lặng.

    VietMed là hội thoại: cả nghìn utterance dính vào vài cụm. Khi một nguồn chỉ
    có 2-3 cụm thì không cách chia nào bám được tỷ lệ, và một split có thể nhận 0.
    """
    msgs = []
    per_src: dict = defaultdict(list)
    for key, idxs in groups.items():
        per_src[rows[idxs[0]]["source"]].append(len(idxs))
    for src in sorted(per_src):
        sizes = sorted(per_src[src], reverse=True)
        n = sum(sizes)
        if len(sizes) < len(SPLITS) * 2:
            msgs.append(f"{src}: chỉ {len(sizes)} cụm cho {n:,} audio "
                        f"(cụm lớn nhất {sizes[0]:,} = {sizes[0] / n * 100:.0f}%) "
                        f"→ không thể bám sát {int(ratios['train'] * 100)}/"
                        f"{int(ratios['validation'] * 100)}/{int(ratios['test'] * 100)}")
        sub = [r for r in rows if r["source"] == src]
        for sp in SPLITS:
            k = sum(1 for r in sub if r["split"] == sp)
            if k == 0:
                msgs.append(f"{src}: split '{sp}' KHÔNG có mẫu nào của nguồn này")
            elif abs(k / len(sub) - ratios[sp]) > 0.10:
                msgs.append(f"{src}: split '{sp}' lệch nhiều — {k / len(sub) * 100:.0f}% "
                            f"thay vì {ratios[sp] * 100:.0f}%")
    return msgs


def report(rows: list[dict]) -> None:
    per = Counter(r["split"] for r in rows)
    n = len(rows)
    print(f"\n{'Split':<12}{'Số audio':>10}{'Tỷ lệ':>9}{'REAL':>9}{'FAKE':>9}{'Giờ':>9}")
    print("-" * 58)
    for s in SPLITS:
        sub = [r for r in rows if r["split"] == s]
        real = sum(1 for r in sub if r["label"] == 0)
        hrs = sum(r["duration"] for r in sub) / 3600
        print(f"{s:<12}{len(sub):>10,}{per[s] / n * 100:>8.1f}%"
              f"{real:>9,}{len(sub) - real:>9,}{hrs:>9.1f}")
    print(f"{'TỔNG':<12}{n:>10,}{100.0:>8.1f}%")

    print(f"\n{'Nguồn':<15}" + "".join(f"{s:>14}" for s in SPLITS))
    print("-" * 58)
    for src in sorted({r["source"] for r in rows}):
        sub = [r for r in rows if r["source"] == src]
        line = f"{src:<15}"
        for s in SPLITS:
            k = sum(1 for r in sub if r["split"] == s)
            line += f"{k:>8,} ({k / len(sub) * 100:>3.0f}%)"
        print(line)

    print(f"\n{'Duration':<15}" + "".join(f"{s:>14}" for s in SPLITS))
    print("-" * 58)
    for b in BUCKETS:
        line = f"{b:<15}"
        for s in SPLITS:
            sub = [r for r in rows if r["split"] == s]
            k = sum(1 for r in sub if bucket(r["duration"]) == b)
            line += f"{k:>8,} ({k / max(len(sub), 1) * 100:>3.0f}%)"
        print(line)

    print(f"\n{'Speaker':<15}" + "".join(f"{s:>14}" for s in SPLITS))
    print("-" * 58)
    line = f"{'số speaker':<15}"
    for s in SPLITS:
        line += f"{len({r['speaker_id'] for r in rows if r['split'] == s}):>14,}"
    print(line)
    line = f"{'số recording':<15}"
    for s in SPLITS:
        line += f"{len({r['recording_id'] for r in rows if r['split'] == s}):>14,}"
    print(line)


def write_back(dest: Path, rows: list[dict]) -> None:
    with open(dest / "metadata.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    cols = list(rows[0].keys())
    flat = [{c: (json.dumps(r[c], ensure_ascii=False)
                 if isinstance(r[c], (dict, list)) else r[c]) for c in cols}
            for r in rows]
    with open(dest / "metadata.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(flat)

    import pyarrow as pa
    import pyarrow.parquet as pq
    pq.write_table(pa.Table.from_pylist(flat), dest / "metadata.parquet",
                   compression="zstd")


def main() -> int:
    p = argparse.ArgumentParser(description="Chia train/validation/test speaker-disjoint")
    p.add_argument("--dataset", default="/data/dataset/vietnamese-audio-deepfake-demo")
    p.add_argument("--train", type=float, default=0.70)
    p.add_argument("--val", type=float, default=0.15)
    p.add_argument("--test", type=float, default=0.15)
    p.add_argument("--dry-run", action="store_true", help="chỉ in kết quả, không ghi")
    a = p.parse_args()

    dest = Path(a.dataset)
    src = dest / "metadata.jsonl"
    if not src.exists():
        print(f"LỖI: không thấy {src} — chạy export_dataset.py trước.", file=sys.stderr)
        return 2
    rows = [json.loads(l) for l in open(src, encoding="utf-8") if l.strip()]
    ratios = {"train": a.train, "validation": a.val, "test": a.test}
    tot = sum(ratios.values())
    if abs(tot - 1.0) > 1e-6:
        print(f"LỖI: tỷ lệ cộng lại phải bằng 1, đang là {tot}", file=sys.stderr)
        return 2

    groups = build_groups(rows)
    sizes = sorted((len(v) for v in groups.values()), reverse=True)
    print(f"Audio        : {len(rows):,}")
    print(f"Cụm liên thông: {len(groups):,}  (lớn nhất {sizes[0]:,} audio"
          f" = {sizes[0] / len(rows) * 100:.1f}% toàn bộ)")

    for i, s in assign(rows, groups, ratios).items():
        rows[i]["split"] = s

    problems = verify(rows)
    report(rows)
    warnings = warn_coarse(rows, groups, ratios)
    if warnings:
        print("\nCẢNH BÁO — ràng buộc disjoint làm tỷ lệ lệch:")
        for m in warnings:
            print("  ⚠", m)
        print("  Đây là giới hạn của dữ liệu, không phải lỗi chia: một recording hội "
              "thoại\n  không tách ra hai split được. Chấp nhận, hoặc đổi tỷ lệ bằng "
              "--train/--val/--test.")
    if problems:
        print("\nKIỂM TRA THẤT BẠI:", file=sys.stderr)
        for m in problems:
            print("  ✗", m, file=sys.stderr)
        return 1
    print("\n✓ speaker-disjoint và recording-disjoint: không có cái nào nằm ở 2 split.")

    if a.dry_run:
        print("(--dry-run: chưa ghi gì)")
        return 0
    write_back(dest, rows)
    print(f"Đã ghi lại metadata.jsonl / metadata.csv / metadata.parquet trong {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
