"""Chọn mẫu FAKE cân bằng với REAL (spec Demo v1 §8 & §9).

Ràng buộc:
  * tổng FAKE = tổng REAL (đúng bằng, không xấp xỉ);
  * phân bố duration của FAKE bám theo phân bố THỰC TẾ của REAL, không phải một
    tỷ lệ cố định — nếu REAL lệch ngắn thì FAKE cũng phải lệch ngắn, nếu không
    model sẽ học duration thay vì học đặc trưng AI-generated;
  * trong mỗi bucket, chia đều cho speaker theo vòng tròn để không một giọng nào
    chiếm phần lớn (VieNeu có 193 speaker).
"""
from __future__ import annotations

import random
from collections import defaultdict

BUCKETS = ("2-4s", "4-6s", "6-8s", "8-10s")


def largest_remainder(total: int, shares: dict[str, float]) -> dict[str, int]:
    """Chia `total` theo tỷ lệ mà tổng các phần vẫn đúng bằng `total`."""
    raw = {b: total * shares.get(b, 0.0) for b in BUCKETS}
    out = {b: int(v) for b, v in raw.items()}
    left = total - sum(out.values())
    for b in sorted(BUCKETS, key=lambda b: raw[b] - out[b], reverse=True):
        if left <= 0:
            break
        out[b] += 1
        left -= 1
    return out


def _round_robin(pool: dict[str, list[str]], need: int, rng: random.Random) -> list[str]:
    """Rút lần lượt mỗi speaker một mẫu cho tới khi đủ `need` hoặc hết mẫu."""
    speakers = sorted(pool)
    rng.shuffle(speakers)
    queues = {s: list(pool[s]) for s in speakers}
    for s in speakers:
        rng.shuffle(queues[s])
    picked: list[str] = []
    while len(picked) < need:
        progressed = False
        for s in speakers:
            if not queues[s]:
                continue
            picked.append(queues[s].pop())
            progressed = True
            if len(picked) >= need:
                break
        if not progressed:
            break            # hết ứng viên trong bucket này
    return picked


def select(candidates, real_bucket_counts: dict[str, int], target: int,
           seed: int = 20260915) -> tuple[list[str], dict]:
    """candidates: các row từ bảng `candidates`. Trả (danh sách id, báo cáo)."""
    rng = random.Random(seed)
    total_real = sum(real_bucket_counts.values())
    if total_real == 0:
        raise ValueError("Chưa có audio REAL nào — chạy xong REAL trước đã.")
    shares = {b: real_bucket_counts.get(b, 0) / total_real for b in BUCKETS}
    targets = largest_remainder(target, shares)

    pools: dict[str, dict[str, list[str]]] = {b: defaultdict(list) for b in BUCKETS}
    for c in candidates:
        if c["bucket"] in pools:
            pools[c["bucket"]][c["speaker_id"]].append(c["item_id"])

    selected: list[str] = []
    taken: dict[str, int] = {}
    used: set[str] = set()
    for b in BUCKETS:
        got = _round_robin(pools[b], targets[b], rng)
        taken[b] = len(got)
        used.update(got)
        selected.extend(got)

    # Bucket nào thiếu thì bù từ bucket còn dư, ưu tiên bucket kề bên để phân bố
    # duration lệch ít nhất có thể.
    deficit = target - len(selected)
    spill: dict[str, int] = {}
    if deficit > 0:
        order = sorted(BUCKETS, key=lambda b: -(len(sum(pools[b].values(), [])) - taken[b]))
        for b in order:
            if deficit <= 0:
                break
            leftover = {s: [i for i in ids if i not in used] for s, ids in pools[b].items()}
            leftover = {s: v for s, v in leftover.items() if v}
            got = _round_robin(leftover, deficit, rng)
            if got:
                spill[b] = len(got)
                used.update(got)
                selected.extend(got)
                taken[b] += len(got)
                deficit -= len(got)

    avail = {b: sum(len(v) for v in pools[b].values()) for b in BUCKETS}
    report = {
        "target": target,
        "selected": len(selected),
        "short_by": max(target - len(selected), 0),
        "real_shares": {b: round(shares[b], 4) for b in BUCKETS},
        "bucket_target": targets,
        "bucket_taken": taken,
        "bucket_available": avail,
        "spill": spill,
        "speakers_used": len({c["speaker_id"] for c in candidates
                              if c["item_id"] in used}),
        "speakers_available": len({c["speaker_id"] for c in candidates}),
        "seed": seed,
    }
    return selected, report
