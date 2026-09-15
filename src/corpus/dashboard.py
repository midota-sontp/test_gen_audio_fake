"""Gom số liệu cho dashboard HTML. Chỉ đọc, không đụng vào tiến độ của job."""
from __future__ import annotations

import json
import shutil
import sqlite3
from pathlib import Path

from .state import State

EXPECTED = {"common_voice": 8963, "vietmed": 9207, "vivos": 12420, "vieneu_tts": 74858}
LABEL_OF = {"common_voice": 0, "vietmed": 0, "vivos": 0, "vieneu_tts": 1}
REASON_VI = {
    "decode_failed": "Lỗi giải mã",
    "empty_audio": "Audio rỗng",
    "sample_rate_invalid": "Sample rate không hợp lệ",
    "duration_out_of_range": "Duration ngoài 2–10s",
    "speech_ratio_too_low": "Speech < 50%",
    "clipping_too_high": "Clipping ≥ 1%",
    "duplicate_raw_sha256": "Trùng (hash gốc)",
    "duplicate_normalized_sha256": "Trùng (hash chuẩn hoá)",
    "normalize_failed": "Lỗi chuẩn hoá",
}
BUCKETS = ("2-4s", "4-6s", "6-8s", "8-10s")


def _progress_json(out: Path) -> dict:
    p = out / "state" / "progress.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def build(out: Path) -> dict:
    out = Path(out)
    db_path = out / "state" / "corpus.sqlite"
    prog = _progress_json(out)
    if not db_path.exists():
        return {"ready": False, "message": "Chưa có job nào chạy ở thư mục này.",
                "out": str(out)}

    st = State(db_path, readonly=True)
    try:
        stats = st.stats()
        buckets = {"real": st.bucket_counts(0), "fake": st.bucket_counts(1)}
        speakers = {r["label"]: r["n"] for r in st.db.execute(
            "SELECT label, COUNT(DISTINCT speaker_id) n FROM items GROUP BY label")}
        hours = {r["label"]: round((r["d"] or 0) / 3600, 2) for r in st.db.execute(
            "SELECT label, SUM(duration) d FROM items GROUP BY label")}
        unknown_spk = st.db.execute(
            "SELECT COUNT(*) n FROM items WHERE speaker_id LIKE '%unknown%'").fetchone()["n"]
        src_rates = {r["source"]: {"mean": round(r["m"] or 0, 3),
                                   "min": r["lo"], "max": r["hi"]}
                     for r in st.db.execute(
                         "SELECT source, AVG(duration) m, MIN(duration) lo, "
                         "MAX(duration) hi FROM items GROUP BY source")}
        selection = st.get_meta("fake_selection")
    except sqlite3.Error as e:
        st.db.close()
        return {"ready": False, "message": f"Không đọc được sqlite: {e}"}
    st.db.close()

    # Con trỏ được đặt tên theo pha ("index:vieneu_tts", "write:vieneu_tts"),
    # nên gom theo (nguồn, pha) rồi để UI hiển thị từng pha một.
    phases: dict[str, dict[str, dict]] = {}
    for c in stats["cursors"]:
        ns, _, src = c["source"].rpartition(":")
        ph = phases.setdefault(src, {}).setdefault(
            ns or "scan", {"name": ns or "scan", "scanned": 0, "units": 0,
                           "units_done": 0})
        ph["scanned"] += c["next_row"]
        ph["units"] += 1
        ph["units_done"] += int(bool(c["unit_done"]))

    sources = []
    for name in ("common_voice", "vietmed", "vivos", "vieneu_tts"):
        ph = sorted(phases.get(name, {}).values(), key=lambda d: d["name"])
        rej = stats["rejected_by_source"].get(name, {})
        acc = stats["accepted_by_source"].get(name, 0)
        cand = stats["candidates"].get(name, {})
        sources.append({
            "name": name,
            "label": LABEL_OF[name],
            "expected": EXPECTED[name],
            "phases": ph,
            "scanned": max((p["scanned"] for p in ph), default=0),
            "units": max((p["units"] for p in ph), default=0),
            "units_done": max((p["units_done"] for p in ph), default=0),
            "accepted": acc,
            "candidates": cand.get("total", 0),
            "selected": cand.get("selected", 0),
            "rejected": sum(rej.values()),
            "reject_reasons": [{"reason": k, "label": REASON_VI.get(k, k), "n": v}
                               for k, v in sorted(rej.items(), key=lambda kv: -kv[1])],
            "duration": src_rates.get(name, {}),
        })

    real, fake = stats["accepted_by_label"]["real"], stats["accepted_by_label"]["fake"]
    try:
        disk = shutil.disk_usage(out)
        disk_free = disk.free
    except OSError:
        disk_free = None

    return {
        "ready": True,
        "out": str(out),
        "status": prog.get("status", "unknown"),
        "updated_at": prog.get("updated_at"),
        "elapsed_sec": prog.get("elapsed_sec"),
        "rate_per_sec": prog.get("rate_per_sec"),
        "processed_this_run": prog.get("processed_this_run"),
        "totals": {
            "real": real, "fake": fake, "all": real + fake,
            "rejected": stats["total_rejected"],
            "hours_real": hours.get(0, 0), "hours_fake": hours.get(1, 0),
            "hours": stats["total_hours"], "bytes": stats["total_bytes"],
            "speakers_real": speakers.get(0, 0), "speakers_fake": speakers.get(1, 0),
            "speakers_unknown_items": unknown_spk,
            "balanced": real == fake and real > 0,
            "gap": abs(real - fake),
        },
        "buckets": buckets,
        "bucket_names": list(BUCKETS),
        "sources": sources,
        "shards": stats["shards"],
        "selection": selection,
        "disk_free": disk_free,
        "config": prog.get("config", {}),
    }
