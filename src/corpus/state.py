"""Trạng thái bền vững của job: con trỏ resume, hash dedupe, thống kê.

Toàn bộ nằm trong MỘT file sqlite để mỗi checkpoint là một transaction duy nhất:
con trỏ, hash, item và offset của shard cùng commit hoặc cùng rollback.
progress.json chỉ là bản chiếu cho người/monitor đọc, không phải nguồn sự thật.
"""
from __future__ import annotations

import json
import os
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS cursor(
    source TEXT NOT NULL, unit TEXT NOT NULL, unit_index INTEGER NOT NULL,
    next_row INTEGER NOT NULL DEFAULT 0, unit_done INTEGER NOT NULL DEFAULT 0,
    updated_at TEXT, PRIMARY KEY(source, unit));
CREATE TABLE IF NOT EXISTS hashes(
    h TEXT PRIMARY KEY, kind TEXT NOT NULL, item_id TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS items(
    item_id TEXT PRIMARY KEY, source TEXT NOT NULL, shard TEXT NOT NULL,
    arcname TEXT NOT NULL, speaker_id TEXT, recording_id TEXT,
    duration REAL, bytes INTEGER, created_at TEXT);
CREATE TABLE IF NOT EXISTS rejects(
    item_id TEXT PRIMARY KEY, source TEXT NOT NULL, reason TEXT NOT NULL,
    detail TEXT, created_at TEXT);
CREATE TABLE IF NOT EXISTS shards(
    name TEXT PRIMARY KEY, status TEXT NOT NULL, n_files INTEGER DEFAULT 0,
    bytes INTEGER DEFAULT 0, tar_offset INTEGER DEFAULT 0, meta_offset INTEGER DEFAULT 0,
    created_at TEXT, closed_at TEXT, pushed_at TEXT, push_target TEXT);
CREATE INDEX IF NOT EXISTS idx_items_source ON items(source);
CREATE INDEX IF NOT EXISTS idx_rejects_reason ON rejects(source, reason);
"""


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class State:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path, isolation_level=None, timeout=60)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.executescript(SCHEMA)
        self._in_tx = False

    # ---------- transaction ----------
    def begin(self) -> None:
        if not self._in_tx:
            self.db.execute("BEGIN IMMEDIATE")
            self._in_tx = True

    def commit(self) -> None:
        if self._in_tx:
            self.db.execute("COMMIT")
            self._in_tx = False

    def rollback(self) -> None:
        if self._in_tx:
            self.db.execute("ROLLBACK")
            self._in_tx = False

    def close(self) -> None:
        self.rollback()
        self.db.close()

    # ---------- meta ----------
    def set_meta(self, key: str, value) -> None:
        self.db.execute("INSERT INTO meta(key,value) VALUES(?,?) "
                        "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                        (key, json.dumps(value, ensure_ascii=False)))

    def get_meta(self, key: str, default=None):
        row = self.db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return json.loads(row["value"]) if row else default

    # ---------- cursor ----------
    def cursor_for(self, source: str, unit: str, unit_index: int) -> tuple[int, bool]:
        row = self.db.execute(
            "SELECT next_row, unit_done FROM cursor WHERE source=? AND unit=?",
            (source, unit)).fetchone()
        if row is None:
            self.db.execute(
                "INSERT INTO cursor(source,unit,unit_index,next_row,unit_done,updated_at) "
                "VALUES(?,?,?,0,0,?)", (source, unit, unit_index, utcnow()))
            return 0, False
        return int(row["next_row"]), bool(row["unit_done"])

    def set_cursor(self, source: str, unit: str, next_row: int, done: bool = False) -> None:
        self.db.execute(
            "UPDATE cursor SET next_row=?, unit_done=?, updated_at=? WHERE source=? AND unit=?",
            (next_row, int(done), utcnow(), source, unit))

    # ---------- dedupe ----------
    def load_hashes(self) -> set[str]:
        return {r[0] for r in self.db.execute("SELECT h FROM hashes")}

    def add_hash(self, h: str, kind: str, item_id: str) -> None:
        self.db.execute("INSERT OR IGNORE INTO hashes(h,kind,item_id) VALUES(?,?,?)",
                        (h, kind, item_id))

    # ---------- kết quả ----------
    def add_item(self, item_id, source, shard, arcname, speaker_id,
                 recording_id, duration, nbytes) -> None:
        self.db.execute(
            "INSERT OR REPLACE INTO items"
            "(item_id,source,shard,arcname,speaker_id,recording_id,duration,bytes,created_at)"
            " VALUES(?,?,?,?,?,?,?,?,?)",
            (item_id, source, shard, arcname, speaker_id, recording_id,
             duration, nbytes, utcnow()))

    def add_reject(self, item_id, source, reason, detail=None) -> None:
        self.db.execute(
            "INSERT OR REPLACE INTO rejects(item_id,source,reason,detail,created_at)"
            " VALUES(?,?,?,?,?)", (item_id, source, reason, detail, utcnow()))

    def has_item(self, item_id: str) -> bool:
        return self.db.execute("SELECT 1 FROM items WHERE item_id=?",
                               (item_id,)).fetchone() is not None

    # ---------- shard ----------
    def open_shard(self) -> sqlite3.Row | None:
        return self.db.execute(
            "SELECT * FROM shards WHERE status='open' ORDER BY name LIMIT 1").fetchone()

    def create_shard(self, name: str) -> None:
        self.db.execute(
            "INSERT OR IGNORE INTO shards(name,status,created_at) VALUES(?, 'open', ?)",
            (name, utcnow()))

    def update_shard(self, name: str, **fields) -> None:
        if not fields:
            return
        cols = ", ".join(f"{k}=?" for k in fields)
        self.db.execute(f"UPDATE shards SET {cols} WHERE name=?",
                        (*fields.values(), name))

    def shards(self, status: str | None = None) -> list[sqlite3.Row]:
        if status:
            return self.db.execute("SELECT * FROM shards WHERE status=? ORDER BY name",
                                   (status,)).fetchall()
        return self.db.execute("SELECT * FROM shards ORDER BY name").fetchall()

    def next_shard_index(self) -> int:
        row = self.db.execute("SELECT COUNT(*) c FROM shards").fetchone()
        return int(row["c"])

    # ---------- thống kê cho monitor ----------
    def stats(self) -> dict:
        db = self.db
        accepted = {r["source"]: r["n"] for r in db.execute(
            "SELECT source, COUNT(*) n FROM items GROUP BY source")}
        rejected = {}
        for r in db.execute(
                "SELECT source, reason, COUNT(*) n FROM rejects GROUP BY source, reason"):
            rejected.setdefault(r["source"], {})[r["reason"]] = r["n"]
        buckets = {"2-4s": 0, "4-6s": 0, "6-8s": 0, "8-10s": 0}
        for r in db.execute(
                "SELECT CASE WHEN duration<4 THEN '2-4s' WHEN duration<6 THEN '4-6s' "
                "WHEN duration<8 THEN '6-8s' ELSE '8-10s' END b, COUNT(*) n "
                "FROM items GROUP BY b"):
            buckets[r["b"]] = r["n"]
        tot = db.execute(
            "SELECT COUNT(*) n, COALESCE(SUM(duration),0) d, COALESCE(SUM(bytes),0) b "
            "FROM items").fetchone()
        return {
            "accepted_by_source": accepted,
            "rejected_by_source": rejected,
            "duration_buckets": buckets,
            "total_accepted": tot["n"],
            "total_rejected": db.execute("SELECT COUNT(*) n FROM rejects").fetchone()["n"],
            "total_hours": round(tot["d"] / 3600.0, 3),
            "total_bytes": tot["b"],
            "speakers": db.execute(
                "SELECT COUNT(DISTINCT speaker_id) n FROM items").fetchone()["n"],
            "cursors": [dict(r) for r in db.execute(
                "SELECT * FROM cursor ORDER BY source, unit_index")],
            "shards": [dict(r) for r in db.execute("SELECT * FROM shards ORDER BY name")],
        }


def write_progress_json(path: Path, payload: dict) -> None:
    """Ghi atomic: tmp -> fsync -> rename, để file luôn đọc được khi job bị kill."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
