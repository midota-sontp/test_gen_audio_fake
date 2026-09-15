"""Vòng chạy chính: duyệt nguồn -> lọc chất lượng -> chuẩn hoá -> ghi shard -> checkpoint."""
from __future__ import annotations

import hashlib
import signal
from collections import Counter
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import normalize as nz
from . import qc
from .kaggle_sync import KaggleError, KaggleSync, PushThrottle
from .state import State, utcnow, write_progress_json
from .writer import DistWriter
from .sources.registry import build as build_source

# config được ghi vào sqlite -> progress.json -> ĐẨY LÊN DATASET KAGGLE.
# Bất cứ thứ gì giống bí mật đều phải bị che trước khi tới đó.
SECRET_FIELDS = {"hf_token", "kaggle_key", "kaggle_token", "token", "api_key"}


def _redact(key: str, value):
    if key in SECRET_FIELDS and value:
        return f"<đã ẩn, {len(str(value))} ký tự>"
    if isinstance(value, Path):
        return str(value)
    return value


@dataclass
class BuildConfig:
    out_dir: Path
    cache_dir: Path
    sources: list[str]
    min_duration: float = 2.0
    max_duration: float = 10.0
    min_speech_ratio: float = 0.50
    max_clipping_ratio: float = 0.01
    vad_aggressiveness: int = 2
    target_sr: int = 16000
    rms_dbfs: float | None = None        # None = giữ nguyên biên độ gốc (spec Demo v1)
    checkpoint_every: int = 10           # 10 audio nhận được thì chốt một lần
    checkpoint_max_processed: int = 250  # chốt cả khi toàn bị loại, để con trỏ không đứng yên
    checkpoint_max_seconds: float = 60.0
    shard_max_bytes: int = 256 * 1024 * 1024
    limit_per_source: int | None = None
    # Pass 1 của FAKE: chỉ chấm chất lượng rồi ghi sổ ứng viên, không ghi audio.
    index_only: bool = False
    # Pass 2: chỉ xử lý đúng các id đã được chọn, item khác bỏ qua trước khi decode.
    only_ids: set | None = None
    cursor_prefix: str = ""          # tách con trỏ của hai pass trên cùng một nguồn
    source_kwargs: dict = field(default_factory=dict)   # tham số riêng cho từng nguồn
    max_seconds: float | None = None     # ngân sách thời gian, dừng sạch trước khi bị kill
    hf_token: str | None = None
    kaggle: KaggleSync | None = None
    push_index_every: float = 120.0
    push_shard_every: float = 900.0
    verbose_every: int = 100


class Builder:
    def __init__(self, cfg: BuildConfig):
        self.cfg = cfg
        self.out = Path(cfg.out_dir)
        self.state = State(self.out / "state" / "corpus.sqlite")
        self.writer = DistWriter(self.out / "dist", self.state, cfg.shard_max_bytes)
        self.progress_path = self.out / "state" / "progress.json"
        self.seen = self.state.load_hashes()
        self.seen_ids = self.state.load_item_ids()
        if cfg.index_only:           # pass 1 chưa ghi vào bảng hashes -> lấy thêm từ candidates
            for name in cfg.sources:
                self.seen |= self.state.candidate_hashes(name)
        self.throttle = PushThrottle(cfg.push_index_every, cfg.push_shard_every)
        self.started = time.time()
        self.stopping = False
        self._pending = _Pending()
        self._since_ckpt = 0
        self._processed_since = 0
        self._last_ckpt = time.time()
        self._processed_total = 0
        self.state.set_meta("config", {
            k: _redact(k, v) for k, v in vars(cfg).items()
            if not isinstance(v, (KaggleSync, set))})
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, self._on_signal)

    def _on_signal(self, signum, frame) -> None:
        if self.stopping:
            return
        print(f"\n[dừng] nhận tín hiệu {signum}, đang chốt checkpoint cuối...", flush=True)
        self.stopping = True

    # ------------------------------------------------------------------ chạy
    def run(self) -> dict:
        for name in self.cfg.sources:
            if self._out_of_budget():
                break
            self._run_source(name)
        self.checkpoint(force=True, final=True)
        stats = self.state.stats()
        self.state.close()
        self.writer.close()
        return stats

    def _out_of_budget(self) -> bool:
        if self.stopping:
            return True
        if self.cfg.max_seconds and time.time() - self.started >= self.cfg.max_seconds:
            print("[dừng] hết ngân sách thời gian.", flush=True)
            self.stopping = True
            return True
        return False

    def _run_source(self, name: str) -> None:
        src = build_source(name, self.cfg.cache_dir, self.cfg.hf_token,
                           **self.cfg.source_kwargs.get(name, {}))
        print(f"\n=== Nguồn: {name} ===", flush=True)
        src.prepare()
        units = src.units()
        taken = 0
        ckey = f"{self.cfg.cursor_prefix}{name}"
        for unit_index, unit in enumerate(units):
            start_row, done = self.state.cursor_for(ckey, unit, unit_index)
            if done:
                print(f"  [bỏ qua] {unit}: đã xong.", flush=True)
                continue
            print(f"  [chạy] {unit} từ dòng {start_row}", flush=True)
            row = start_row - 1
            for row, item in src.iter_unit(unit, start_row):
                self.process(item, unit=unit, row=row)
                self._pending.cursor[(ckey, unit)] = (row + 1, False)
                self._processed_total += 1
                self._processed_since += 1
                taken += 1
                self.maybe_checkpoint()
                if self._out_of_budget():
                    self.checkpoint(force=True)
                    return
                if self.cfg.limit_per_source and taken >= self.cfg.limit_per_source:
                    self.checkpoint(force=True)
                    return
            self._pending.cursor[(ckey, unit)] = (row + 1, True)
            self.checkpoint(force=True)

    # ------------------------------------------------------- xử lý một audio
    def process(self, item, unit: str = "", row: int = -1) -> bool:
        cfg = self.cfg
        if cfg.only_ids is not None and item.item_id not in cfg.only_ids:
            return False                       # không được chọn -> bỏ qua trước cả decode
        if item.item_id in self.seen_ids or item.item_id in self._pending.item_ids:
            # Cùng một id gặp lại (mirror Common Voice 20: split validation chứa
            # trọn cả train lẫn test). Đếm riêng chứ không im lặng và cũng không
            # tính là "bị loại" — nó là cùng một audio đã xử lý xong rồi.
            self._pending.skips[(item.source, "duplicate_id")] += 1
            return False

        # 1. Decode
        try:
            x, sr = qc.decode(item.audio_bytes)
        except Exception as e:
            return self._reject(item, qc.RejectReason.DECODE, str(e)[:200])
        if x.size == 0:
            return self._reject(item, qc.RejectReason.EMPTY)
        if sr <= 0:
            return self._reject(item, qc.RejectReason.SR_UNKNOWN, str(sr))

        # 2. Duration
        duration = x.shape[0] / sr
        if not (cfg.min_duration <= duration <= cfg.max_duration):
            return self._reject(item, qc.RejectReason.DURATION, f"{duration:.3f}s")

        # 3. Speech ratio (đo trên bản 16 kHz mono, cũng là bản sẽ xuất ra)
        mono16 = nz.to_mono_16k(x, sr, cfg.target_sr)
        speech = qc.speech_ratio(mono16, cfg.vad_aggressiveness)
        if speech < cfg.min_speech_ratio:
            return self._reject(item, qc.RejectReason.SPEECH, f"{speech:.3f}")

        # 4. Clipping (đo trên audio GỐC)
        clip = qc.clipping_ratio(x)
        if clip >= cfg.max_clipping_ratio:
            return self._reject(item, qc.RejectReason.CLIPPING, f"{clip:.4f}")

        # 5. SHA256 trên audio gốc -> dedupe
        h_raw = qc.sha256_samples(x, sr)
        if h_raw in self.seen:
            return self._reject(item, qc.RejectReason.DUP_RAW, h_raw[:16])

        # 6. Normalize -> 16 kHz mono PCM16
        try:
            y = mono16
            gain = 1.0
            if cfg.rms_dbfs is not None:
                y, gain = nz.apply_rms(y, cfg.rms_dbfs)
            pcm16, peak = nz.to_pcm16(y)
            wav = nz.encode_wav(pcm16, cfg.target_sr)
        except Exception as e:
            return self._reject(item, qc.RejectReason.NORMALIZE, str(e)[:200])

        h_norm = hashlib.sha256(pcm16.tobytes()).hexdigest()
        if h_norm in self.seen:
            return self._reject(item, qc.RejectReason.DUP_NORM, h_norm[:16])

        out_duration_ = len(pcm16) / cfg.target_sr
        if cfg.index_only:
            # Ghi sổ ứng viên, chưa ghi audio: số FAKE cuối cùng phụ thuộc số REAL,
            # mà số REAL chỉ biết sau khi quét xong toàn bộ nguồn REAL.
            self.seen.add(h_raw)
            self.seen.add(h_norm)
            self._pending.candidates.append((
                item.item_id, item.source, unit, row, item.speaker_id,
                item.recording_id, out_duration_, _bucket(out_duration_),
                round(speech, 4), round(clip, 6), h_raw, h_norm))
            self._pending.item_ids.add(item.item_id)
            self.seen_ids.add(item.item_id)
            self._since_ckpt += 1
            return True

        # 7. Ghi audio + metadata
        out_duration = out_duration_
        arcname = f"audio/{'fake' if item.label else 'real'}/{item.source}/{item.item_id}.wav"
        shard = self.writer.shard
        self.writer.add(arcname, wav, {
            "id": item.item_id,
            "audio": arcname,
            "label": item.label,
            "speaker_id": item.speaker_id,
            "speaker_known": item.speaker_known,
            "source": item.source,
            "generator": item.generator,
            "recording_id": item.recording_id,
            "duration": round(out_duration, 4),
            "sample_rate": cfg.target_sr,
            "channels": 1,
            "split": None,                      # gán ở bước chia speaker-disjoint
            "text": item.extra.get("text"),
            "gender": item.extra.get("gender"),
            "speech_ratio": round(speech, 4),
            "rms_db": round(qc.rms_dbfs(y), 2),
            "clipping_ratio": round(clip, 6),
            "original_sample_rate": sr,
            "original_channels": int(x.shape[1]),
            "original_split": item.original_split,
            "original_duration": round(duration, 4),
            "norm_gain": round(gain, 6),
            "norm_peak": round(peak, 6),
            "sha256_raw": h_raw,
            "sha256_norm": h_norm,
            "shard": shard,
            "bytes": len(wav),
            "extra": {k: v for k, v in item.extra.items() if k not in ("text", "gender")},
        })

        self.seen.add(h_raw)
        self.seen.add(h_norm)
        self._pending.hashes.append((h_raw, "raw", item.item_id))
        self._pending.hashes.append((h_norm, "norm", item.item_id))
        self._pending.items.append((item.item_id, item.source, shard, arcname,
                                    item.speaker_id, item.recording_id,
                                    out_duration, len(wav), item.label, item.generator))
        self._pending.item_ids.add(item.item_id)
        self.seen_ids.add(item.item_id)
        self._since_ckpt += 1
        return True

    def _reject(self, item, reason: str, detail: str | None = None) -> bool:
        self._pending.rejects.append((item.item_id, item.source, reason, detail))
        self.seen_ids.add(item.item_id)
        return False

    # --------------------------------------------------------- checkpoint
    def maybe_checkpoint(self) -> None:
        cfg = self.cfg
        if (self._since_ckpt >= cfg.checkpoint_every
                or self._processed_since >= cfg.checkpoint_max_processed
                or time.time() - self._last_ckpt >= cfg.checkpoint_max_seconds):
            self.checkpoint()

    def checkpoint(self, force: bool = False, final: bool = False) -> None:
        if not force and not self._pending:
            return
        tar_off, meta_off = self.writer.flush()

        st = self.state
        st.begin()
        try:
            for h, kind, iid in self._pending.hashes:
                st.add_hash(h, kind, iid)
            for rec in self._pending.items:
                st.add_item(*rec)
            for rec in self._pending.candidates:
                st.add_candidate(*rec)
            for rec in self._pending.rejects:
                st.add_reject(*rec)
            for (source, reason), k in self._pending.skips.items():
                st.add_skip(source, reason, k)
            for (source, unit), (nxt, done) in self._pending.cursor.items():
                st.set_cursor(source, unit, nxt, done)
            st.update_shard(self.writer.shard, n_files=self.writer.n_files,
                            bytes=self.writer.nbytes, tar_offset=tar_off,
                            meta_offset=meta_off)
            st.commit()
        except Exception:
            st.rollback()
            raise
        self._pending = _Pending()
        self._since_ckpt = 0
        self._processed_since = 0
        self._last_ckpt = time.time()

        stats = self._write_progress(final=final)
        if self.writer.should_rotate():
            closed = self.writer.close_shard()
            self._push_shard(closed, final=True)
        self._sync(stats, final=final)

    def _write_progress(self, final: bool = False) -> dict:
        stats = self.state.stats()
        elapsed = time.time() - self.started
        payload = {
            "updated_at": utcnow(),
            "status": "done" if final and not self.stopping else
                      ("stopped" if self.stopping else "running"),
            "elapsed_sec": round(elapsed, 1),
            "processed_this_run": self._processed_total,
            "rate_per_sec": round(self._processed_total / elapsed, 2) if elapsed else 0,
            "config": self.state.get_meta("config", {}),
            **stats,
        }
        write_progress_json(self.progress_path, payload)
        return payload

    # --------------------------------------------------------- Kaggle
    def _push_shard(self, shard: str, final: bool = False) -> None:
        k = self.cfg.kaggle
        if not k or not k.enabled:
            return
        if not final and not self.throttle.due("shard"):
            return
        tar = self.out / "dist" / "audio" / f"{shard}.tar"
        meta = self.out / "dist" / "metadata" / f"{shard}.jsonl"
        row = next((s for s in self.state.shards() if s["name"] == shard), None)
        n_files = (row["n_files"] if row else 0) or 0
        if not tar.exists() or n_files == 0:
            return          # shard rỗng vừa tạo -> đừng tạo dataset rác trên Kaggle
        try:
            action = k.push_shard(shard, tar, meta,
                                  f"{shard} @ {utcnow()} ({n_files} files)")
            self.state.begin()
            self.state.update_shard(shard, pushed_at=utcnow(),
                                    push_target=f"{k.owner}/{k.slug_base}-{shard.split('_')[-1]}")
            self.state.commit()
            print(f"  [kaggle] {action} shard {shard}", flush=True)
        except KaggleError as e:
            self.state.rollback()
            print(f"  [kaggle] LỖI đẩy {shard}: {e}", flush=True)

    def _sync(self, stats: dict, final: bool = False) -> None:
        k = self.cfg.kaggle
        if not k or not k.enabled:
            return
        self._push_shard(self.writer.shard, final=final)
        if not (final or self.throttle.due("index")):
            return
        manifest = {
            "slug_base": k.slug_base,
            "owner": k.owner,
            "total_accepted": stats.get("total_accepted"),
            "total_hours": stats.get("total_hours"),
            "shards": [{"name": s["name"], "status": s["status"], "n_files": s["n_files"],
                        "bytes": s["bytes"], "dataset": s["push_target"],
                        "pushed_at": s["pushed_at"]} for s in stats.get("shards", [])],
        }
        try:
            k.push_index(self.progress_path, manifest, f"progress @ {utcnow()}")
        except KaggleError as e:
            print(f"  [kaggle] LỖI đẩy index: {e}", flush=True)


def _bucket(d: float) -> str:
    return "2-4s" if d < 4 else "4-6s" if d < 6 else "6-8s" if d < 8 else "8-10s"


@dataclass
class _Pending:
    items: list = field(default_factory=list)
    candidates: list = field(default_factory=list)
    skips: Counter = field(default_factory=Counter)
    rejects: list = field(default_factory=list)
    hashes: list = field(default_factory=list)
    cursor: dict = field(default_factory=dict)
    item_ids: set = field(default_factory=set)

    def __bool__(self) -> bool:
        return bool(self.items or self.candidates or self.rejects
                    or self.skips or self.cursor)
