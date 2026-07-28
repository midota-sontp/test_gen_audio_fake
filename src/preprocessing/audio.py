"""Audio preprocessing: mono / 16 kHz / VAD / length-normalize / 16-bit PCM.

Reads the preliminary manifest from stage 00 (data/manifests/raw.csv), writes
processed clips to data/processed/<split>/ and the final per-split manifests
plus a stats.json.
"""
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

from ..monitoring.status import RunStatus, resolve_run_id
from ..utils.config import Config, resolve
from ..utils.logging import get_logger

log = get_logger("preprocess")


def _load_mono_16k(path: Path, sr: int) -> np.ndarray:
    # librosa handles decode + downmix-to-mono + resample in one shot
    y, _ = librosa.load(str(path), sr=sr, mono=True)
    return y.astype(np.float32)


def _vad_concat(y: np.ndarray, top_db: int) -> np.ndarray:
    """Keep voiced segments (energy-based) and concatenate them."""
    if y.size == 0:
        return y
    intervals = librosa.effects.split(y, top_db=top_db)
    if len(intervals) == 0:
        return np.zeros(0, dtype=y.dtype)
    return np.concatenate([y[s:e] for s, e in intervals]).astype(np.float32)


def _fix_length(y: np.ndarray, target_len: int) -> np.ndarray:
    if len(y) >= target_len:
        return y[:target_len]
    return np.pad(y, (0, target_len - len(y)))


def _peak_normalize(y: np.ndarray) -> np.ndarray:
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    return (y / peak * 0.98).astype(np.float32) if peak > 1e-6 else y


def preprocess_all(cfg: Config) -> dict:
    pcfg = cfg.preprocess
    sr = int(pcfg.sample_rate)
    target_len = int(round(float(pcfg.target_seconds) * sr))
    min_len = int(round(float(pcfg.min_seconds) * sr))

    raw_manifest = resolve(cfg.paths.manifest_dir) / "raw.csv"
    processed_dir = resolve(cfg.paths.processed_dir)
    manifest_dir = resolve(cfg.paths.manifest_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    with open(raw_manifest) as f:
        rows = list(csv.DictReader(f))

    status = RunStatus(resolve(cfg.monitoring.status_file), resolve_run_id(cfg))
    status.start_stage("preprocess", total=len(rows))

    per_split_rows: dict[str, list[dict]] = defaultdict(list)
    stats = {
        "n_input": len(rows),
        "errors": 0,
        "dropped_short": 0,
        "dropped_silent": 0,
        "kept": 0,
        "per_split": {},
        "durations_sec": [],
    }

    for r in rows:
        src = resolve(r["path"])
        split = r["split"]
        label = int(r["label"])
        try:
            y = _load_mono_16k(src, sr)
        except Exception as e:  # corrupt / unreadable
            log.warning("skip (load error) %s: %s", src.name, e)
            stats["errors"] += 1
            continue

        if bool(pcfg.vad.enable):
            voiced = _vad_concat(y, int(pcfg.vad.top_db))
            if voiced.size == 0:
                stats["dropped_silent"] += 1
                continue
            y = voiced

        if len(y) < min_len:
            stats["dropped_short"] += 1
            continue

        y = _peak_normalize(_fix_length(y, target_len))

        out = processed_dir / split / src.name
        out.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(out), y, sr, subtype=str(pcfg.bit_depth))
        rel = out.relative_to(resolve(".")).as_posix()
        per_split_rows[split].append(
            {"path": rel, "label": label, "speaker": r["speaker"]}
        )
        stats["kept"] += 1
        stats["durations_sec"].append(round(len(y) / sr, 3))

    for split in ("train", "val", "test"):
        rws = per_split_rows.get(split, [])
        out_csv = manifest_dir / f"{split}.csv"
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["path", "label", "speaker"])
            w.writeheader()
            w.writerows(rws)
        c = Counter(int(x["label"]) for x in rws)
        stats["per_split"][split] = {"real": c[0], "fake": c[1], "total": len(rws)}
        log.info("%-5s -> %d clips (real=%d fake=%d)", split, len(rws), c[0], c[1])

    (manifest_dir / "stats.json").write_text(json.dumps(stats, indent=2))
    log.info(
        "kept=%d errors=%d dropped_short=%d dropped_silent=%d",
        stats["kept"], stats["errors"], stats["dropped_short"], stats["dropped_silent"],
    )
    status.finish_stage("preprocess", kept=stats["kept"], errors=stats["errors"],
                        dropped_short=stats["dropped_short"],
                        dropped_silent=stats["dropped_silent"])
    return stats
