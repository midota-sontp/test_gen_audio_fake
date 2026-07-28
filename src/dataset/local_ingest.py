"""Stage 00 (local variant) — build data/manifests/raw.csv from a LOCAL pre-split
dataset instead of downloading from HuggingFace.

Input: `dataset/metadata/metadata.csv` with columns
    audio_path,label,speaker,text,generator,split
  * label: 0 = real (VIVOS), 1 = fake (Fish Speech S2 clone of the same speaker)
  * split: `train` / `test`, already SPEAKER-DISJOINT
    (train = VIVOSSPK* speakers, test = VIVOSDEV* speakers, zero overlap).

The upstream trainer needs a validation split, and the existing dataset only ships
train/test. So we carve a VAL split out of the TRAIN speakers, keeping it
speaker-disjoint from the remaining train (a speaker never appears in two splits).
TEST is passed through untouched (unseen speakers → honest generalization EER).

Because each speaker owns both its real recordings and its fake clones (same
speaker id), splitting by speaker automatically keeps both classes in every split.

Output: `data/manifests/raw.csv` (path,label,speaker,language,split) — the exact
schema stage 01 (preprocess) consumes, so the rest of the pipeline is unchanged.
"""
from __future__ import annotations

import csv
import random
from collections import Counter, defaultdict
from pathlib import Path

from ..monitoring.status import RunStatus, _stamp
from ..utils.config import Config, resolve
from ..utils.logging import get_logger

log = get_logger("dataset")


def _carve_val_from_train(train_rows, val_ratio, rng):
    """Move ~val_ratio of TRAIN samples to a new `val` split, chosen speaker-disjoint.

    Speakers are packed largest-first into val until the val sample quota is met, so
    val is speaker-disjoint from the remaining train and both classes stay present.
    """
    spk_rows: dict[str, list[dict]] = defaultdict(list)
    for r in train_rows:
        spk_rows[r["speaker"]].append(r)

    total = len(train_rows)
    quota = int(round(total * float(val_ratio)))
    speakers = sorted(spk_rows, key=lambda s: len(spk_rows[s]), reverse=True)
    rng.shuffle(speakers)  # reproducible tie-break among equal-size speakers

    val_speakers: set[str] = set()
    filled = 0
    for spk in speakers:
        if filled >= quota:
            break
        val_speakers.add(spk)
        filled += len(spk_rows[spk])

    for r in train_rows:
        r["split"] = "val" if r["speaker"] in val_speakers else "train"


def build_manifest(cfg: Config) -> Path:
    dcfg = cfg.dataset
    rng = random.Random(cfg.seed)

    meta_path = resolve(dcfg.get_path("local_metadata", "dataset/metadata/metadata.csv"))
    data_root = dcfg.get_path("local_root", "dataset")  # prefixed onto audio_path
    val_ratio = float(dcfg.get_path("val_ratio", 0.15))

    manifest_dir = resolve(cfg.paths.manifest_dir)
    manifest_dir.mkdir(parents=True, exist_ok=True)

    # first stage → start a fresh run id (downstream stages read it from status)
    run_id = cfg.get("_run_id") or _stamp()
    cfg["_run_id"] = run_id
    status = RunStatus(resolve(cfg.monitoring.status_file), run_id, reset=True)
    status.start_stage("download", detail=str(meta_path))

    with open(meta_path) as f:
        src = list(csv.DictReader(f))
    if not src:
        raise RuntimeError(f"No rows in {meta_path}")

    rows: list[dict] = []
    for r in src:
        rows.append({
            "path": (Path(data_root) / r["audio_path"]).as_posix(),
            "label": int(r["label"]),
            "speaker": r["speaker"],
            "language": "vi",
            "split": r["split"].strip().lower(),
        })

    train_rows = [r for r in rows if r["split"] == "train"]
    _carve_val_from_train(train_rows, val_ratio, rng)

    manifest_path = manifest_dir / "raw.csv"
    with open(manifest_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["path", "label", "speaker", "language", "split"])
        w.writeheader()
        w.writerows(rows)

    log.info("Wrote %d clips + manifest %s (val carved at ratio %.2f)",
             len(rows), manifest_path, val_ratio)
    _log_split_stats(rows)

    got = Counter(r["label"] for r in rows)
    status.finish_stage("download", real=got[0], fake=got[1], total=len(rows))
    return manifest_path


def _log_split_stats(rows) -> None:
    per = defaultdict(Counter)
    spk = defaultdict(set)
    for r in rows:
        per[r["split"]][r["label"]] += 1
        spk[r["split"]].add(r["speaker"])
    for s in ("train", "val", "test"):
        c = per[s]
        log.info("  %-5s: real=%d fake=%d total=%d | speakers=%d",
                 s, c[0], c[1], c[0] + c[1], len(spk[s]))
    tr, va, te = spk["train"], spk["val"], spk["test"]
    leak = (tr & va) | (tr & te) | (va & te)
    log.info("  speaker leakage across splits: %d %s", len(leak), "(OK)" if not leak else "(!)")
