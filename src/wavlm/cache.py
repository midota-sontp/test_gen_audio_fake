"""Extract & cache one frozen-WavLM embedding per processed clip (idempotent)."""
from __future__ import annotations

import csv

import soundfile as sf
import torch

from ..monitoring.status import RunStatus, resolve_run_id
from ..utils.config import Config, resolve
from ..utils.device import select_device
from ..utils.logging import get_logger
from .extractor import WavLMExtractor

log = get_logger("extract")


def _read_manifest(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def _resolve_checkpoint(ckpt: str):
    """Pick the WavLM weights to load, preferring a local copy over a download.

    Order: a bundled ./models/wavlm-base dir (offline) -> the configured value if it
    is itself a local path -> the configured value as an HF id (transformers
    downloads it on first run). Lets the same config run offline here and online
    elsewhere without shipping the 360MB weights through git.
    """
    bundled = resolve("models/wavlm-base")
    if bundled.exists():
        return str(bundled)
    local = resolve(ckpt)
    return str(local) if local.exists() else ckpt


def extract_all(cfg: Config) -> None:
    device = select_device(cfg.extract.device)
    log.info("Extracting embeddings on %s", device)
    extractor = WavLMExtractor(
        _resolve_checkpoint(cfg.paths.wavlm_checkpoint),
        device=device,
        output_layer=cfg.extract.get_path("output_layer"),
        pooling=cfg.extract.pooling,
    )

    manifest_dir = resolve(cfg.paths.manifest_dir)
    emb_root = resolve(cfg.paths.embedding_dir)
    sr = int(cfg.preprocess.sample_rate)
    log_every = int(cfg.extract.batch_log_every)

    splits = {s: _read_manifest(manifest_dir / f"{s}.csv") for s in ("train", "val", "test")}
    total = sum(len(v) for v in splits.values())
    status = RunStatus(resolve(cfg.monitoring.status_file), resolve_run_id(cfg))
    status.start_stage("extract", total=total, detail=str(device))

    processed = 0
    for split, rows in splits.items():
        out_dir = emb_root / split
        out_dir.mkdir(parents=True, exist_ok=True)
        done = 0
        for i, r in enumerate(rows):
            out = out_dir / f"sample_{i:04d}.pt"
            if not out.exists():
                wav, file_sr = sf.read(str(resolve(r["path"])), dtype="float32")
                if file_sr != sr:
                    log.warning("%s sr=%d != %d", r["path"], file_sr, sr)
                emb = extractor.extract(torch.from_numpy(wav))
                torch.save({"emb": emb, "label": int(r["label"]), "path": r["path"]}, out)
            done += 1
            processed += 1
            if done % log_every == 0:
                log.info("  %-5s %d/%d", split, done, len(rows))
            if processed % log_every == 0:
                status.update_stage("extract", processed=processed, current_split=split)
        log.info("%-5s: %d embeddings in %s", split, done, out_dir)
    status.finish_stage("extract", processed=processed)
