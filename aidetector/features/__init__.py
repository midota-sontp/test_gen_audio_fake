"""Trích + cache đặc trưng.

Cache theo `utt_id` (không theo chỉ số) nên thêm/bớt dữ liệu hay đổi cách chia
split đều không làm hỏng cache cũ. Mỗi backbone/layer/pooling có thư mục riêng.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..corpus.manifest import Manifest
from ..corpus.schema import Record
from ..corpus.spec import AudioSpec, load_audio
from ..utils import ensure_dir, get_logger, progress
from .backbones import (  # noqa: F401
    Backbone,
    available_backbones,
    build_backbone,
    get_backbone,
    register,
)

log = get_logger("aidetector.features")


class FeatureStore:
    """Kho embedding trên đĩa: `<root>/<cache_key>/<utt_id>.npy`."""

    def __init__(self, root: str | Path, backbone: Backbone) -> None:
        self.dir = ensure_dir(Path(root) / backbone.cache_key)
        self.backbone = backbone

    def path_for(self, utt_id: str) -> Path:
        return self.dir / f"{utt_id}.npy"

    def has(self, utt_id: str) -> bool:
        return self.path_for(utt_id).exists()

    def save(self, utt_id: str, vector: np.ndarray) -> None:
        np.save(self.path_for(utt_id), vector.astype(np.float32))

    def load(self, utt_id: str) -> np.ndarray:
        return np.load(self.path_for(utt_id))

    def load_many(self, records: list[Record]) -> tuple[np.ndarray, np.ndarray, list[Record]]:
        """Trả (X [N, D], y [N], danh sách bản ghi có embedding)."""
        vectors, labels, kept = [], [], []
        for rec in records:
            if not self.has(rec.utt_id):
                continue
            vectors.append(self.load(rec.utt_id))
            labels.append(rec.label_int)
            kept.append(rec)
        if not vectors:
            return np.zeros((0, self.backbone.output_dim), np.float32), np.zeros((0,), np.int64), []
        return (
            np.stack(vectors).astype(np.float32),
            np.asarray(labels, dtype=np.int64),
            kept,
        )

    def write_meta(self) -> None:
        (self.dir / "meta.json").write_text(
            json.dumps(
                {
                    "backbone": self.backbone.id,
                    "checkpoint": self.backbone.checkpoint,
                    "output_layer": self.backbone.output_layer,
                    "pooling": self.backbone.pooling,
                    "output_dim": self.backbone.output_dim,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )


def extract_features(
    manifest: Manifest,
    backbone: Backbone,
    spec: AudioSpec,
    cache_root: str | Path = "features",
    batch_size: int = 8,
    splits: tuple[str, ...] | None = None,
    overwrite: bool = False,
) -> dict:
    """Trích embedding cho mọi bản ghi (bỏ qua bản đã có trong cache)."""
    store = FeatureStore(cache_root, backbone)
    records = [r for r in manifest if splits is None or r.split in splits]
    todo = [r for r in records if overwrite or not store.has(r.utt_id)]

    log.info(
        "Đặc trưng: %s · layer=%d · pooling=%s · dim=%d",
        backbone.checkpoint, backbone.output_layer, backbone.pooling, backbone.output_dim,
    )
    log.info("Cần trích %d / %d bản ghi (cache: %s)", len(todo), len(records), store.dir)
    if not todo:
        store.write_meta()
        return {"extracted": 0, "cached": len(records), "dim": backbone.output_dim}

    backbone.ensure_loaded()
    done = failed = 0
    for start in progress(range(0, len(todo), batch_size),
                          total=(len(todo) + batch_size - 1) // batch_size,
                          label="extract"):
        batch = todo[start : start + batch_size]
        waveforms, ids = [], []
        for rec in batch:
            try:
                waveforms.append(load_audio(manifest.abs_path(rec), spec.sample_rate))
                ids.append(rec.utt_id)
            except Exception as exc:  # noqa: BLE001
                log.warning("Bỏ %s: %s", rec.utt_id, exc)
                failed += 1
        if not waveforms:
            continue
        vectors = backbone.embed(waveforms)
        for utt_id, vector in zip(ids, vectors):
            store.save(utt_id, vector)
            done += 1

    store.write_meta()
    log.info("Đã trích %d embedding (lỗi %d) → %s", done, failed, store.dir)
    return {"extracted": done, "failed": failed, "cached": len(records) - len(todo),
            "dim": backbone.output_dim, "cache_dir": str(store.dir)}
