"""Tầng augmentation — sinh thêm bản biến dạng, GIỮ NGUYÊN bản clean.

Corpus sau bước này chứa cả bản sạch lẫn bản nhiễu/nén của cùng một utterance
(đúng yêu cầu "phải có cả clean/noisy"). Bản augment mang `parent_utt_id` trỏ về
bản gốc và **thừa hưởng split của bản gốc**, nên không bao giờ có chuyện bản
augment nằm ở train còn bản gốc nằm ở test.
"""

from __future__ import annotations

import random
from collections import Counter
from pathlib import Path

import numpy as np

from ..corpus.manifest import Manifest
from ..corpus.schema import Record
from ..corpus.spec import (
    AudioSpec,
    check_quality,
    iter_audio_files,
    load_audio,
    normalize_level,
    sanitize,
)
from ..utils import get_logger, progress, stable_id, stable_rand
from .ops import OPS, has_ffmpeg  # noqa: F401

log = get_logger("aidetector.augment")


class AugmentChain:
    """Chuỗi phép augment lấy ngẫu nhiên theo xác suất khai báo trong config.

    ```yaml
    augment:
      copies: 1
      max_ops: 2
      ops:
        codec:          {p: 0.5}
        background_noise: {p: 0.4, snr_range: [5, 20]}
        reverb:         {p: 0.2}
    ```
    """

    def __init__(
        self,
        ops_config: dict[str, dict],
        max_ops: int = 2,
        noise_dir: str | Path | None = None,
        rir_dir: str | Path | None = None,
    ) -> None:
        unknown = set(ops_config) - set(OPS)
        if unknown:
            raise KeyError(
                f"Phép augment không tồn tại: {', '.join(sorted(unknown))}. "
                f"Hiện có: {', '.join(sorted(OPS))}"
            )
        self.ops_config = ops_config
        self.max_ops = max_ops
        self.noise_files = list(iter_audio_files(noise_dir)) if noise_dir and Path(noise_dir).is_dir() else []
        self.rir_files = list(iter_audio_files(rir_dir)) if rir_dir and Path(rir_dir).is_dir() else []
        if noise_dir and not self.noise_files:
            log.warning("Không tìm thấy file nhiễu trong %s — dùng nhiễu Gauss thay thế", noise_dir)
        if self.noise_files:
            log.info("Nhiễu nền: %d file từ %s", len(self.noise_files), noise_dir)
        if self.rir_files:
            log.info("RIR: %d file từ %s", len(self.rir_files), rir_dir)
        if "codec" in ops_config and not has_ffmpeg():
            log.warning("Không tìm thấy ffmpeg — bỏ qua phép augment `codec` (MP3/AAC)")

    def _extra_kwargs(self, name: str) -> dict:
        if name == "background_noise":
            return {"noise_files": self.noise_files}
        if name == "reverb":
            return {"rir_files": self.rir_files}
        return {}

    def apply(self, audio: np.ndarray, sr: int, rng: random.Random) -> tuple[np.ndarray, str]:
        """Trả (audio đã biến dạng, mô tả chuỗi phép đã dùng)."""
        picked = [name for name, cfg in self.ops_config.items() if rng.random() < float(cfg.get("p", 0.0))]
        rng.shuffle(picked)
        picked = picked[: self.max_ops]

        tags: list[str] = []
        for name in picked:
            params = {k: v for k, v in self.ops_config[name].items() if k != "p"}
            params = {k: (tuple(v) if isinstance(v, list) else v) for k, v in params.items()}
            try:
                audio, tag = OPS[name](audio, sr, rng, **params, **self._extra_kwargs(name))
            except Exception as exc:  # noqa: BLE001
                log.debug("Phép %s lỗi, bỏ qua: %s", name, exc)
                continue
            if tag:
                tags.append(tag)
        return sanitize(audio), "+".join(tags)


def augment_corpus(
    manifest: Manifest,
    spec: AudioSpec,
    chain: AugmentChain,
    copies: int = 1,
    splits: tuple[str, ...] = ("train",),
    seed: int = 42,
    overwrite: bool = False,
) -> dict:
    """Sinh bản augment cho các split chỉ định (mặc định chỉ train).

    Val/test nên giữ sạch để số đo phản ánh đúng dữ liệu thật; muốn đo độ bền
    trước nhiễu thì thêm "test" vào `splits` và xem breakdown theo cột `augment`.
    """
    sources = [
        r for r in manifest
        if not r.augment and (not r.split or r.split in splits)
    ]
    if not sources:
        log.warning("Không có bản ghi nào để augment (splits=%s)", ", ".join(splits))
        return {"created": 0}

    stats: Counter[str] = Counter()
    per_label: Counter[str] = Counter()

    for rec in progress(sources, total=len(sources), label="augment"):
        try:
            audio = load_audio(manifest.abs_path(rec), spec.sample_rate)
        except Exception as exc:  # noqa: BLE001
            log.warning("Không đọc được %s: %s", rec.path, exc)
            stats["read_error"] += 1
            continue

        for copy_idx in range(copies):
            aug_id = f"{rec.utt_id}-aug{copy_idx}"
            if not overwrite and aug_id in manifest:
                stats["skip_exists"] += 1
                continue

            rng = stable_rand(seed, rec.utt_id, copy_idx)
            out, tag = chain.apply(audio.copy(), spec.sample_rate, rng)
            if not tag:                                  # không phép nào trúng xác suất
                stats["skip_no_op"] += 1
                continue

            out = normalize_level(out, spec)             # giữ đúng chuẩn mức âm lượng
            issues = check_quality(out, spec, aug_id)
            if issues:
                log.debug("Bỏ %s: %s", aug_id, "; ".join(i.code for i in issues))
                stats["drop_invalid"] += 1
                continue

            aug = Record(
                **{**rec.to_row(), "utt_id": aug_id, "path": "",
                   "augment": tag, "parent_utt_id": rec.utt_id}
            )
            manifest.write_audio(aug, out, spec)
            stats["created"] += 1
            per_label[rec.label] += 1

    log.info(
        "Augment: tạo %d bản (real=%d, fake=%d) · bỏ %d · đã có %d",
        stats["created"], per_label.get("real", 0), per_label.get("fake", 0),
        stats["drop_invalid"], stats["skip_exists"],
    )
    if per_label and min(per_label.values()) and max(per_label.values()) / min(per_label.values()) > 1.25:
        log.warning(
            "Số bản augment giữa hai lớp lệch nhau (real=%d, fake=%d) — augmentation "
            "có thể trở thành manh mối. Cân nhắc cân bằng lại corpus trước khi augment.",
            per_label.get("real", 0), per_label.get("fake", 0),
        )
    return dict(stats)
