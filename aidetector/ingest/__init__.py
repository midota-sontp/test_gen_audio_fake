"""Chạy ingest: nguồn thô → chuẩn corpus."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np

from ..corpus.manifest import Manifest
from ..corpus.schema import LABEL_REAL, Record, make_utt_id
from ..corpus.spec import AudioSpec, normalize, normalize_file
from ..utils import get_logger, progress, slugify
from .base import (  # noqa: F401
    SourceAdapter,
    SourceItem,
    available_adapters,
    detect_adapter,
    get_adapter,
    register,
)

# Import để các adapter tự đăng ký vào registry.
from . import common_voice, folder, hf, vivos  # noqa: F401,E402

log = get_logger("aidetector.ingest")


def _resample(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr:
        return np.asarray(audio, dtype=np.float32)
    import librosa

    return librosa.resample(np.asarray(audio, dtype=np.float32), orig_sr=src_sr, target_sr=dst_sr)


def ingest_source(
    manifest: Manifest,
    adapter: SourceAdapter,
    root: Path | None,
    source_name: str,
    spec: AudioSpec,
    limit: int | None = None,
    per_speaker: int | None = None,
    overwrite: bool = False,
    language: str = "vi",
) -> dict:
    """Duyệt adapter, chuẩn hoá từng utterance rồi ghi vào corpus.

    Idempotent: `utt_id` suy ra từ (source, speaker, key) nên chạy lại chỉ bổ sung
    phần thiếu, trừ khi bật `overwrite`.
    """
    source_name = slugify(source_name)
    stats = Counter()
    speaker_count: Counter[str] = Counter()

    items = adapter.iter_items(root) if root is not None else adapter.iter_items()  # type: ignore[call-arg]
    total = adapter.count_hint(root) if root is not None else None

    for item in progress(items, total=total, label=f"ingest:{source_name}"):
        if limit is not None and stats["kept"] >= limit:
            break
        speaker = slugify(item.speaker or "unknown", 32)
        if per_speaker is not None and speaker_count[speaker] >= per_speaker:
            stats["skip_speaker_full"] += 1
            continue

        base_id = make_utt_id(source_name, speaker, item.key)
        if not overwrite and base_id in manifest:
            stats["skip_exists"] += 1
            speaker_count[speaker] += 1
            continue

        # Chuẩn hoá — có thể trả nhiều đoạn nếu file dài hơn max_seconds.
        if item.audio is not None:
            audio = _resample(item.audio, int(item.sample_rate or spec.sample_rate), spec.sample_rate)
            chunks = normalize(audio, spec)
        elif item.audio_path is not None:
            chunks = normalize_file(item.audio_path, spec)
        else:
            stats["skip_no_audio"] += 1
            continue

        if not chunks:
            stats["drop_invalid"] += 1
            continue

        label = str(item.meta.get("label", LABEL_REAL))
        generator = str(item.meta.get("generator", ""))
        for idx, chunk in enumerate(chunks):
            utt_id = base_id if idx == 0 else f"{base_id}-{idx}"
            rec = Record(
                utt_id=utt_id,
                path="",                       # write_audio sẽ điền đúng vị trí chuẩn
                label=label,
                source=source_name,
                speaker=speaker,
                text=item.text.strip(),
                generator=generator,
                language=item.language or language,
                split=item.split_hint,
            )
            manifest.write_audio(rec, chunk, spec)
            stats["kept"] += 1
        speaker_count[speaker] += 1

    log.info(
        "Ingest %s: giữ %d · bỏ (không đạt chuẩn) %d · đã có sẵn %d · %d speaker",
        source_name, stats["kept"], stats["drop_invalid"], stats["skip_exists"], len(speaker_count),
    )
    # Corpus một speaker không chia speaker-disjoint được — báo ngay ở đây thay vì
    # để lỗi lộ ra ba bước sau dưới dạng "split thiếu một lớp".
    if stats["kept"] and len(speaker_count) < 3:
        log.warning(
            "Chỉ nhận diện được %d speaker — không đủ để chia train/val/test "
            "speaker-disjoint. Có thể adapter %r đọc sai cấu trúc thư mục; kiểm tra cột "
            "`speaker` trong manifest.csv.", len(speaker_count), type(adapter).name,
        )
    if stats["skip_speaker_full"] > 10 * max(stats["kept"], 1):
        log.warning(
            "Bỏ qua %d utterance vì chạm trần --per-speaker=%s trong khi chỉ giữ được %d "
            "— trần này đang siết quá chặt so với số speaker thực tế.",
            stats["skip_speaker_full"], per_speaker, stats["kept"],
        )
    # Trả về đủ mọi khoá (kể cả 0) để phía gọi không phải dò KeyError.
    keys = ("kept", "drop_invalid", "skip_exists", "skip_no_audio", "skip_speaker_full")
    return {
        "source": source_name,
        "speakers": len(speaker_count),
        **{key: stats[key] for key in keys},
    }
