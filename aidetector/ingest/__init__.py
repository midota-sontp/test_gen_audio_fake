"""Chạy ingest: nguồn thô → chuẩn corpus."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterator

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


def _spread_by_speaker(items: Iterator[SourceItem]) -> Iterator[SourceItem]:
    """Xếp lại thứ tự nguồn theo vòng tròn qua speaker, để `limit` cắt đều.

    Adapter duyệt theo thư mục nên nó trả hết giọng này mới sang giọng khác. Cắt theo
    thứ tự đó ở `--limit` nghĩa là những speaker cuối bảng không có lấy một utterance —
    mà cả pipeline này dựa vào speaker: chia tập là speaker-disjoint, và fake chỉ sinh
    được cho speaker đã có real. Mất speaker ở đây là mất luôn ở mọi bước sau.

    Chỉ gom được khi item mang ĐƯỜNG DẪN. Adapter trả audio trong bộ nhớ (HuggingFace)
    thì gom cả nguồn vào RAM là không chấp nhận được, nên những nguồn đó giữ lối chảy cũ.
    """
    buffered: list[SourceItem] = []
    for item in items:
        if item.audio is not None:
            log.info("Nguồn trả audio trong bộ nhớ — giữ thứ tự tuần tự, không rải "
                     "theo speaker. Dùng --per-speaker nếu cần phủ đều.")
            yield from buffered
            yield item
            yield from items
            return
        buffered.append(item)

    by_speaker: dict[str, list[SourceItem]] = defaultdict(list)
    for item in buffered:
        by_speaker[item.speaker or "unknown"].append(item)
    log.info("Rải %d utterance của %d speaker theo vòng tròn để --limit cắt đều",
             len(buffered), len(by_speaker))

    queues = [iter(recs) for _, recs in sorted(by_speaker.items())]
    while queues:
        for queue in list(queues):
            nxt = next(queue, None)
            if nxt is None:
                queues.remove(queue)
            else:
                yield nxt


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
    drops: Counter[str] = Counter()      # vì sao clip bị loại, không chỉ bao nhiêu
    speaker_count: Counter[str] = Counter()

    # `limit` là TỔNG utterance của nguồn này TRONG CORPUS, không phải "thêm bao nhiêu
    # lần này". Corpus sống qua nhiều phiên, nên `--limit 4000` chạy lại phải ra đúng
    # 4000 chứ không cộng thêm 4000 mỗi lượt. `ref_utt_id` rỗng để chỉ đếm phần do
    # ingest tạo, không đếm fake (fake thừa hưởng cùng `source` với real gốc).
    already = sum(1 for r in manifest
                  if r.source == source_name and not r.augment and not r.ref_utt_id)
    if limit is not None and already >= limit:
        log.info("Nguồn %s đã có %d/%d utterance trong corpus — không nạp thêm.",
                 source_name, already, limit)
        return {"source": source_name, "speakers": 0, "already": already,
                **{k: 0 for k in ("kept", "drop_invalid", "skip_exists",
                                  "skip_no_audio", "skip_speaker_full")}}

    items = adapter.iter_items(root) if root is not None else adapter.iter_items()  # type: ignore[call-arg]
    total = adapter.count_hint(root) if root is not None else None
    if limit is not None:
        items = _spread_by_speaker(items)

    for item in progress(items, total=total, label=f"ingest:{source_name}"):
        if limit is not None and already + stats["kept"] >= limit:
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
            chunks = normalize(audio, spec, drops)
        elif item.audio_path is not None:
            chunks = normalize_file(item.audio_path, spec, drops)
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
        "Ingest %s: giữ %d · bỏ (không đạt chuẩn) %d · chạm trần speaker %d · "
        "đã có sẵn %d · %d speaker",
        source_name, stats["kept"], stats["drop_invalid"], stats["skip_speaker_full"],
        stats["skip_exists"], len(speaker_count),
    )
    # Corpus một speaker không chia speaker-disjoint được — báo ngay ở đây thay vì
    # để lỗi lộ ra ba bước sau dưới dạng "split thiếu một lớp".
    if stats["kept"] and len(speaker_count) < 3:
        log.warning(
            "Chỉ nhận diện được %d speaker — không đủ để chia train/val/test "
            "speaker-disjoint. Có thể adapter %r đọc sai cấu trúc thư mục; kiểm tra cột "
            "`speaker` trong manifest.csv.", len(speaker_count), type(adapter).name,
        )
    # Nguồn bị loại gần nửa thì `--limit` không còn nói lên nguồn cấp được bao nhiêu.
    # Dòng này phải là WARNING: nó là trần cứng cho N_REAL, mà ở dạng INFO nó nằm lẫn
    # giữa hàng chục dòng log và người ta chỉ phát hiện khi thấy corpus hụt.
    processed = stats["kept"] + stats["drop_invalid"]
    if stats["drop_invalid"] > 0.2 * max(processed, 1):
        from ..corpus.spec import RECOVERABLE_SECONDS

        why = " · ".join(f"{k}={v}" for k, v in drops.most_common()) or "không rõ"
        # Chỉ ước trần khi còn phần CHƯA xét. Xét hết rồi thì `kept` chính là trần, và
        # in thêm một con số ngoại suy xấp xỉ nó chỉ làm người đọc tưởng mình còn thiếu.
        chua_xet = stats["skip_speaker_full"] + stats["skip_exists"]
        tran = (f" Cả nguồn chỉ cấp được khoảng {int(total * stats['kept'] / processed):,} "
                "utterance đạt chuẩn, nâng trần cao hơn mức đó cũng không có thêm."
                if total and chua_xet else " Đã xét hết nguồn.")
        log.warning(
            "Bỏ %d/%d utterance ĐÃ XÉT (%.0f%%) — %s.%s",
            stats["drop_invalid"], processed,
            100 * stats["drop_invalid"] / processed, why, tran,
        )
        if drops["too_short"]:
            log.warning(
                "  → %d clip ngắn hơn min_seconds=%.1fs (short_policy=%r). Trong đó %d "
                "clip vẫn dài ≥%.1fs, tức hạ min_seconds xuống %.1f lấy lại được %d "
                "utterance. Đừng đổi short_policy sang 'pad': real bị đệm im lặng mà fake "
                "thì không là tự tạo dấu hiệu phân biệt hai lớp.",
                drops["too_short"], spec.min_seconds, spec.short_policy,
                drops["too_short_but_over_ref"], RECOVERABLE_SECONDS,
                RECOVERABLE_SECONDS, drops["too_short_but_over_ref"],
            )
    # Đây là bộ đếm hay bị bỏ qua nhất: nó không phải "lỗi", nhưng nó là thứ quyết định
    # con số cuối cùng, và nếu không nói ra thì "vì sao chỉ có N?" không trả lời được.
    if stats["skip_speaker_full"] > 0.1 * max(processed, 1):
        extra = int(total * stats["kept"] / processed) - stats["kept"] if total else 0
        log.warning(
            "Chưa xét %d utterance vì speaker đã đủ trần --per-speaker=%s. Đó là lý do "
            "corpus dừng ở %d chứ không phải nguồn đã cạn%s.",
            stats["skip_speaker_full"], per_speaker, stats["kept"],
            f" — bỏ trần sẽ thêm khoảng {extra:,} utterance nữa" if extra > 0 else "",
        )
    # Trả về đủ mọi khoá (kể cả 0) để phía gọi không phải dò KeyError.
    keys = ("kept", "drop_invalid", "skip_exists", "skip_no_audio", "skip_speaker_full")
    return {
        "source": source_name,
        "speakers": len(speaker_count),
        "already": already,
        "drops": dict(drops),
        **{key: stats[key] for key in keys},
    }
