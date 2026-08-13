"""Chạy sinh FAKE dataset từ corpus REAL."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from ..corpus.manifest import Manifest
from ..corpus.schema import LABEL_FAKE, LABEL_REAL, Record, make_utt_id
from ..corpus.spec import AudioSpec, normalize
from ..utils import get_logger, progress, stable_id, stable_rand
from .base import (  # noqa: F401
    KIND_CLONE,
    KIND_TTS,
    Availability,
    Generator,
    available_generators,
    get_generator,
    register,
)
from .texts import FALLBACK_SENTENCES, is_usable, load_texts

# Import để engine tự đăng ký.
from . import kokoro_vi, omnivoice, piper  # noqa: F401,E402

log = get_logger("aidetector.generate")

#: Nguồn ghi vào manifest cho fake sinh từ câu dự phòng (không có real đối chứng).
FALLBACK_SOURCE = "fallback"

# Reference cho voice cloning: OmniVoice khuyến nghị 3–25 giây.
MIN_REF_SECONDS = 3.0
MAX_REF_SECONDS = 25.0


def _resample(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr:
        return np.asarray(audio, dtype=np.float32)
    import librosa

    return librosa.resample(np.asarray(audio, dtype=np.float32), orig_sr=src_sr, target_sr=dst_sr)


def _pick_targets(
    manifest: Manifest, count: int, engine_id: str, min_words: int, max_words: int
) -> list[Record]:
    """Chọn các utterance REAL làm "khuôn" cho fake, rải đều theo speaker.

    Fake sinh ra sẽ mang cùng nội dung và cùng speaker với utterance gốc, nên mỗi
    fake luôn có một real đối chứng — loại bỏ nhiễu do nội dung/người nói.
    """
    pool = [
        r for r in manifest.reals
        if not r.augment and r.text and is_usable(r.text, min_words, max_words)
    ]
    if not pool:
        return []

    by_speaker: dict[str, list[Record]] = defaultdict(list)
    for rec in pool:
        by_speaker[rec.speaker].append(rec)
    for recs in by_speaker.values():
        recs.sort(key=lambda r: r.utt_id)

    rng = stable_rand("pick_targets", engine_id)
    speakers = sorted(by_speaker)
    rng.shuffle(speakers)

    # Round-robin qua speaker để không dồn hết fake vào một vài giọng.
    chosen: list[Record] = []
    cursors = {spk: 0 for spk in speakers}
    while len(chosen) < count:
        progressed = False
        for spk in speakers:
            idx = cursors[spk]
            if idx >= len(by_speaker[spk]):
                continue
            chosen.append(by_speaker[spk][idx])
            cursors[spk] = idx + 1
            progressed = True
            if len(chosen) >= count:
                break
        if not progressed:
            break
    return chosen


def _pick_reference(
    manifest: Manifest, target: Record, spec: AudioSpec
) -> Record | None:
    """Chọn một utterance KHÁC của cùng speaker làm audio tham chiếu để clone."""
    candidates = [
        r for r in manifest.reals
        if r.speaker == target.speaker
        and r.utt_id != target.utt_id
        and not r.augment
        and MIN_REF_SECONDS <= r.duration <= MAX_REF_SECONDS
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda r: r.utt_id)
    rng = stable_rand("ref", target.utt_id)
    return candidates[rng.randrange(len(candidates))]


def build_generator(engine_id: str, device: str, options: dict | None = None) -> Generator:
    cls = get_generator(engine_id)
    status = cls.availability()
    if not status:
        raise RuntimeError(
            f"Engine {engine_id!r} chưa dùng được: {status.reason}."
            + (f" Cài bằng: {status.hint}" if status.hint else "")
        )
    return cls(device=device, **(options or {}))


def generate_fakes(
    manifest: Manifest,
    engine_id: str,
    spec: AudioSpec,
    count: int,
    device: str = "cpu",
    options: dict | None = None,
    voices: list[str] | None = None,
    extra_texts: str | Path | None = None,
    min_words: int = 6,
    max_words: int = 40,
    overwrite: bool = False,
) -> dict:
    """Sinh `count` audio giả bằng một engine, ghi thẳng vào corpus."""
    # Danh sách giọng do config quyết định phải tới được cả generator, để engine
    # không nạp sẵn những giọng sẽ không dùng.
    options = dict(options or {})
    if voices:
        options.setdefault("voices", list(voices))
    gen = build_generator(engine_id, device, options)
    voice_list = list(voices or gen.voices()) or [None]  # type: ignore[list-item]

    targets = _pick_targets(manifest, count, engine_id, min_words, max_words)
    if not targets:
        fallback = load_texts(extra_texts) if extra_texts else FALLBACK_SENTENCES
        log.warning(
            "Corpus REAL không có transcript dùng được — chuyển sang %d câu dự phòng. "
            "Fake sẽ KHÔNG ghép cặp được với real (khác nội dung, khác người nói), nên "
            "mô hình dễ học nhầm sang đặc trưng nội dung. Hãy dùng bộ dữ liệu có "
            "transcript (VIVOS, Common Voice) nếu muốn kết quả đáng tin.",
            len(fallback),
        )
        # `speaker` để RỖNG có chủ đích: câu dự phòng không thuộc về người nói thật
        # nào cả. Bịa ra speaker giả ở đây sẽ khiến bước chia tập speaker-disjoint
        # tưởng chúng là người thật và rải fake khắp các split, trong khi toàn bộ
        # real (ít speaker) dồn vào một split — kết cục là train/val mất hẳn một lớp.
        targets = [
            Record(utt_id="", path="", label=LABEL_REAL, source=FALLBACK_SOURCE,
                   speaker="", text=text)
            for text in (fallback * (count // max(len(fallback), 1) + 1))
        ][:count]

    log.info(
        "Engine %s (%s) · %d mẫu · %d giọng · device=%s",
        engine_id, gen.kind, len(targets), len(voice_list), device,
    )
    gen.ensure_loaded()

    stats: Counter[str] = Counter()
    for i, target in enumerate(progress(targets, total=len(targets), label=f"generate:{engine_id}")):
        voice = voice_list[i % len(voice_list)]
        # Câu dự phòng không có utt gốc ⇒ lấy chính nội dung câu làm khoá, nếu không
        # mọi bản sinh ra sẽ trùng utt_id và đè lên nhau.
        key = f"{target.utt_id}|{voice}" if target.utt_id else f"fallback:{stable_id(target.text)}|{voice}"
        utt_id = make_utt_id(engine_id, target.speaker, key)
        if not overwrite and utt_id in manifest:
            stats["skip_exists"] += 1
            continue

        ref_path = ref_text = None
        if gen.kind == KIND_CLONE:
            ref = _pick_reference(manifest, target, spec) if target.utt_id else None
            if ref is None:
                stats["skip_no_reference"] += 1
                continue
            ref_path = str(manifest.abs_path(ref))
            ref_text = ref.text

        try:
            audio, sample_rate = gen.synthesize(
                target.text, voice=voice, ref_audio=ref_path, ref_text=ref_text
            )
        except Exception as exc:  # noqa: BLE001 — engine bên thứ ba, lỗi rất đa dạng
            log.warning("Sinh lỗi (%s, utt=%s): %s", engine_id, target.utt_id, exc)
            stats["error"] += 1
            continue

        chunks = normalize(_resample(audio, sample_rate, spec.sample_rate), spec)
        if not chunks:
            stats["drop_invalid"] += 1
            continue

        for idx, chunk in enumerate(chunks):
            rec = Record(
                utt_id=utt_id if idx == 0 else f"{utt_id}-{idx}",
                path="",
                label=LABEL_FAKE,
                source=target.source,
                # Giữ nguyên speaker gốc: fake và real đối chứng luôn nằm cùng một
                # phía khi chia tập theo speaker ⇒ không rò rỉ nội dung/giọng nói.
                speaker=target.speaker,
                text=target.text,
                generator=gen.tag(voice),
                ref_utt_id=target.utt_id,
                language=target.language,
                split=target.split,
            )
            manifest.write_audio(rec, chunk, spec)
            stats["kept"] += 1

    gen.unload()
    log.info(
        "Engine %s: tạo %d · lỗi %d · không đạt chuẩn %d · đã có %d",
        engine_id, stats["kept"], stats["error"], stats["drop_invalid"], stats["skip_exists"],
    )
    attempted = stats["kept"] + stats["drop_invalid"] + stats["error"]
    if attempted and stats["drop_invalid"] / attempted > 0.25:
        log.warning(
            "%d/%d mẫu bị loại vì không đạt chuẩn %g–%g giây. Câu quá ngắn thì audio "
            "sinh ra cũng ngắn — chỉnh `generate.min_words` (đang %d) lên cao hơn để "
            "chọn câu dài hơn.",
            stats["drop_invalid"], attempted, spec.min_seconds, spec.max_seconds, min_words,
        )

    keys = ("kept", "error", "drop_invalid", "skip_exists", "skip_no_reference")
    return {"engine": engine_id, **{key: stats[key] for key in keys}}
