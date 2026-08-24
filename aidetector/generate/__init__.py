"""Chạy sinh FAKE dataset từ corpus REAL."""

from __future__ import annotations

import shutil
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable

import numpy as np

from ..corpus.manifest import Manifest
from ..corpus.schema import LABEL_FAKE, LABEL_REAL, Record, make_utt_id
from ..corpus.spec import AudioSpec, load_audio, normalize, save_audio
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
#: Zero-shot cloning giống hay không phụ thuộc trước hết vào ĐỘ DÀI reference: 3 giây
#: là mức tối thiểu chạy được, không phải mức đủ để lấy ra danh tính một người. Nhưng
#: corpus của dự án ép mọi audio về 3–10 giây (thực tế VIVOS trung bình ~3,7 giây), nên
#: một utterance đơn lẻ luôn nằm ở đúng cái mức tối thiểu đó. Ghép nhiều utterance của
#: CÙNG speaker lại cho tới mốc này để reference có đủ chất liệu.
TARGET_REF_SECONDS = 12.0
#: Khoảng lặng chèn giữa hai utterance khi ghép — đánh dấu ranh giới câu cho model,
#: khớp với dấu chấm nối giữa hai transcript.
REF_GAP_SECONDS = 0.3
#: Số tổ hợp reference khác nhau cho mỗi speaker. Nhiều tổ hợp ⇒ fake của cùng một
#: speaker không bị đúc ra từ đúng một khuôn giọng. Nhưng mỗi tổ hợp là một file tạm
#: phải ghi ra đĩa, nên cho mỗi mẫu một tổ hợp riêng là trả giá đĩa mà không được thêm
#: gì: vài tổ hợp/speaker đã đủ đa dạng.
REF_VARIANTS = 4
#: Lưu manifest sau mỗi bao nhiêu mẫu.
#:
#: CLI chỉ gọi `manifest.save()` khi engine chạy XONG, mà sinh vài nghìn mẫu cloning
#: mất nhiều giờ trong phiên Kaggle 9 tiếng. Bị ngắt giữa chừng thì file wav vẫn nằm
#: trên đĩa nhưng manifest không có dòng nào — lần sau chạy lại sẽ làm lại từ đầu. Ghi
#: một CSV vài nghìn dòng là chuyện vài chục milli-giây, rẻ hơn hàng giờ GPU rất nhiều.
SAVE_EVERY = 50


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


def _is_partial_chunk(manifest: Manifest, rec: Record) -> bool:
    """Bản ghi này có phải một MẢNH của file dài bị cắt không?

    `long_policy: split` cắt file dài thành nhiều đoạn, nhưng mọi đoạn đều mang
    NGUYÊN transcript của file gốc. Dùng một mảnh như vậy làm reference là đưa cho
    engine cặp (audio 3 giây, transcript của 12 giây): gióng hàng audio–text sai
    hoàn toàn, và bộ ước lượng độ dài của OmniVoice lấy tốc độ nói = số ký tự
    ref_text / thời lượng ref_audio nên sẽ đọc câu đích nhanh gấp mấy lần bình thường.
    """
    if f"{rec.utt_id}-1" in manifest:  # đoạn 0 của một file đã bị cắt
        return True
    base, sep, tail = rec.utt_id.rpartition("-")
    return bool(sep) and tail.isdigit() and base in manifest


def _pick_reference(manifest: Manifest, target: Record) -> list[Record]:
    """Chọn các utterance KHÁC của cùng speaker, ghép lại làm reference để clone.

    Trả về danh sách theo thứ tự sẽ ghép, tổng độ dài hướng tới `TARGET_REF_SECONDS`
    và không vượt `MAX_REF_SECONDS`. Rỗng nghĩa là speaker này không có reference
    dùng được.
    """
    candidates = [
        r for r in manifest.reals
        if r.speaker == target.speaker
        and not r.augment
        # Không có transcript thì không dùng làm reference: engine sẽ phải tự nhận
        # dạng bằng ASR, chậm và thêm một nguồn sai.
        and r.text.strip()
        and not _is_partial_chunk(manifest, r)
        and MIN_REF_SECONDS <= r.duration <= MAX_REF_SECONDS
    ]
    if not candidates:
        return []
    candidates.sort(key=lambda r: r.utt_id)
    # Thứ tự bốc phụ thuộc (speaker, variant) chứ không phải từng target: các target
    # cùng speaker sẽ dùng chung một nhúm tổ hợp reference, nên file ghép được tái sử
    # dụng thay vì mỗi mẫu ghi ra một file mới.
    variant = stable_rand("ref-variant", target.utt_id).randrange(REF_VARIANTS)
    stable_rand("ref", target.speaker, variant).shuffle(candidates)

    chosen: list[Record] = []
    total = 0.0
    for rec in candidates:
        # Clone từ chính câu đích thì fake chỉ là bản đọc lại của real đối chứng.
        if rec.utt_id == target.utt_id:
            continue
        extra = rec.duration + (REF_GAP_SECONDS if chosen else 0.0)
        if chosen and total + extra > MAX_REF_SECONDS:
            continue
        chosen.append(rec)
        total += extra
        if total >= TARGET_REF_SECONDS:
            break
    return chosen


def _as_sentence(text: str) -> str:
    """Thêm dấu kết câu nếu thiếu, để transcript ghép lại có ranh giới câu rõ ràng."""
    text = text.strip()
    return text if not text or text[-1] in ".!?…,;:" else text + "."


def _materialize_reference(
    manifest: Manifest,
    recs: list[Record],
    spec: AudioSpec,
    work_dir: Path,
    cache: dict[str, tuple[str, str]],
) -> tuple[str, str]:
    """Biến danh sách reference thành (đường dẫn audio, transcript tương ứng).

    Một bản ghi thì dùng thẳng file trong corpus. Nhiều bản ghi thì ghép audio (chèn
    khoảng lặng) và ghép transcript theo ĐÚNG thứ tự đó — hai bên phải khớp nhau,
    nếu không thì reference dài ra mà chất lượng lại tệ hơn cả reference ngắn.
    """
    if len(recs) == 1:
        return str(manifest.abs_path(recs[0])), recs[0].text

    key = "|".join(r.utt_id for r in recs)
    if key in cache:
        return cache[key]

    gap = np.zeros(int(REF_GAP_SECONDS * spec.sample_rate), dtype=np.float32)
    parts: list[np.ndarray] = []
    for rec in recs:
        if parts:
            parts.append(gap)
        parts.append(load_audio(manifest.abs_path(rec), spec.sample_rate))
    # Không chạy `normalize()` ở đây: các đoạn đã chuẩn hoá sẵn, gọi lại chỉ khiến nó
    # bị cắt silence lần hai rồi tách trở lại thành nhiều đoạn ≤ max_seconds.
    path = work_dir / f"ref-{stable_id(key)}.wav"
    save_audio(path, np.concatenate(parts), spec)
    cache[key] = (str(path), " ".join(_as_sentence(r.text) for r in recs))
    return cache[key]


def build_generator(engine_id: str, device: str, options: dict | None = None) -> Generator:
    cls = get_generator(engine_id)
    status = cls.availability()
    if not status:
        raise RuntimeError(
            f"Engine {engine_id!r} chưa dùng được: {status.reason}."
            + (f" Cài bằng: {status.hint}" if status.hint else "")
        )
    return cls(device=device, **(options or {}))


def _report_progress(
    manifest: Manifest,
    targets: list[Record],
    voice_list: list[str | None],
    engine_id: str,
    gen: Generator,
) -> dict:
    """Đếm còn thiếu bao nhiêu, không nạp model.

    Dùng ĐÚNG công thức utt_id của vòng sinh thật, nên con số này là chính xác chứ
    không phải ước lượng — nó là thứ để quyết định còn phải chạy bao lâu nữa.
    """
    by_speaker: dict[str, Counter[str]] = defaultdict(Counter)
    done = todo = 0
    for i, target in enumerate(targets):
        voice = voice_list[i % len(voice_list)]
        marker = f"{voice}|{gen.variant}" if gen.variant else str(voice)
        key = (f"{target.utt_id}|{marker}" if target.utt_id
               else f"fallback:{stable_id(target.text)}|{marker}")
        by_speaker[target.speaker]["all"] += 1
        if make_utt_id(engine_id, target.speaker, key) in manifest:
            done += 1
            by_speaker[target.speaker]["done"] += 1
        else:
            todo += 1

    # Tiến độ theo SPEAKER, vì đó là đơn vị chốt: biết còn bao nhiêu giọng chưa động
    # tới thì ước được còn mấy lần đồng bộ nữa.
    finished = sorted(s for s, c in by_speaker.items() if c["done"] == c["all"])
    partial = sorted(s for s, c in by_speaker.items() if 0 < c["done"] < c["all"])
    untouched = sorted(s for s, c in by_speaker.items() if c["done"] == 0)

    log.info("Engine %s: %d/%d mẫu đã có · còn %d phải sinh",
             engine_id, done, len(targets), todo)
    log.info("Speaker: %d xong · %d dở dang · %d chưa động tới%s",
             len(finished), len(partial), len(untouched),
             f" ({', '.join(untouched[:5])}…)" if len(untouched) > 5
             else (f" ({', '.join(untouched)})" if untouched else ""))
    return {"engine": engine_id, "kept": 0, "error": 0, "drop_invalid": 0,
            "skip_exists": done, "skip_no_reference": 0, "todo": todo,
            "targets": len(targets), "speakers_done": len(finished),
            "speakers_partial": len(partial), "speakers_todo": len(untouched)}


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
    dry_run: bool = False,
    on_speaker_done: "Callable[[str, dict], None] | None" = None,
) -> dict:
    """Sinh `count` audio giả bằng một engine, ghi thẳng vào corpus.

    `dry_run=True` chỉ đếm còn thiếu bao nhiêu rồi thoát — không nạp model, không tốn
    GPU. Câu "đã tới đâu rồi, còn phải chạy bao lâu nữa" phải trả lời được trong vài
    giây, không phải bằng cách chạy thử.

    `on_speaker_done(speaker, stats)` được gọi mỗi khi xong hết phần của một speaker,
    SAU khi manifest đã lưu — đây là mốc an toàn để đẩy dữ liệu ra ngoài. Mốc theo
    speaker chứ không theo số mẫu vì nó là ranh giới có nghĩa: mất kết nối thì phần đã
    đẩy luôn là những giọng hoàn chỉnh, không phải một nhúm mẫu lẻ giữa chừng.
    """
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

    # Thứ tự CHỌN vẫn là round-robin (phân bổ đều cho mọi speaker), chỉ đổi thứ tự
    # CHẠY sang gom theo speaker — `sort` ổn định nên trong mỗi speaker giữ nguyên.
    # Chạy xen kẽ thì không bao giờ có thời điểm nào "xong một giọng" để chốt tiến độ.
    targets.sort(key=lambda t: t.speaker)

    if dry_run:
        return _report_progress(manifest, targets, voice_list, engine_id, gen)

    log.info(
        "Engine %s (%s) · %d mẫu · %d giọng · %d speaker · device=%s",
        engine_id, gen.kind, len(targets), len(voice_list),
        len({t.speaker for t in targets}), device,
    )
    gen.ensure_loaded()

    stats: Counter[str] = Counter()
    # Reference ghép từ nhiều utterance phải nằm ở đâu đó ngoài corpus — nó là đầu vào
    # tạm của engine, không phải dữ liệu của dataset.
    ref_dir = Path(tempfile.mkdtemp(prefix="aidetector-ref-")) if gen.kind == KIND_CLONE else None
    ref_cache: dict[str, tuple[str, str]] = {}
    ref_seconds: list[float] = []
    try:
        _run_batch(
            gen, manifest, targets, voice_list, spec, engine_id, overwrite,
            stats, ref_dir, ref_cache, ref_seconds, on_speaker_done,
        )
    finally:
        if ref_dir is not None:
            shutil.rmtree(ref_dir, ignore_errors=True)

    gen.unload()
    log.info(
        "Engine %s: tạo %d · lỗi %d · không đạt chuẩn %d · đã có %d",
        engine_id, stats["kept"], stats["error"], stats["drop_invalid"], stats["skip_exists"],
    )
    if ref_seconds:
        mean_ref = sum(ref_seconds) / len(ref_seconds)
        log.info("Reference clone: trung bình %.1f giây/mẫu", mean_ref)
        if mean_ref < TARGET_REF_SECONDS * 0.6:
            log.warning(
                "Reference trung bình chỉ %.1f giây (muốn ~%.0f giây). Zero-shot cloning "
                "ở mức này ra giọng KHÔNG giống người nói gốc. Nguyên nhân thường là mỗi "
                "speaker có quá ít bản ghi trong corpus — tăng `--per-speaker` khi ingest "
                "để mỗi speaker có nhiều utterance ghép lại.",
                mean_ref, TARGET_REF_SECONDS,
            )
    if stats["skip_exists"]:
        log.warning(
            "%d mẫu đã có sẵn nên KHÔNG sinh lại. `utt_id` chỉ gồm câu đích + giọng + "
            "biến thể engine, nên mọi thay đổi về CÁCH tổng hợp (xử lý text, knob giải mã, "
            "dtype, phiên bản thư viện) đều vô hình với phép kiểm này: audio cũ sống sót và "
            "trộn lẫn với audio mới dưới cùng một tag, làm hỏng mọi phép đo chất lượng. "
            "Vừa sửa cách tổng hợp thì phải chạy lại kèm --overwrite.",
            stats["skip_exists"],
        )
    if stats["skip_no_reference"]:
        log.warning(
            "%d mẫu bị bỏ vì speaker không có utterance nào dùng làm reference được "
            "(cần bản ghi CÓ transcript, 3–25 giây, không phải mảnh của file bị cắt).",
            stats["skip_no_reference"],
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


def _run_batch(
    gen: Generator,
    manifest: Manifest,
    targets: list[Record],
    voice_list: list[str | None],
    spec: AudioSpec,
    engine_id: str,
    overwrite: bool,
    stats: Counter[str],
    ref_dir: Path | None,
    ref_cache: dict[str, tuple[str, str]],
    ref_seconds: list[float],
    on_speaker_done: "Callable[[str, dict], None] | None" = None,
) -> None:
    """Vòng lặp sinh — tách khỏi `generate_fakes` để thư mục reference tạm luôn được dọn."""
    since_save = 0
    speaker: str | None = None

    def close_speaker() -> None:
        """Chốt một giọng: lưu manifest TRƯỚC rồi mới báo, để hook thấy trạng thái thật."""
        if speaker is None:
            return
        manifest.save()
        log.info("Xong speaker %s · tổng đã tạo %d", speaker, stats["kept"])
        if on_speaker_done is not None:
            on_speaker_done(speaker, dict(stats))

    for i, target in enumerate(progress(targets, total=len(targets), label=f"generate:{engine_id}")):
        if target.speaker != speaker:
            close_speaker()
            speaker = target.speaker
            since_save = 0
        voice = voice_list[i % len(voice_list)]
        # Câu dự phòng không có utt gốc ⇒ lấy chính nội dung câu làm khoá, nếu không
        # mọi bản sinh ra sẽ trùng utt_id và đè lên nhau. Biến thể engine (vd checkpoint
        # khác mặc định) cũng vào khoá, để hai lượt A/B nằm cạnh nhau trong corpus thay
        # vì lượt sau bị bỏ qua vì "đã có". Biến thể mặc định là "" ⇒ khoá y như cũ.
        marker = f"{voice}|{gen.variant}" if gen.variant else str(voice)
        key = f"{target.utt_id}|{marker}" if target.utt_id else f"fallback:{stable_id(target.text)}|{marker}"
        utt_id = make_utt_id(engine_id, target.speaker, key)
        if not overwrite and utt_id in manifest:
            stats["skip_exists"] += 1
            continue

        ref_path = ref_text = None
        if gen.kind == KIND_CLONE:
            assert ref_dir is not None
            refs = _pick_reference(manifest, target) if target.utt_id else []
            if not refs:
                stats["skip_no_reference"] += 1
                continue
            ref_path, ref_text = _materialize_reference(
                manifest, refs, spec, ref_dir, ref_cache
            )
            ref_seconds.append(sum(r.duration for r in refs))

        try:
            audio, sample_rate = gen.synthesize(
                target.text, voice=voice, ref_audio=ref_path, ref_text=ref_text,
                language=target.language or None,
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

        since_save += 1
        if since_save >= SAVE_EVERY:
            manifest.save()
            since_save = 0

    close_speaker()
