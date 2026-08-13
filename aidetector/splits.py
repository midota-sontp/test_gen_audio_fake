"""Chia train / val / test.

Hai nguyên tắc chống rò rỉ:

1. **Speaker-disjoint** — một speaker chỉ thuộc đúng một split. Nếu không, mô hình
   có thể nhận diện giọng quen thay vì nhận diện dấu vết giả mạo.
2. **Bản augment theo bản gốc** — mọi biến thể của một utterance nằm cùng split
   với bản gốc. Vì fake được sinh từ chính transcript của real (cùng speaker),
   quy tắc 1 đồng thời khoá luôn cả nội dung câu nói.

Tuỳ chọn `holdout_generators` đẩy toàn bộ audio của một engine sang test và cấm nó
xuất hiện ở train — đây là phép đo quan trọng nhất: mô hình có tổng quát hoá sang
engine CHƯA TỪNG THẤY hay không.
"""

from __future__ import annotations

from collections import Counter, defaultdict

from .corpus.manifest import Manifest
from .corpus.schema import LABEL_FAKE, LABEL_REAL
from .utils import get_logger, stable_rand

log = get_logger("aidetector.splits")

SPLITS = ("train", "val", "test")


def assign_splits(
    manifest: Manifest,
    ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 42,
    holdout_generators: list[str] | None = None,
    respect_source_hints: bool = False,
    strict: bool = True,
) -> dict:
    """Gán cột `split` cho mọi bản ghi trong manifest (ghi đè giá trị cũ).

    `strict=True` (mặc định) ném lỗi nếu có split thiếu hẳn một lớp — trạng thái đó
    khiến train/evaluate vô nghĩa, và nếu để lọt thì lỗi chỉ lộ ra ở bước sau dưới
    dạng khó hiểu ("EER nan", "thiếu best.pt").
    """
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"Tổng ratios phải bằng 1, nhận {ratios} = {sum(ratios)}")
    holdout = {g.lower() for g in (holdout_generators or [])}

    records = list(manifest)
    if not records:
        raise ValueError("Corpus rỗng — chạy `ingest` trước khi chia split.")

    # --- 1. Chia speaker ----------------------------------------------------
    speakers = sorted({r.speaker for r in records if r.speaker})
    speaker_split: dict[str, str] = {}

    if respect_source_hints:
        # Nguồn đã chia sẵn (vd VIVOS train/test) — tôn trọng, chỉ carve val từ train.
        hinted: dict[str, Counter] = defaultdict(Counter)
        for r in records:
            if r.speaker and r.split in SPLITS:
                hinted[r.speaker][r.split] += 1
        for spk, counts in hinted.items():
            speaker_split[spk] = counts.most_common(1)[0][0]
        log.info("Dùng split sẵn có từ nguồn cho %d speaker", len(speaker_split))

    unassigned = [s for s in speakers if s not in speaker_split]
    rng = stable_rand("splits", seed)
    rng.shuffle(unassigned)

    if respect_source_hints and speaker_split:
        # Chỉ cần tách val từ nhóm train hiện có.
        train_speakers = [s for s, v in speaker_split.items() if v == "train"]
        rng.shuffle(train_speakers)
        n_val = max(1, int(len(train_speakers) * ratios[1])) if len(train_speakers) > 1 else 0
        for spk in train_speakers[:n_val]:
            speaker_split[spk] = "val"
        for spk in unassigned:
            speaker_split[spk] = "train"
    else:
        n = len(unassigned)
        n_train = int(n * ratios[0])
        n_val = int(n * ratios[1])
        for i, spk in enumerate(unassigned):
            speaker_split[spk] = "train" if i < n_train else ("val" if i < n_train + n_val else "test")

    # --- 2. Gán cho từng bản ghi -------------------------------------------
    by_id = {r.utt_id: r for r in records}
    # Bản ghi không thuộc speaker nào (vd fake sinh từ câu dự phòng) không thể chia
    # theo speaker — rải theo đúng tỉ lệ, tất định theo utt_id.
    homeless = [r for r in records if not (by_id.get(r.parent_utt_id) or r).speaker]
    homeless.sort(key=lambda r: r.utt_id)
    if homeless:
        log.info("%d bản ghi không có speaker — chia theo tỉ lệ thay vì theo speaker",
                 len(homeless))
    homeless_split = {}
    n_train, n_val = int(len(homeless) * ratios[0]), int(len(homeless) * ratios[1])
    for i, rec in enumerate(stable_rand("homeless", seed).sample(homeless, len(homeless))):
        homeless_split[rec.utt_id] = (
            "train" if i < n_train else ("val" if i < n_train + n_val else "test")
        )

    for rec in records:
        # Bản augment luôn bám theo bản gốc.
        parent = by_id.get(rec.parent_utt_id) if rec.parent_utt_id else None
        base = parent or rec
        if not base.speaker:
            rec.split = homeless_split.get(base.utt_id, "train")
            continue
        rec.split = speaker_split.get(base.speaker, "train")

    # --- 3. Engine bị giữ lại chỉ để test ----------------------------------
    moved = 0
    for rec in records:
        if rec.generator and rec.engine.lower() in holdout:
            if rec.split != "test":
                rec.split = "test"
                moved += 1
    if holdout:
        log.info("Giữ engine %s riêng cho test (%d bản ghi chuyển sang test)",
                 ", ".join(sorted(holdout)), moved)

    # --- 4. Kiểm tra rò rỉ --------------------------------------------------
    report = _report(manifest, speaker_split)
    leaks = report["speaker_leaks"]
    if leaks:
        log.error("RÒ RỈ speaker giữa các split: %s", ", ".join(leaks[:10]))
    else:
        log.info("Không có rò rỉ speaker giữa các split ✔")

    broken = []
    for split in SPLITS:
        counts = report["counts"][split]
        n_real, n_fake = counts.get(LABEL_REAL, 0), counts.get(LABEL_FAKE, 0)
        log.info(
            "  %-5s: %5d utt (real=%d fake=%d) · %d speaker",
            split, sum(counts.values()), n_real, n_fake, report["speakers"][split],
        )
        if n_real == 0 or n_fake == 0:
            broken.append(f"{split} (real={n_real}, fake={n_fake})")

    if broken:
        message = (
            "Split thiếu một trong hai lớp: " + "; ".join(broken) + ".\n"
            + _diagnose(manifest, report)
        )
        if strict:
            raise ValueError(message)
        log.warning(message)
    return report


def _diagnose(manifest: Manifest, report: dict) -> str:
    """Đoán nguyên nhân thường gặp để người dùng biết phải sửa ở đâu."""
    n_speakers = len(report["speaker_split"])
    real_speakers = {r.speaker for r in manifest.reals if r.speaker}
    unpaired = [r for r in manifest.fakes if r.ref_utt_id not in manifest]
    hints = []

    if n_speakers < len(SPLITS):
        hints.append(
            f"Chỉ có {n_speakers} speaker — không đủ để chia speaker-disjoint thành "
            f"{len(SPLITS)} tập. Cần bộ dữ liệu nhiều người nói hơn, hoặc bỏ bớt "
            f"`--per-speaker` khi ingest."
        )
    if len(real_speakers) <= 1:
        hints.append(
            f"Toàn bộ audio thật chỉ thuộc {len(real_speakers)} speaker nên dồn hết vào "
            f"một tập. Kiểm tra adapter ingest có nhận đúng speaker không "
            f"(`python -m aidetector validate` và cột `speaker` trong manifest.csv)."
        )
    if unpaired:
        hints.append(
            f"{len(unpaired)}/{len(manifest.fakes)} audio giả không gắn với utterance "
            f"thật nào — thường do corpus REAL thiếu transcript nên `generate` phải "
            f"dùng câu dự phòng. Hãy dùng bộ dữ liệu có transcript."
        )
    return "\n".join(f"  → {h}" for h in hints) if hints else \
        "  → Kiểm tra lại tỉ lệ real/fake trong corpus (`python -m aidetector info`)."


def _report(manifest: Manifest, speaker_split: dict[str, str]) -> dict:
    counts: dict[str, Counter] = {s: Counter() for s in SPLITS}
    speakers: dict[str, set] = {s: set() for s in SPLITS}
    for rec in manifest:
        if rec.split not in counts:
            continue
        counts[rec.split][rec.label] += 1
        if rec.speaker:
            speakers[rec.split].add(rec.speaker)

    leaks = []
    for a in SPLITS:
        for b in SPLITS:
            if a < b:
                for spk in speakers[a] & speakers[b]:
                    leaks.append(f"{spk} ({a}∩{b})")
    return {
        "counts": {s: dict(c) for s, c in counts.items()},
        "speakers": {s: len(v) for s, v in speakers.items()},
        "speaker_leaks": leaks,
        "speaker_split": speaker_split,
    }
