"""Chạy ingest: nguồn thô → chuẩn corpus."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterator

import numpy as np

from ..corpus.manifest import Manifest
from ..corpus.schema import LABEL_REAL, Record, make_utt_id
from ..corpus.spec import AUDIO_EXTENSIONS, AudioSpec, normalize, normalize_file
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
from . import canonical, common_voice, folder, hf, vivos  # noqa: F401,E402

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


def screen_source_file(
    f: Path, spec: AudioSpec | None = None, min_sample_rate: int = 16_000,
) -> str | None:
    """Một file nguồn có đạt chuẩn corpus không — trả lý do bị loại, hoặc None nếu đạt.

    Đánh giá bằng ĐÚNG `check_quality` và ĐÚNG `AudioSpec` mà `validate` dùng, trên bản
    audio đã chuẩn hoá. Nếu chỉ đọc header thì "đạt chuẩn" ở đây và ở `validate` là hai
    thứ khác nhau, và một file lọt cửa này vẫn bị loại ba bước sau.

    Chuẩn hoá ở đây chỉ để QUYẾT ĐỊNH rồi bỏ đi — `convert_flat_recordings` chép file
    GỐC. Nhờ vậy tín hiệu vẫn chỉ đi qua chuỗi chuẩn hoá một lần, ở `ingest`.

    Truyền hàm khác vào `convert_flat_recordings(screen=...)` để thay; chữ ký chỉ cần
    `(Path) -> str | None`, và hàm này dùng được làm nền.
    """
    import soundfile as sf

    from ..corpus.spec import check_quality

    spec = spec or AudioSpec()

    # Sample rate phải xét TRƯỚC khi chuẩn hoá: `load_audio` resample lên 16 kHz nên sau
    # đó không còn dấu vết gì của việc nguồn vốn là 8 kHz — mà phần phổ nâng lên là bịa,
    # và đó đúng là chiều mô hình lấy làm đường tắt.
    try:
        info = sf.info(str(f))
    except Exception as exc:  # noqa: BLE001 — nguồn ngoài, đủ kiểu hỏng
        return f"đọc không được ({type(exc).__name__})"
    if info.samplerate < min_sample_rate:
        return f"sample rate {info.samplerate} < {min_sample_rate}"

    reasons: Counter[str] = Counter()
    chunks = normalize_file(f, spec, reasons)
    if not chunks:
        ma = next(iter(reasons), "không chuẩn hoá được")
        return {"too_short": f"ngắn hơn {spec.min_seconds:g}s sau khi cắt silence",
                "too_long": f"dài hơn {spec.max_seconds:g}s",
                "empty": "rỗng",
                "unreadable": "đọc không được"}.get(ma, ma)
    for c in chunks:
        loi = check_quality(c, spec, str(f))
        if loi:
            return loi[0].code
    return None


def convert_flat_recordings(
    raw: str | Path,
    out: str | Path,
    source: str,
    screen: "Callable[[Path], str | None] | None" = None,
    spec: AudioSpec | None = None,
    min_sample_rate: int = 16_000,
    speaker_from: str = "stem",
) -> dict:
    """Bộ dữ liệu phẳng → cây chuẩn, và chỉ đưa vào những file ĐẠT.

        /dataset_A/56456456456456.mp3
          → đánh giá → đạt → out/real/<source>/56456456456456/56456456456456_001.mp3

    Tên file là danh tính duy nhất có được, nên mỗi bản thu thành một thư mục. File không
    đạt thì không vào cây — đỡ cho `ingest` phải giải mã rồi `validate` phải loại.

    `screen(f) -> str | None` là phép đánh giá: trả chuỗi lý do để loại, `None` để nhận.
    Mặc định là `screen_source_file`, dùng đúng `check_quality` và đúng `AudioSpec` mà
    `validate` dùng — nên "đạt chuẩn" ở đây và ở `validate` là cùng một nghĩa.

    CHỈ chép file, không giải mã: resample, chuẩn mức, cắt độ dài là việc của `ingest`.
    """
    import shutil
    from collections import Counter

    raw, out = Path(raw), Path(out)
    # Phép đánh giá là tham số: mỗi bộ dữ liệu có kiểu rác riêng, và cái gì đáng loại ở
    # nguồn thì chỉ người biết bộ đó mới nói được.
    danh_gia = screen or (lambda f: screen_source_file(f, spec, min_sample_rate))
    bo: Counter[str] = Counter()
    dem = giu = 0
    for f in sorted(p for p in raw.rglob("*") if p.suffix.lower() in AUDIO_EXTENSIONS):
        dem += 1
        ly_do = danh_gia(f)
        if ly_do:
            bo[ly_do] += 1
            continue
        ten = f.stem if speaker_from == "stem" else f.parent.name
        thu_muc = out / "real" / source / slugify(ten, 48)
        thu_muc.mkdir(parents=True, exist_ok=True)
        # Đánh số trong thư mục: nhiều file cùng dồn về một speaker thì không đè nhau.
        so = len([p for p in thu_muc.iterdir() if p.is_file()]) + 1
        dich = thu_muc / f"{slugify(ten, 48)}_{so:03d}{f.suffix.lower()}"
        if not dich.exists():
            shutil.copy(f, dich)
        giu += 1

    log.info("convert_flat_recordings: %d file nguồn · %d đạt · %d loại → %s/real/%s/",
             dem, giu, dem - giu, out, source)
    for ly_do, n in bo.most_common():
        log.info("  loại %d file: %s", n, ly_do)
    if giu < 3:
        log.warning("Chỉ %d bản thu đạt ⇒ %d speaker. Chia tập speaker-disjoint cần ít "
                    "nhất 3 — kiểm lại xem tên file có thật là danh tính người nói không.",
                    giu, giu)
    return {"recordings": dem, "kept": giu, "rejected": dict(bo), "root": str(out)}


def convert_and_verify(
    source: str,
    raw: str | Path,
    convert: "Callable[[Path, Path], None] | None" = None,
    out: str | Path = "converted",
    already: int = 0,
) -> dict:
    """Một bước: hỏi kho → convert nếu chưa có → kiểm đầu vào đạt chuẩn.

    Ba việc này đi cùng nhau nên để cùng một chỗ: tách ra thì rất dễ có đường đi bỏ qua
    phép kiểm — và đường bị bỏ qua đúng là đường hay hỏng nhất (adapter sẵn có đọc sai
    tầng thư mục speaker của một bộ dữ liệu lạ).

    `convert(raw, out)` do người gọi viết vì mỗi bộ dữ liệu lưu một kiểu; nó chỉ dựng lại
    CẤU TRÚC, còn chuẩn hoá audio là việc của `ingest_source`.

    Ném `ValueError` khi đầu vào không đạt chuẩn — đi tiếp chỉ để phát hiện ở bước đắt
    hơn. Trả về `root` là thư mục mà `ingest` nên đọc.
    """
    raw = Path(raw)
    if already:
        log.info("Kho đã có nguồn %r: %d utterance real — bỏ qua convert.", source, already)
        return {"source": source, "root": str(raw), "already": already,
                "skipped": True, "converted": False, "report": None}

    root = raw
    if convert is not None:
        root = Path(out)
        if not root.exists():
            convert(raw, root)
        _kiem_cay_convert(root, source)
        log.info("Đã convert %s → %s", raw, root)

    adapter_cls, score, effective = detect_adapter(root)
    report = ingest_source(Manifest(root / ".khong-dung"), adapter_cls(), effective,
                           source, AudioSpec(), dry_run=True)
    if not report["ok"]:
        raise ValueError("Đầu vào chưa đạt chuẩn:\n"
                         + "\n".join(f"  • {s}" for s in report["problems"]))
    return {"source": source, "root": str(effective), "already": 0,
            "skipped": False, "converted": convert is not None, "report": report}


def _kiem_cay_convert(root: Path, source: str) -> None:
    """Cây do người gọi vừa dựng có đúng ba tầng và đúng tên nguồn không.

    Kiểm trước khi giao cho adapter, vì lỗi ở đây có thông điệp cụ thể hơn nhiều so với
    "adapter không nhận ra thư mục này".
    """
    thu_muc = root / "real"
    wav = [p for p in thu_muc.glob("*/*/*") if p.suffix.lower() in AUDIO_EXTENSIONS] \
        if thu_muc.is_dir() else []
    sai = []
    if not wav:
        sai.append(f"không có file nào ở {root}/real/<nguồn>/<speaker>/")
    elif source not in {p.parent.parent.name for p in wav}:
        # `source` là khoá hỏi kho ở lượt sau; lệch một chữ là phiên sau tra ra "chưa có"
        # rồi convert lại từ đầu.
        sai.append(f"không có thư mục real/{source}/ — tên nguồn phải khớp {source!r}, "
                   f"đang thấy: {sorted({p.parent.parent.name for p in wav})[:4]}")
    if sai:
        raise ValueError("CONVERT dựng ra cây sai chuẩn đầu vào:\n"
                         + "\n".join(f"  • {s}" for s in sai))


def _preview(adapter: SourceAdapter, items, source_name: str, total: int | None) -> dict:
    """Adapter đọc ra gì — đếm mà KHÔNG giải mã một file audio nào.

    Adapter đọc sai cấu trúc (gộp hết vào một speaker, không thấy transcript) là hỏng
    toàn bộ những gì phía sau: chia tập speaker-disjoint cần nhiều giọng, và fake cần
    transcript để sinh. Biết điều đó sau khi đã giải mã 12.000 file là trả giá vô ích.
    """
    dem: Counter[str] = Counter()
    co_text = 0
    mau: list[str] = []
    for item in items:
        speaker = slugify(item.speaker or "unknown", 32)
        dem[speaker] += 1
        if item.text.strip():
            co_text += 1
        if len(mau) < 3:
            mau.append(f"real/{slugify(source_name)}/{speaker}/  ← {item.key}")

    tong = sum(dem.values())
    log.info("Xem trước %s (%s): %d utterance · %d speaker · %d có transcript",
             source_name, adapter.name, tong, len(dem), co_text)
    for dong in mau:
        log.info("  %s", dong)

    # Ba điều kiện này KHÔNG phải cảnh báo mà là điều kiện đủ để đi tiếp. Đầu vào thiếu
    # một trong ba thì mọi bước sau đều vô nghĩa, và phát hiện ở bước sau nghĩa là đã trả
    # tiền giải mã cả bộ dữ liệu — hoặc tệ hơn, trả cả giờ GPU sinh fake.
    sai = []
    if not tong:
        sai.append(f"adapter {adapter.name!r} không đọc ra utterance nào")
    if tong and len(dem) < 3:
        sai.append(f"chỉ {len(dem)} speaker — chia tập speaker-disjoint cần ít nhất 3; "
                   f"adapter {adapter.name!r} có thể đọc sai tầng thư mục speaker")
    if tong and not co_text:
        sai.append("không utterance nào có transcript — fake không ghép cặp được với real")
    for dong in sai:
        log.error("%s", dong)
    if total and abs(total - tong) > max(1, total * 0.02):
        log.warning("count_hint nói %d nhưng duyệt ra %d — adapter bỏ sót hoặc đếm thừa.",
                    total, tong)
    if not sai:
        log.info("✔ đầu vào đạt chuẩn để nạp")
    return {"source": source_name, "adapter": adapter.name, "items": tong,
            "speakers": len(dem), "with_text": co_text, "kept": 0, "already": 0,
            "dry_run": True, "ok": not sai, "problems": sai,
            "drop_invalid": 0, "skip_exists": 0, "skip_no_audio": 0,
            "skip_speaker_full": 0, "drops": {}}


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
    dry_run: bool = False,
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

    if dry_run:
        return _preview(adapter, items, source_name, total)

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
