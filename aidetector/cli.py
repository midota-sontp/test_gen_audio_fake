"""Giao diện dòng lệnh.

    python -m aidetector <lệnh> [tuỳ chọn]

Các lệnh khớp 1-1 với sơ đồ pipeline:

    ingest    Vietnamese real speech ─→ REAL dataset (chuẩn hoá về chuẩn corpus)
    generate  REAL ─→ voice cloning / TTS ─→ FAKE dataset
    split     chia train/val/test speaker-disjoint
    augment   thêm bản nhiễu / nén cho train (giữ nguyên bản clean)
    features  ─→ WavLM (hoặc backbone khác) ─→ embedding cache
    train     ─→ Classifier
    evaluate  ─→ REAL / FAKE + số đo EER
    detect    chấm điểm một file audio bất kỳ
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .config import Config
from .corpus.manifest import Manifest, find_manifest, find_shards
from .corpus.spec import AudioSpec
from .env import detect_platform, find_kaggle_datasets, free_space_gb, warn_if_constrained
from .utils import get_logger, resolve_device, set_seed, setup_logging, timed

log = get_logger("aidetector.cli")

#: Thứ tự mặc định. `split` phải chạy TRƯỚC `augment` để bản augment chỉ sinh cho
#: train và thừa hưởng đúng split của bản gốc — val/test luôn giữ audio sạch.
STAGES = ("ingest", "generate", "split", "augment", "features", "train", "evaluate")


# --------------------------------------------------------------------- helpers
def _load(args) -> tuple[Config, Manifest, AudioSpec]:
    cfg = Config.load(args.config, args.set or [])
    set_seed(int(cfg.get("seed", 42)))
    spec = AudioSpec.from_config(cfg.section("audio"))
    root = args.corpus or cfg.get("paths.corpus", "corpus")
    warn_if_constrained(root)
    return cfg, Manifest.load(root), spec


def _backbone(cfg: Config, device: str):
    from .features.backbones import build_backbone

    return build_backbone(cfg.section("features.backbone"), device)


def _device(cfg: Config, args, key: str) -> str:
    return resolve_device(getattr(args, "device", None) or cfg.get(key, "auto"))


# ---------------------------------------------------------------------- lệnh
def cmd_ingest(args) -> int:
    from .ingest import detect_adapter, get_adapter, ingest_source

    cfg, manifest, spec = _load(args)

    if args.hf:
        from .ingest.hf import HuggingFaceAdapter

        adapter = HuggingFaceAdapter(args.hf, split=args.hf_split, config=args.hf_config)
        root, name = None, args.name or args.hf.replace("/", "-")
    else:
        if not args.path:
            log.error("Cần --path <thư mục dataset> hoặc --hf <repo_id>")
            return 2
        root = Path(args.path).expanduser().resolve()
        if not root.exists():
            log.error("Không tồn tại: %s", root)
            return 2
        if args.adapter == "auto":
            # `root` có thể bị đẩy xuống sâu hơn: Kaggle/Zenodo hay bọc thêm tầng.
            cls, _, root = detect_adapter(root)
        else:
            cls = get_adapter(args.adapter)
        adapter = cls()
        name = args.name or root.name

    with timed(f"ingest {name}", log):
        result = ingest_source(
            manifest, adapter, root, name, spec,
            limit=args.limit, per_speaker=args.per_speaker,
            overwrite=args.overwrite,
            language=cfg.get("language", "vi"),
            dry_run=args.dry_run,
        )
    if args.dry_run:
        # Không đạt chuẩn đầu vào ⇒ mã khác 0 ⇒ `run()` ở notebook dừng cả phiên. Đi tiếp
        # với đầu vào hỏng chỉ để phát hiện ở bước đắt hơn.
        return 0 if result.get("ok") else 1
    manifest.save()
    print(manifest.summary())
    # Nguồn đã đủ theo `--limit` thì không nạp gì là ĐÚNG, không phải lỗi: phiên sau
    # chạy lại cùng lệnh phải đi tiếp được, chứ không dừng cả notebook ở đây.
    return 0 if (result.get("kept") or result.get("already")) else 1


def _speaker_hook(command: str | None, manifest):
    """Chạy một lệnh shell mỗi khi xong phần của một speaker.

    Đây là mốc an toàn để đẩy corpus ra ngoài: manifest đã lưu, và phần đã xong luôn là
    những GIỌNG hoàn chỉnh chứ không phải một nhúm mẫu lẻ giữa chừng. Lệnh nhận thông
    tin qua biến môi trường, khỏi phải chèn chuỗi vào dòng lệnh.

    Hook hỏng KHÔNG được làm hỏng lượt sinh: mất kết nối lúc đẩy dataset thì corpus vẫn
    còn trên đĩa, còn bỏ dở nhiều giờ GPU thì không lấy lại được.
    """
    if not command:
        return None

    import os
    import subprocess

    def hook(speaker: str, stats: dict) -> None:
        env = {
            **os.environ,
            "AIDETECTOR_SPEAKER": speaker,
            "AIDETECTOR_KEPT": str(stats.get("kept", 0)),
            "AIDETECTOR_CORPUS": str(manifest.root),
        }
        log.info("Hook sau speaker %s: %s", speaker, command)
        done = subprocess.run(command, shell=True, env=env, capture_output=True, text=True)
        out = (done.stdout + done.stderr).strip()
        if out:
            log.info("  %s", out.replace("\n", "\n  ")[-1500:])
        if done.returncode:
            log.warning("Hook trả mã %d — bỏ qua, lượt sinh vẫn chạy tiếp.", done.returncode)

    return hook


def cmd_generate(args) -> int:
    from .generate import available_generators, generate_fakes

    cfg, manifest, spec = _load(args)
    device = _device(cfg, args, "generate.device")

    engines = args.engines or cfg.get("generate.engines", ["piper"])
    if isinstance(engines, str):
        engines = [e.strip() for e in engines.split(",") if e.strip()]

    total_real = len(manifest.reals)
    if not total_real:
        log.error("Corpus chưa có audio REAL. Chạy `ingest` trước.")
        return 2

    # Mặc định: tổng fake ≈ tổng real, chia đều cho các engine.
    if args.count:
        per_engine = args.count
    else:
        ratio = float(cfg.get("generate.fake_to_real_ratio", 1.0))
        per_engine = max(1, int(total_real * ratio / max(len(engines), 1)))

    known = available_generators()
    results: list[dict] = []
    failures: list[str] = []       # engine đã chạy nhưng hỏng
    skipped: list[str] = []        # engine chưa cài, không hề được thử
    unknown: list[str] = []
    for engine_id in engines:
        if engine_id not in known:
            log.error("Bỏ qua engine lạ %r (hiện có: %s)", engine_id, ", ".join(sorted(known)))
            unknown.append(engine_id)
            continue
        status = known[engine_id].availability()
        if not status:
            log.warning("Bỏ qua %s — %s%s", engine_id, status.reason,
                        f". Cài: {status.hint}" if status.hint else "")
            skipped.append(f"{engine_id}: {status.reason}")
            continue
        try:
            with timed(f"generate {engine_id} ({per_engine} mẫu)", log):
                results.append(generate_fakes(
                    manifest, engine_id, spec, per_engine, device=device,
                    options=cfg.section(f"generate.options.{engine_id}"),
                    voices=cfg.get(f"generate.voices.{engine_id}"),
                    extra_texts=cfg.get("generate.texts_file"),
                    min_words=int(cfg.get("generate.min_words", 6)),
                    max_words=int(cfg.get("generate.max_words", 40)),
                    overwrite=args.overwrite,
                    dry_run=args.dry_run,
                    on_speaker_done=_speaker_hook(args.after_speaker, manifest),
                ))
        except Exception as exc:  # noqa: BLE001 — engine bên thứ ba, hỏng đủ kiểu
            # Một engine hỏng KHÔNG được xoá công của các engine khác: dữ liệu đã
            # sinh vẫn giữ nguyên, và các engine còn lại vẫn chạy tiếp.
            log.error("Engine %s thất bại: %s", engine_id, exc)
            failures.append(f"{engine_id}: {exc}")
        manifest.save()          # lưu sau mỗi engine để không mất công nếu ngắt giữa chừng

    print(manifest.summary())
    produced = sum(r.get("kept", 0) for r in results)
    for label, items in (("Engine chạy lỗi", failures), ("Engine chưa cài, đã bỏ qua", skipped)):
        if items:
            log.warning("%s:\n%s", label, "\n".join(f"  • {i}" for i in items))

    if unknown:
        return 2                      # sai tên engine là lỗi cấu hình, phải báo
    if produced:
        return 0
    if failures:
        log.error("Mọi engine được thử đều lỗi — không sinh được audio giả nào.")
        return 1
    if results:
        # Engine ĐÃ chạy mà không sinh gì là chuyện bình thường: `--dry-run` chỉ đếm, và
        # phiên nối tiếp thì phần cần sinh có thể đã đủ. Báo "chưa cài engine" ở đây là
        # chẩn đoán sai, đẩy người dùng đi tìm một lỗi không tồn tại.
        existing = sum(r.get("skip_exists", 0) for r in results)
        if args.dry_run:
            log.info("Chỉ đếm (--dry-run) — chưa sinh gì.")
        else:
            log.info("Không có mẫu nào cần sinh thêm · %d mẫu đã có sẵn.", existing)
        return 0
    # Chưa cài engine nào thì đây là bỏ qua, không phải hỏng: các engine tuỳ chọn
    # (vd OmniVoice cần GPU) vắng mặt không được làm dừng cả pipeline.
    log.warning(
        "Không engine nào trong %s được cài — bỏ qua bước sinh fake. "
        "Xem `python -m aidetector info` để biết cách cài.", ", ".join(engines),
    )
    return 0


def cmd_augment(args) -> int:
    from .augment import AugmentChain, augment_corpus

    cfg, manifest, spec = _load(args)
    chain = AugmentChain(
        ops_config=cfg.section("augment.ops"),
        max_ops=int(cfg.get("augment.max_ops", 2)),
        noise_dir=cfg.get("augment.noise_dir"),
        rir_dir=cfg.get("augment.rir_dir"),
    )
    splits = tuple(args.splits or cfg.get("augment.splits", ["train"]))
    with timed("augment", log):
        augment_corpus(
            manifest, spec, chain,
            copies=args.copies or int(cfg.get("augment.copies", 1)),
            splits=splits, seed=int(cfg.get("seed", 42)), overwrite=args.overwrite,
        )
    manifest.save()
    print(manifest.summary())
    return 0


def cmd_split(args) -> int:
    from .splits import assign_splits

    cfg, manifest, _ = _load(args)
    ratios = tuple(cfg.get("splits.ratios", [0.7, 0.15, 0.15]))
    with timed("split", log):
        report = assign_splits(
            manifest, ratios=ratios, seed=int(cfg.get("seed", 42)),
            holdout_generators=args.holdout or cfg.get("splits.holdout_generators", []),
            respect_source_hints=bool(cfg.get("splits.respect_source_hints", False)),
        )
    manifest.save()
    return 1 if report["speaker_leaks"] else 0


def cmd_features(args) -> int:
    from .features import extract_features

    cfg, manifest, spec = _load(args)
    device = _device(cfg, args, "features.device")
    backbone = _backbone(cfg, device)
    with timed("features", log):
        extract_features(
            manifest, backbone, spec,
            cache_root=cfg.get("paths.features", "features"),
            batch_size=int(cfg.get("features.batch_size", 8)),
            overwrite=args.overwrite,
        )
    return 0


def cmd_train(args) -> int:
    from .train import train

    cfg, manifest, _ = _load(args)
    device = _device(cfg, args, "train.device")
    backbone = _backbone(cfg, device)
    with timed("train", log):
        train(
            manifest, backbone, cfg,
            cache_root=cfg.get("paths.features", "features"),
            checkpoint_dir=cfg.get("paths.checkpoints", "checkpoints"),
            report_dir=cfg.get("paths.reports", "reports"),
            device=device,
        )
    return 0


def cmd_evaluate(args) -> int:
    from .evaluate import evaluate

    cfg, manifest, _ = _load(args)
    device = _device(cfg, args, "train.device")
    backbone = _backbone(cfg, device)
    with timed("evaluate", log):
        evaluate(
            manifest, backbone,
            checkpoint=args.checkpoint or Path(cfg.get("paths.checkpoints", "checkpoints")) / "best.pt",
            cache_root=cfg.get("paths.features", "features"),
            report_dir=cfg.get("paths.reports", "reports"),
            split=args.split, device=device,
        )
    return 0


def cmd_detect(args) -> int:
    from .detect import Detector

    cfg = Config.load(args.config, args.set or [])
    detector = Detector(
        checkpoint=args.checkpoint or Path(cfg.get("paths.checkpoints", "checkpoints")) / "best.pt",
        device=args.device or cfg.get("train.device", "auto"),
        threshold=args.threshold,
    )
    results = detector.predict_many(args.files)
    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
    else:
        for res in results:
            if "error" in res:
                print(f"✖ {res['path']}: {res['error']}")
                continue
            icon = "🔴" if res["label"] == "FAKE" else "🟢"
            print(f"{icon} {res['label']:<4} · P(fake)={res['score_fake']:.3f} · "
                  f"{res['duration']}s · {res['path']}")
            if res.get("warning"):
                print(f"   ⚠ {res['warning']}")
    return 0


def cmd_validate(args) -> int:
    """Soi toàn corpus xem có đúng chuẩn audio không."""
    from collections import Counter

    from .corpus.spec import check_quality, load_audio

    _, manifest, spec = _load(args)
    if not len(manifest):
        log.error("Corpus rỗng.")
        return 2

    # Bản ghi đã soi qua ĐÚNG chuẩn này rồi thì bỏ qua: soi lại là đọc lại từng file
    # audio của cả corpus mỗi phiên, trong khi phần đã duyệt không thể đổi kết quả.
    # Đổi chuẩn ⇒ vân tay đổi ⇒ mọi bản ghi tự động được soi lại.
    van_tay = spec.check_fingerprint
    can_soi = [r for r in manifest if args.recheck or r.checked != van_tay]
    da_duyet = len(manifest) - len(can_soi)
    log.info("Kiểm tra %d bản ghi theo chuẩn: %s", len(can_soi), spec.describe())
    if da_duyet:
        log.info("Bỏ qua %d bản ghi đã duyệt theo đúng chuẩn này (dùng --recheck để soi lại)",
                 da_duyet)
    issues: Counter[str] = Counter()
    broken: list[str] = []
    vua_duyet = 0
    from .utils import progress

    for rec in progress(can_soi, total=len(can_soi), label="validate"):
        path = manifest.abs_path(rec)
        if not path.exists():
            issues["missing_file"] += 1
            broken.append(rec.utt_id)
            continue
        try:
            audio = load_audio(path, spec.sample_rate)
        except Exception:  # noqa: BLE001
            issues["unreadable"] += 1
            broken.append(rec.utt_id)
            continue
        loi = check_quality(audio, spec, rec.utt_id)
        for issue in loi:
            issues[issue.code] += 1
            broken.append(rec.utt_id)
        loi_schema = rec.validate()
        for err in loi_schema:
            issues[f"schema:{err.split(':')[0]}"] += 1
            # Trước đây lỗi schema chỉ được ĐẾM: `--fix` không dọn được nó, nên corpus
            # dính một bản ghi sai schema là `validate` đỏ vĩnh viễn và ô A4 dừng
            # notebook sau nhiều giờ sinh, không có đường ra ngoài sửa tay manifest.
            broken.append(rec.utt_id)
        # Đóng dấu chỉ khi đạt CẢ HAI. Đóng dấu một bản ghi sai schema là lần sau nó
        # được bỏ qua, không ai soi nữa, và `--fix` không bao giờ chạm tới nó.
        if not loi and not loi_schema:
            rec.checked = van_tay      # đóng dấu để phiên sau khỏi đọc lại file này
            vua_duyet += 1

    # Dấu vừa đóng phải xuống đĩa, nếu không phiên sau lại đọc lại toàn bộ.
    if vua_duyet:
        manifest.save()
        log.info("Đã đóng dấu %d bản ghi đạt chuẩn", vua_duyet)

    print(manifest.summary())
    if issues:
        print("\nVấn đề phát hiện được:")
        for code, count in issues.most_common():
            print(f"  {code:<20} {count}")
        hong = set(broken)
        # Giữ lại bản ghi TRƯỚC khi loại, để `--prune-files` còn biết file nằm đâu.
        cu_the = {u: manifest.get(u) for u in hong}
        print(f"\n{len(hong)} bản ghi không đạt chuẩn.")
        if args.fix:
            # Hỏng cả một mảng lớn thì nguyên nhân nằm ở chuỗi chuẩn hoá, ở adapter, hay
            # ở chính spec — không phải vài file xấu. Tự loại lúc đó là xoá corpus mà
            # tưởng mình đang dọn dẹp.
            ty_le = len(hong) / max(len(manifest), 1)
            if ty_le > 0.2 and not args.force:
                log.error(
                    "%.0f%% corpus không đạt chuẩn — mức đó là lỗi hệ thống, không phải "
                    "dữ liệu xấu lẻ tẻ. KHÔNG tự loại; tìm nguyên nhân trước, hoặc "
                    "--force nếu đây đúng là chủ ý.",
                    100 * ty_le,
                )
                return 1
            # Bản ghi TỪNG đạt một chuẩn khác mà giờ không đạt chuẩn hiện tại nghĩa
            # là chuẩn vừa đổi, không phải dữ liệu vừa hỏng. Siết `MIN_SECONDS` rồi để
            # `--fix` lặng lẽ xoá phần không còn lọt là mất dữ liệu mà không ai quyết.
            tung_dat = [u for u in hong
                        if (r := manifest.get(u)) is not None and r.checked
                        and r.checked != van_tay]
            if tung_dat and not args.force:
                log.error(
                    "%d bản ghi từng đạt chuẩn KHÁC giờ không đạt chuẩn hiện tại (%s). "
                    "Đổi chuẩn là một quyết định, không phải dọn rác — thêm --force nếu "
                    "thật sự muốn loại chúng.", len(tung_dat), spec.describe(),
                )
                return 1
            for utt_id in hong:
                manifest.remove(utt_id)
            manifest.save()
            print(f"Đã loại {len(hong)} bản ghi khỏi manifest (--fix).")
            if args.prune_files:
                # Không xoá mặc định: bản ghi bị loại vì chuẩn có thể được nhận lại khi
                # chuẩn đổi, và file đã xoá thì phải ingest lại từ nguồn. Nhưng để đó thì
                # mỗi phiên `ingest` nạp lại rồi cấp số MỚI, tích dần file mồ côi.
                xoa = 0
                for utt_id in hong:
                    r = cu_the.get(utt_id)
                    if r is None:
                        continue
                    f = Path(manifest.root) / r.path
                    if f.exists():
                        f.unlink()
                        xoa += 1
                print(f"Đã xoá {xoa} file audio (--prune-files).")
            # Đã dọn xong thì corpus đạt chuẩn — trả 0, nếu không phía gọi (notebook)
            # sẽ dừng cả phiên ngay sau khi việc đã được sửa.
            return 0
        return 1
    print("\n✔ Toàn bộ corpus đạt chuẩn.")
    return 0


def cmd_progress(args) -> int:
    """Đã sinh fake xong tới speaker nào — trạng thái để nối tiếp giữa các phiên.

    Đơn vị là SPEAKER vì đó là ranh giới `generate` chốt tiến độ (xem `--after-speaker`).
    Đích của một speaker là số utterance real của giọng đó **đủ điều kiện làm khuôn**:
    có transcript và lọt bộ lọc số từ. Lấy mẫu số là mọi real thì tiến độ không bao giờ
    tới 100% và người đọc tưởng còn nợ.

    Ghi ra JSON để đẩy kèm dataset: nó vài KB, đọc được ngay trên trang dataset, và là
    thứ phiên sau so sánh trước khi quyết định có được đẩy đè hay không.
    """
    from collections import defaultdict

    from .generate.texts import is_usable

    cfg, manifest, _ = _load(args)
    min_w = int(cfg.get("generate.min_words", 6))
    max_w = int(cfg.get("generate.max_words", 40))

    khuon: dict[str, list] = defaultdict(list)
    for rec in manifest.reals:
        if not rec.augment and rec.text and is_usable(rec.text, min_w, max_w):
            khuon[rec.speaker].append(rec)
    co_fake = {f.ref_utt_id for f in manifest.fakes if not f.augment}

    xong: list[str] = []
    dang_do: dict[str, dict] = {}
    chua: list[str] = []
    for spk in sorted(khuon):
        dich = len(khuon[spk])
        da = sum(1 for r in khuon[spk] if r.utt_id in co_fake)
        if da >= dich:
            xong.append(spk)
        elif da:
            dang_do[spk] = {"fake": da, "target": dich}
        else:
            chua.append(spk)

    tong_dich = sum(len(v) for v in khuon.values())
    tong_da = sum(1 for recs in khuon.values() for r in recs if r.utt_id in co_fake)
    state = {
        "dataset_records": len(manifest),
        "real": len(manifest.reals),
        "fake": len(manifest.fakes),
        "targets_total": tong_dich,
        "targets_done": tong_da,
        "speakers_total": len(khuon),
        "speakers_done": xong,
        "speakers_partial": dang_do,
        "speakers_todo": chua,
        "engines": {},
        # Tách theo NGUỒN: phiên sau hỏi "dataset_A đã có trên kho chưa, duyệt tới đâu"
        # mà không phải tải cả manifest về đọc.
        "by_source": {},
    }
    for rec in manifest:
        if rec.augment:
            continue
        o = state["by_source"].setdefault(rec.source or "?",
                                          {"real": 0, "fake": 0, "approved": 0})
        o["fake" if rec.is_fake else "real"] += 1
        if rec.checked:
            o["approved"] += 1
    for f in manifest.fakes:
        if not f.augment:
            state["engines"][f.generator] = state["engines"].get(f.generator, 0) + 1

    print(f"Tiến độ sinh fake: {tong_da}/{tong_dich} khuôn"
          f" ({100 * tong_da / max(tong_dich, 1):.0f}%)")
    print(f"Speaker: {len(xong)} xong · {len(dang_do)} dở dang · {len(chua)} chưa động tới"
          f"  (tổng {len(khuon)})")
    if chua:
        print("  chưa động tới: " + ", ".join(chua[:8]) + ("…" if len(chua) > 8 else ""))
    if dang_do:
        vai = list(dang_do.items())[:5]
        print("  dở dang: " + " · ".join(f"{s} {v['fake']}/{v['target']}" for s, v in vai))
    for ten, o in sorted(state["by_source"].items()):
        print(f"  nguồn {ten:<24} real {o['real']:>6} · fake {o['fake']:>6}"
              f" · đã duyệt {o['approved']:>6}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(state, ensure_ascii=False, indent=1),
                                  encoding="utf-8")
        print(f"→ {args.out}")
    return 0


def cmd_migrate(args) -> int:
    """Dọn corpus về đúng cấu trúc thư mục hiện hành."""
    _, manifest, _ = _load(args)
    if not len(manifest):
        log.error("Corpus rỗng.")
        return 2
    with timed("migrate", log):
        result = manifest.migrate_layout(dry_run=args.dry_run)
    if not args.dry_run and result["moved"]:
        manifest.save()
    print(f"Đúng chỗ sẵn: {result['kept']} · chuyển: {result['moved']}"
          f" · nhận lại từ lượt ngắt: {result['resumed']} · thiếu file: {result['missing']}")
    return 0


def cmd_pack(args) -> int:
    """Gói corpus thành một zip để chuyển giữa các phiên Kaggle/Colab."""
    from .packaging import pack_corpus

    cfg = Config.load(args.config, args.set or [])
    root = args.corpus or cfg.get("paths.corpus", "corpus")
    out = args.out or Path(detect_platform().work_dir) / "corpus.zip"
    with timed(f"pack → {out}", log):
        pack_corpus(root, out, compress=args.compress)
    return 0


def cmd_unpack(args) -> int:
    from .packaging import unpack_corpus

    cfg = Config.load(args.config, args.set or [])
    root = args.corpus or cfg.get("paths.corpus", "corpus")
    with timed(f"unpack {args.archive}", log):
        manifest = unpack_corpus(args.archive, root, overwrite=args.overwrite)
    print(manifest.summary())
    return 0


def cmd_info(args) -> int:
    """Liệt kê mọi thành phần cắm rời đang có + tình trạng cài đặt."""
    from .features.backbones import available_backbones
    from .generate import available_generators
    from .ingest import available_adapters
    from .models import available_heads
    from .augment.ops import OPS, has_ffmpeg

    cfg = Config.load(args.config, args.set or [])
    spec = AudioSpec.from_config(cfg.section("audio"))
    platform = detect_platform()
    print(f"ai-detector {__version__}")
    print(f"Chuẩn audio : {spec.describe()}")
    print(f"Thiết bị    : {resolve_device('auto')}   ffmpeg: {'có' if has_ffmpeg() else 'KHÔNG'}")
    print(f"Môi trường  : {platform.describe()} · trống {free_space_gb(platform.work_dir):.1f} GB")
    if platform.name == "kaggle":
        mounted = find_kaggle_datasets()
        print("Dataset mount: " + (", ".join(p.name for p in mounted) if mounted
                                   else "(chưa add dataset nào vào notebook)"))

    print("\nNguồn dữ liệu (ingest):")
    for name, cls in sorted(available_adapters().items()):
        print(f"  {name:<16} {cls.description}")

    print("\nEngine sinh fake (generate):")
    for name, cls in sorted(available_generators().items()):
        status = cls.availability()
        mark = "✔" if status else "✖"
        note = "" if status else f"  ({status.reason} → {status.hint})"
        print(f"  {mark} {name:<14} [{cls.kind}] {cls.description}{note}")

    print("\nPhép augment:")
    print("  " + ", ".join(sorted(OPS)))

    print("\nBackbone:")
    for name, cls in sorted(available_backbones().items()):
        print(f"  {name:<16} {cls.description} (mặc định: {cls.default_checkpoint})")

    print("\nClassifier head:")
    for name, cls in sorted(available_heads().items()):
        print(f"  {name:<16} {cls.description}")

    root = args.corpus or cfg.get("paths.corpus", "corpus")
    # Nhận cả corpus tách theo bộ (`<bộ>/metadata.csv`) và cấu trúc gộp cũ ở gốc.
    if find_shards(Path(root)) or find_manifest(Path(root)):
        print()
        print(Manifest.load(root).summary())
    return 0


def cmd_run(args) -> int:
    """Chạy nhiều stage liên tiếp."""
    stages = args.stages or list(STAGES)
    unknown = [s for s in stages if s not in STAGES]
    if unknown:
        log.error("Stage không tồn tại: %s. Hợp lệ: %s", ", ".join(unknown), " ".join(STAGES))
        return 2
    skip = set(args.skip or [])
    handlers = {
        "ingest": cmd_ingest, "generate": cmd_generate, "augment": cmd_augment,
        "split": cmd_split, "features": cmd_features, "train": cmd_train,
        "evaluate": cmd_evaluate,
    }
    for stage in stages:
        if stage in skip:
            log.info("⤼ bỏ qua %s", stage)
            continue
        if stage == "ingest" and not (args.path or args.hf):
            log.info("⤼ bỏ qua ingest (không có --path/--hf)")
            continue
        log.info("═" * 60)
        log.info("STAGE: %s", stage.upper())
        code = handlers[stage](args)
        if code != 0:
            log.error("Stage %s trả mã lỗi %d — dừng pipeline.", stage, code)
            return code
    return 0


# ---------------------------------------------------------------------- parser
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m aidetector",
        description="Phát hiện giọng nói giả (deepfake audio) tiếng Việt",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--version", action="version", version=f"ai-detector {__version__}")

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("-c", "--config", default="configs/default.yaml", help="file config YAML")
    common.add_argument("--set", action="append", metavar="KEY=VALUE",
                        help="ghi đè config, vd: --set train.lr=1e-4")
    common.add_argument("--corpus", help="thư mục corpus (mặc định lấy từ config)")
    common.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    common.add_argument("--overwrite", action="store_true", help="ghi đè dữ liệu đã có")

    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("ingest", parents=[common], help="nạp dataset thật về chuẩn corpus")
    p.add_argument("path", nargs="?", help="thư mục dataset")
    p.add_argument("--path", dest="path_opt", help=argparse.SUPPRESS)
    p.add_argument("--adapter", default="auto", help="auto | vivos | common_voice | folder | labeled_folder")
    p.add_argument("--name", help="tên nguồn ghi vào manifest (mặc định: tên thư mục)")
    p.add_argument("--hf", help="repo_id trên HuggingFace, vd: AILAB-VNUHCM/vivos")
    p.add_argument("--hf-split", default="train")
    p.add_argument("--hf-config", default=None)
    p.add_argument("--limit", type=int, help="tối đa bao nhiêu utterance")
    p.add_argument("--per-speaker", type=int, help="tối đa bao nhiêu utterance mỗi speaker")
    p.add_argument("--dry-run", action="store_true",
                   help="xem adapter đọc ra gì, không giải mã và không ghi corpus")
    p.set_defaults(func=cmd_ingest)

    p = sub.add_parser("generate", parents=[common], help="sinh FAKE dataset bằng TTS/voice cloning")
    p.add_argument("--engines", nargs="*", help="vd: piper kokoro omnivoice")
    p.add_argument("--count", type=int, help="số mẫu mỗi engine")
    p.add_argument("--device", help="cpu | mps | cuda | auto")
    p.add_argument("--dry-run", action="store_true",
                   help="chỉ đếm còn thiếu bao nhiêu mẫu, không nạp model, không sinh")
    p.add_argument("--after-speaker", metavar="CMD",
                   help="lệnh chạy mỗi khi xong một speaker (mốc để đồng bộ corpus ra ngoài)")
    p.set_defaults(func=cmd_generate)

    p = sub.add_parser("augment", parents=[common], help="sinh thêm bản nhiễu/nén")
    p.add_argument("--copies", type=int, help="số bản augment cho mỗi utterance")
    p.add_argument("--splits", nargs="*", help="split được augment (mặc định: train)")
    p.set_defaults(func=cmd_augment)

    p = sub.add_parser("split", parents=[common], help="chia train/val/test speaker-disjoint")
    p.add_argument("--holdout", nargs="*", help="engine chỉ xuất hiện ở test")
    p.set_defaults(func=cmd_split)

    p = sub.add_parser("features", parents=[common], help="trích embedding bằng backbone")
    p.add_argument("--device", help="cpu | mps | cuda | auto")
    p.set_defaults(func=cmd_features)

    p = sub.add_parser("train", parents=[common], help="huấn luyện classifier")
    p.add_argument("--device")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("evaluate", parents=[common], help="đánh giá trên tập test")
    p.add_argument("--split", default="test")
    p.add_argument("--checkpoint")
    p.add_argument("--device")
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("detect", parents=[common], help="chấm điểm file audio")
    p.add_argument("files", nargs="+")
    p.add_argument("--checkpoint")
    p.add_argument("--threshold", type=float)
    p.add_argument("--device")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_detect)

    p = sub.add_parser("validate", parents=[common], help="kiểm tra corpus có đúng chuẩn không")
    p.add_argument("--fix", action="store_true", help="loại bỏ bản ghi hỏng khỏi manifest")
    p.add_argument("--recheck", action="store_true",
                   help="soi lại cả những bản ghi đã duyệt theo đúng chuẩn hiện tại")
    p.add_argument("--force", action="store_true",
                   help="cho phép --fix loại cả bản ghi từng đạt một chuẩn khác")
    p.add_argument("--prune-files", action="store_true",
                   help="xoá luôn file audio của bản ghi bị --fix loại")
    p.set_defaults(func=cmd_validate)

    p = sub.add_parser("migrate", parents=[common],
                       help="dọn corpus về đúng cấu trúc thư mục hiện hành")
    p.add_argument("--dry-run", action="store_true", help="chỉ đếm, không di chuyển file")
    p.set_defaults(func=cmd_migrate)

    p = sub.add_parser("progress", parents=[common],
                       help="đã sinh fake xong tới speaker nào (trạng thái nối tiếp)")
    p.add_argument("--out", help="ghi trạng thái ra file JSON")
    p.set_defaults(func=cmd_progress)

    p = sub.add_parser("pack", parents=[common], help="gói corpus thành 1 file zip (Kaggle/Colab)")
    p.add_argument("--out", help="đường dẫn zip đầu ra (mặc định: <thư mục làm việc>/corpus.zip)")
    p.add_argument("--compress", action="store_true", help="nén thật (chậm hơn, WAV vốn khó nén)")
    p.set_defaults(func=cmd_pack)

    p = sub.add_parser("unpack", parents=[common], help="bung zip corpus vào thư mục corpus")
    p.add_argument("archive", help="file zip do `pack` tạo ra")
    p.set_defaults(func=cmd_unpack)

    p = sub.add_parser("info", parents=[common], help="liệt kê thành phần khả dụng + trạng thái corpus")
    p.set_defaults(func=cmd_info)

    p = sub.add_parser("run", parents=[common], help="chạy nhiều stage liên tiếp")
    # Không dùng `choices` ở đây: argparse đối chiếu cả giá trị mặc định ([]) với
    # choices khi nargs="*", nên lệnh không truyền stage nào sẽ bị báo lỗi oan.
    p.add_argument("stages", nargs="*", metavar="STAGE", help=f"mặc định: {' '.join(STAGES)}")
    p.add_argument("--skip", nargs="*", choices=list(STAGES))
    p.add_argument("--path", help="thư mục dataset cho stage ingest")
    p.add_argument("--adapter", default="auto")
    p.add_argument("--name")
    p.add_argument("--hf")
    p.add_argument("--hf-split", default="train")
    p.add_argument("--hf-config", default=None)
    p.add_argument("--limit", type=int)
    p.add_argument("--per-speaker", type=int)
    p.add_argument("--engines", nargs="*")
    p.add_argument("--count", type=int)
    p.add_argument("--copies", type=int)
    p.add_argument("--splits", nargs="*")
    p.add_argument("--holdout", nargs="*")
    p.add_argument("--split", default="test")
    p.add_argument("--checkpoint")
    p.add_argument("--device")
    p.set_defaults(func=cmd_run)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(args.log_level)
    # `ingest <path>` và `ingest --path <path>` là một.
    if getattr(args, "path_opt", None):
        args.path = args.path_opt
    try:
        return args.func(args)
    except KeyboardInterrupt:
        log.warning("Đã dừng theo yêu cầu người dùng.")
        return 130
    except (FileNotFoundError, KeyError, ValueError, RuntimeError) as exc:
        log.error("%s: %s", type(exc).__name__, exc)
        if args.log_level == "DEBUG":
            raise
        return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
