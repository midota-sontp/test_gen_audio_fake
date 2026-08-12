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
from .corpus.manifest import Manifest
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
        cls = get_adapter(args.adapter) if args.adapter != "auto" else detect_adapter(root)[0]
        adapter = cls()
        name = args.name or root.name

    with timed(f"ingest {name}", log):
        result = ingest_source(
            manifest, adapter, root, name, spec,
            limit=args.limit, per_speaker=args.per_speaker,
            overwrite=args.overwrite,
            language=cfg.get("language", "vi"),
        )
    manifest.save()
    print(manifest.summary())
    return 0 if result.get("kept") else 1


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
    results = []
    for engine_id in engines:
        if engine_id not in known:
            log.error("Bỏ qua engine lạ %r (hiện có: %s)", engine_id, ", ".join(sorted(known)))
            continue
        status = known[engine_id].availability()
        if not status:
            log.warning("Bỏ qua %s — %s%s", engine_id, status.reason,
                        f". Cài: {status.hint}" if status.hint else "")
            continue
        with timed(f"generate {engine_id} ({per_engine} mẫu)", log):
            results.append(generate_fakes(
                manifest, engine_id, spec, per_engine, device=device,
                options=cfg.section(f"generate.options.{engine_id}"),
                voices=cfg.get(f"generate.voices.{engine_id}"),
                extra_texts=cfg.get("generate.texts_file"),
                min_words=int(cfg.get("generate.min_words", 6)),
                max_words=int(cfg.get("generate.max_words", 40)),
                overwrite=args.overwrite,
            ))
        manifest.save()          # lưu sau mỗi engine để không mất công nếu ngắt giữa chừng

    print(manifest.summary())
    return 0 if any(r.get("kept") for r in results) else 1


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

    log.info("Kiểm tra %d bản ghi theo chuẩn: %s", len(manifest), spec.describe())
    issues: Counter[str] = Counter()
    broken: list[str] = []
    from .utils import progress

    for rec in progress(list(manifest), total=len(manifest), label="validate"):
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
        for issue in check_quality(audio, spec, rec.utt_id):
            issues[issue.code] += 1
            broken.append(rec.utt_id)
        for err in rec.validate():
            issues[f"schema:{err.split(':')[0]}"] += 1

    print(manifest.summary())
    if issues:
        print("\nVấn đề phát hiện được:")
        for code, count in issues.most_common():
            print(f"  {code:<20} {count}")
        print(f"\n{len(set(broken))} bản ghi không đạt chuẩn.")
        if args.fix:
            for utt_id in set(broken):
                manifest.remove(utt_id)
            manifest.save()
            print(f"Đã loại {len(set(broken))} bản ghi khỏi manifest (--fix).")
        return 1
    print("\n✔ Toàn bộ corpus đạt chuẩn.")
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
    if (Path(root) / "manifest.csv").exists():
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
    p.set_defaults(func=cmd_ingest)

    p = sub.add_parser("generate", parents=[common], help="sinh FAKE dataset bằng TTS/voice cloning")
    p.add_argument("--engines", nargs="*", help="vd: piper kokoro omnivoice")
    p.add_argument("--count", type=int, help="số mẫu mỗi engine")
    p.add_argument("--device", help="cpu | mps | cuda | auto")
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
    p.set_defaults(func=cmd_validate)

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
