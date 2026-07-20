#!/usr/bin/env python3
"""CLI entry point for the VIVOS + Fish Speech S2 fake-speech dataset generator.

    python cli.py --dataset vivos --output dataset --config config.yaml
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.generator import DatasetGenerator  # noqa: E402


def setup_logging(output_root: Path) -> None:
    log_dir = output_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    handlers = [logging.StreamHandler(), logging.FileHandler(log_dir / "generate.log")]
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        handlers=handlers,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate a VIVOS fake-speech dataset with Fish Speech S2.")
    ap.add_argument("--config", default="config.yaml", help="YAML config path")
    ap.add_argument("--dataset", help="override dataset_root (VIVOS folder with train/ test/)")
    ap.add_argument("--output", help="override output_root")
    ap.add_argument("--splits", nargs="*", help="override splits, e.g. --splits train")
    ap.add_argument("--overwrite", action="store_true", help="regenerate even if outputs exist")
    args = ap.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = Path(__file__).resolve().parent / cfg_path
    with open(cfg_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if args.dataset:
        config["dataset_root"] = args.dataset
    if args.output:
        config["output_root"] = args.output
    if args.splits:
        config["splits"] = args.splits
    if args.overwrite:
        config["overwrite"] = True

    output_root = Path(config["output_root"])
    setup_logging(output_root)
    log = logging.getLogger("cli")
    log.info("dataset_root=%s output_root=%s generator=%s splits=%s overwrite=%s",
             config["dataset_root"], config["output_root"],
             config.get("generator"), config.get("splits"), config.get("overwrite"))

    DatasetGenerator(config).run()


if __name__ == "__main__":
    main()
