#!/usr/bin/env python3
"""Orchestrator — run stages 00→04 sequentially from a single config.

    python run_pipeline.py                        # full pipeline
    python run_pipeline.py --skip download        # reuse existing raw data
    python run_pipeline.py --only extract train   # run a subset of stages
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.utils.config import load_config          # noqa: E402
from src.utils.logging import get_logger          # noqa: E402
from src.utils.seed import set_seed               # noqa: E402

log = get_logger("pipeline")

STAGES = ["download", "preprocess", "extract", "train", "evaluate"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/mvp.yaml")
    ap.add_argument("--skip", nargs="*", default=[], choices=STAGES)
    ap.add_argument("--only", nargs="*", default=None, choices=STAGES)
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.seed)
    stages = args.only if args.only else [s for s in STAGES if s not in args.skip]

    for stage in stages:
        log.info("=" * 60)
        log.info("STAGE: %s", stage)
        log.info("=" * 60)
        if stage == "download":
            if str(cfg.dataset.get_path("source", "hf")).lower() == "local":
                from src.dataset.local_ingest import build_manifest
                build_manifest(cfg)
            else:
                from src.dataset.hf_download import download_and_split
                download_and_split(cfg)
        elif stage == "preprocess":
            from src.preprocessing.audio import preprocess_all
            preprocess_all(cfg)
        elif stage == "extract":
            from src.wavlm.cache import extract_all
            extract_all(cfg)
        elif stage == "train":
            from src.trainer.train import train
            train(cfg)
        elif stage == "evaluate":
            from src.evaluator.evaluate import evaluate
            evaluate(cfg)
    log.info("Pipeline finished: %s", stages)


if __name__ == "__main__":
    main()
