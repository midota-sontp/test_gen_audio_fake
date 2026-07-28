#!/usr/bin/env python3
"""Stage 04 — evaluate best checkpoint on the test set; write metrics + plots."""
import argparse

import _bootstrap  # noqa: F401
from src.evaluator.evaluate import evaluate
from src.utils.config import load_config
from src.utils.seed import set_seed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/mvp.yaml")
    ap.add_argument("--checkpoint", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg.seed)
    evaluate(cfg, checkpoint=args.checkpoint)


if __name__ == "__main__":
    main()
