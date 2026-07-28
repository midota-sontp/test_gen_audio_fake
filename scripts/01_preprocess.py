#!/usr/bin/env python3
"""Stage 01 — preprocess raw clips (mono/16k/VAD/length) and write manifests."""
import argparse

import _bootstrap  # noqa: F401
from src.preprocessing.audio import preprocess_all
from src.utils.config import load_config
from src.utils.seed import set_seed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/mvp.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg.seed)
    preprocess_all(cfg)


if __name__ == "__main__":
    main()
