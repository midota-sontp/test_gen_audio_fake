#!/usr/bin/env python3
"""Stage 00 — download a balanced subset and split it speaker-disjoint."""
import argparse

import _bootstrap  # noqa: F401
from src.dataset.hf_download import download_and_split
from src.utils.config import load_config
from src.utils.seed import set_seed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/mvp.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg.seed)
    download_and_split(cfg)


if __name__ == "__main__":
    main()
