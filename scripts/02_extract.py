#!/usr/bin/env python3
"""Stage 02 — extract & cache one frozen-WavLM embedding per processed clip."""
import argparse

import _bootstrap  # noqa: F401
from src.utils.config import load_config
from src.utils.seed import set_seed
from src.wavlm.cache import extract_all


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/mvp.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg.seed)
    extract_all(cfg)


if __name__ == "__main__":
    main()
