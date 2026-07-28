#!/usr/bin/env python3
"""Stage 03 — train the MLP classifier over cached embeddings."""
import argparse

import _bootstrap  # noqa: F401
from src.trainer.train import train
from src.utils.config import load_config
from src.utils.seed import set_seed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/mvp.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg.seed)
    train(cfg)


if __name__ == "__main__":
    main()
