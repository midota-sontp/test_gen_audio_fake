#!/usr/bin/env python3
"""Peek at a HF dataset: print feature schema, detected columns, and the
distribution of label/utt_type values over the first N streamed rows.

Useful to confirm label values before a real download run:
    HF_TOKEN=... python scripts/inspect_dataset.py --n 300
"""
import argparse
from collections import Counter

import _bootstrap  # noqa: F401
from src.dataset.hf_download import (
    _AUDIO_HINTS, _FILE_HINTS, _LABEL_HINTS, _SPEAKER_HINTS, _hf_token, _pick_column,
)
from src.utils.config import load_config


def main() -> None:
    from datasets import load_dataset

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/mvp.yaml")
    ap.add_argument("--n", type=int, default=200)
    args = ap.parse_args()
    cfg = load_config(args.config)

    ds = load_dataset(cfg.dataset.hf_id, split=cfg.dataset.source_split,
                      streaming=True, token=_hf_token())
    feats = ds.features or {}
    print("FEATURES:", {k: str(v) for k, v in feats.items()})
    print("audio  ->", _pick_column(feats, _AUDIO_HINTS, None))
    print("label  ->", _pick_column(feats, _LABEL_HINTS, None))
    print("speaker->", _pick_column(feats, _SPEAKER_HINTS, None))
    print("file   ->", _pick_column(feats, _FILE_HINTS, None))

    label_col = _pick_column(feats, _LABEL_HINTS, None)
    counters: dict[str, Counter] = {}
    for i, ex in enumerate(ds):
        if i >= args.n:
            break
        for col in feats:
            v = ex.get(col)
            if isinstance(v, (str, int, bool)):
                counters.setdefault(col, Counter())[str(v)] += 1
    for col, c in counters.items():
        print(f"\n{col} distinct (top10): {c.most_common(10)}")
    if label_col:
        print(f"\n>> Set dataset.fake/real_label_values to match `{label_col}` values above.")


if __name__ == "__main__":
    main()
