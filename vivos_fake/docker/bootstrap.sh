#!/usr/bin/env bash
# Container bootstrap: fetch S2 weights + VIVOS if missing, then generate.
# Everything runs inside Docker — no native setup needed. All steps are guarded
# by existence checks, so re-runs skip what's already there (resume-friendly).
set -euo pipefail

CKPT="${FISH_CKPT:-/app/third_party/fish-speech/checkpoints/s2-pro}"
DATASET="${VIVOS_DEST:-/data/vivos}"

if [ ! -f "$CKPT/codec.pth" ]; then
  echo ">> [1/3] Downloading Fish Speech S2-Pro weights -> $CKPT ..."
  python -c "from huggingface_hub import snapshot_download; snapshot_download('fishaudio/s2-pro', local_dir='$CKPT')"
else
  echo ">> [1/3] S2 weights present."
fi

if ! find "$DATASET" -name prompts.txt -print -quit 2>/dev/null | grep -q .; then
  echo ">> [2/3] VIVOS not found in $DATASET — downloading from Kaggle ..."
  python -m src.vivos_download "$DATASET"
else
  echo ">> [2/3] VIVOS present in $DATASET."
fi

echo ">> [3/3] Generating dataset ..."
exec python cli.py "$@"
