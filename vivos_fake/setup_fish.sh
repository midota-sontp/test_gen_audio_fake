#!/usr/bin/env bash
# One-time setup: install fish-speech (open-source S2) + download S2-Pro weights.
# Run natively (Apple Silicon MPS / CUDA) — not in a CPU-only container.
set -euo pipefail

REPO_DIR="${FISH_REPO_DIR:-third_party/fish-speech}"
CKPT_DIR="$REPO_DIR/checkpoints/s2-pro"

echo ">> Installing generation deps ..."
pip install -r requirements.txt

if [ ! -d "$REPO_DIR/.git" ]; then
  echo ">> Cloning fishaudio/fish-speech -> $REPO_DIR ..."
  git clone https://github.com/fishaudio/fish-speech.git "$REPO_DIR"
fi

echo ">> Installing fish-speech (editable) ..."
pip install -e "$REPO_DIR"

echo ">> Downloading S2-Pro weights (public) -> $CKPT_DIR ..."
huggingface-cli download fishaudio/s2-pro --local-dir "$CKPT_DIR"

echo ">> Done: codec=$CKPT_DIR/codec.pth model=$CKPT_DIR/*.safetensors"
