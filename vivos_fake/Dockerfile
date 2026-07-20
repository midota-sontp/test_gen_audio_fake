# Fake-speech dataset generator (VIVOS + Fish Speech S2).
#
# Based on Fish Audio's OFFICIAL CPU image, which ships a uv-managed venv at
# /app/.venv (python 3.12) with torch-cpu + the fish_speech package + all its
# runtime deps already installed. We put that venv on PATH, add our 3 light deps
# into it via uv, and run from a separate /work dir (so we don't clobber the base
# image's own /app files).
#
# GPU: swap the base for the CUDA image and set FISH_DEVICE=cuda:
#   FROM fishaudio/fish-speech:server-cuda-v2.0.0-beta
FROM fishaudio/fish-speech:server-cpu-v2.0.0-beta

# put the image's venv first on PATH -> `python` / `sys.executable` = the env that
# actually has torch + fish_speech + huggingface_hub.
ENV VIRTUAL_ENV=/app/.venv \
    PATH=/app/.venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# our data-prep deps, installed INTO the venv (uv is the image's package manager;
# the venv has no pip). torch/transformers/etc. are untouched.
RUN uv pip install --python /app/.venv/bin/python kagglehub pyyaml tqdm

WORKDIR /work
COPY . /work

ENTRYPOINT ["python", "cli.py"]
CMD ["--config", "config.yaml"]
