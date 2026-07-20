# Fake-speech dataset generator (VIVOS + Fish Speech S2).
#
# Based on Fish Audio's OFFICIAL CPU image, which already ships the `fish_speech`
# package + the correct CPU torch — so we avoid `pip install fish-speech` (which
# pins torch==2.8.0 CUDA and drags in the whole train/serve dep tree). We only add
# our light data-prep deps on top.
#
# GPU: swap the base for the CUDA image and set FISH_DEVICE=cuda:
#   FROM fishaudio/fish-speech:server-cuda-v2.0.0-beta
FROM fishaudio/fish-speech:server-cpu-v2.0.0-beta

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/app/.hfcache \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# light deps only (torch, fish_speech, librosa, soundfile, numpy already in base).
# These are unrelated to torch/transformers, so they won't disturb the base pins.
RUN pip install kagglehub pyyaml tqdm

COPY . /app

ENTRYPOINT ["python", "cli.py"]
CMD ["--config", "config.yaml"]
