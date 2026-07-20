# Fake-speech dataset generator (VIVOS + Fish Speech S2).
#
# By default installs the CPU build of torch (works everywhere, incl. Docker on
# macOS which has NO GPU/MPS access — generation is then SLOW). For a Linux box
# with an NVIDIA GPU, build with:
#   docker build --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu121 -t vivos-fake .
FROM python:3.11-slim

ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/app/.hfcache \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# system libs: git (clone fish-speech), ffmpeg + libsndfile1 (audio I/O)
RUN apt-get update && apt-get install -y --no-install-recommends \
        git ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# torch first (pinned index), then our light deps
RUN pip install torch torchaudio --index-url ${TORCH_INDEX}
COPY requirements.txt .
RUN pip install -r requirements.txt

# fish-speech source (weights are NOT baked in — mount them at runtime)
RUN git clone --depth 1 https://github.com/fishaudio/fish-speech.git /app/third_party/fish-speech \
    && pip install -e /app/third_party/fish-speech

# project code
COPY . /app

ENTRYPOINT ["python", "cli.py"]
CMD ["--config", "config.yaml"]
