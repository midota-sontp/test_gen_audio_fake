# CPU-only image (Docker on macOS cannot access Apple MPS/Metal).
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/workspace/data/.hfcache

WORKDIR /workspace

# System deps for librosa/soundfile audio decoding
RUN apt-get update && apt-get install -y --no-install-recommends \
        libsndfile1 ffmpeg git && \
    rm -rf /var/lib/apt/lists/*

# Install CPU-only torch/torchaudio, then the rest of the requirements
COPY requirements.txt .
RUN pip install --index-url https://download.pytorch.org/whl/cpu \
        torch==2.2.2 torchaudio==2.2.2 && \
    pip install -r requirements.txt

# Project code (data/checkpoints/reports come in via mounted volumes)
COPY src/ src/
COPY scripts/ scripts/
COPY configs/ configs/
COPY run_pipeline.py .

CMD ["python", "run_pipeline.py", "--config", "configs/mvp.yaml"]
