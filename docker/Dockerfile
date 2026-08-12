# CPU-only image cho toàn pipeline (chuẩn hoá dữ liệu → WavLM → classifier).
# Engine sinh fake cài thêm qua ARG WITH_GENERATORS=1.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/hf_cache \
    OMP_NUM_THREADS=4

# ffmpeg cần cho augment MP3/AAC và đọc mp3/m4a; libsndfile cho soundfile.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg libsndfile1 git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Torch CPU trước (bánh xe riêng, nhẹ hơn bản mặc định rất nhiều).
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
        torch torchaudio

COPY requirements.txt requirements-generate.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Engine TTS/cloning là tuỳ chọn: docker compose build --build-arg WITH_GENERATORS=1
ARG WITH_GENERATORS=0
RUN if [ "$WITH_GENERATORS" = "1" ]; then \
        pip install --no-cache-dir -r requirements-generate.txt; \
    fi

COPY aidetector ./aidetector
COPY configs ./configs

ENTRYPOINT ["python", "-m", "aidetector"]
CMD ["info"]
