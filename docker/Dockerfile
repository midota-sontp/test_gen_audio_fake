FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    CORPUS_OUT=/data/out \
    CORPUS_CACHE=/data/cache

RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg libsndfile1 ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ /app/src/
COPY scripts/ /app/scripts/

RUN mkdir -p /data/out /data/cache
VOLUME ["/data"]

ENTRYPOINT ["python", "/app/scripts/build_corpus.py"]
CMD ["--out", "/data/out", "--cache", "/data/cache"]
