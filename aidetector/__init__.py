"""ai-detector — phát hiện giọng nói tổng hợp/nhân bản tiếng Việt.

Luồng:
    Vietnamese real speech
        ├── REAL dataset
        └── Voice cloning / TTS (OmniVoice · Kokoro · Piper · ...) ── FAKE dataset
                                        │
                                Audio augmentation
                                        │
                                      WavLM
                                        │
                                   Classifier ── REAL / FAKE
"""

__version__ = "2.0.0"

# Phiên bản của CHUẨN corpus. Tăng lên khi schema manifest đổi.
#   1 — một `metadata.csv` gộp ở gốc, cây `<label>/<nguồn|engine>/<speaker>/`
#   2 — mỗi bộ dữ liệu một thư mục tự chứa: `<bộ>/metadata.csv` + `<bộ>/real|fake/…`
CORPUS_SCHEMA_VERSION = 2
