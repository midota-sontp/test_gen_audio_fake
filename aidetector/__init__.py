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
CORPUS_SCHEMA_VERSION = 1
