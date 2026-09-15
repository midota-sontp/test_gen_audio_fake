from __future__ import annotations

from pathlib import Path

from .common_voice import CommonVoiceSource
from .vietmed import VietMedSource
from .vivos import VivosSource

# Thêm nguồn FAKE (vd. vieneu_tts) vào đây khi có HF token đã accept terms.
REGISTRY = {
    "common_voice": CommonVoiceSource,
    "vietmed": VietMedSource,
    "vivos": VivosSource,
}
REAL_SOURCES = ["common_voice", "vietmed", "vivos"]


def build(name: str, cache_dir: Path, hf_token: str | None = None):
    if name not in REGISTRY:
        raise KeyError(f"Nguồn không tồn tại: {name}. Có: {', '.join(REGISTRY)}")
    return REGISTRY[name](cache_dir, hf_token)
