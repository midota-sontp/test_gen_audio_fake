from __future__ import annotations

from pathlib import Path

from .common_voice import CommonVoiceSource
from .vieneu import VieNeuSource
from .vietmed import VietMedSource
from .vivos import VivosSource

REGISTRY = {
    "common_voice": CommonVoiceSource,
    "vietmed": VietMedSource,
    "vivos": VivosSource,
    "vieneu_tts": VieNeuSource,
}
REAL_SOURCES = ["common_voice", "vietmed", "vivos"]
FAKE_SOURCES = ["vieneu_tts"]


def build(name: str, cache_dir: Path, hf_token: str | None = None, **kwargs):
    if name not in REGISTRY:
        raise KeyError(f"Nguồn không tồn tại: {name}. Có: {', '.join(REGISTRY)}")
    return REGISTRY[name](cache_dir, hf_token, **kwargs)
