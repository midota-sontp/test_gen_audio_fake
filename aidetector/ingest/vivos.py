"""VIVOS — bộ đọc tiếng Việt (~15 giờ, 46 speaker) của AILAB HCMUS.

Cấu trúc:
    vivos/
      train/{waves/VIVOSSPK01/VIVOSSPK01_R001.wav, prompts.txt, genders.txt}
      test/ {waves/VIVOSDEV01/...,               prompts.txt}
`prompts.txt`: mỗi dòng `VIVOSSPK01_R001 nội dung câu nói`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

from ..utils import get_logger
from .base import SourceAdapter, SourceItem, register

log = get_logger("aidetector.ingest.vivos")


@register
class VivosAdapter(SourceAdapter):
    name = "vivos"
    description = "VIVOS Vietnamese speech corpus (waves/ + prompts.txt)"

    @classmethod
    def probe(cls, root: Path) -> float:
        score = 0.0
        for split in ("train", "test"):
            if (root / split / "prompts.txt").exists():
                score += 0.4
            if (root / split / "waves").is_dir():
                score += 0.1
        # Tên speaker đặc trưng VIVOSSPK* / VIVOSDEV*
        if any(root.glob("*/waves/VIVOS*")):
            score += 0.2
        return min(score, 1.0)

    @staticmethod
    def _read_prompts(path: Path) -> dict[str, str]:
        prompts: dict[str, str] = {}
        if not path.exists():
            return prompts
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            utt, _, text = line.strip().partition(" ")
            if utt:
                prompts[utt] = text.strip()
        return prompts

    def count_hint(self, root: Path) -> int | None:
        return sum(1 for _ in root.glob("*/waves/*/*.wav")) or None

    def iter_items(self, root: Path) -> Iterator[SourceItem]:
        for split_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            waves = split_dir / "waves"
            if not waves.is_dir():
                continue
            prompts = self._read_prompts(split_dir / "prompts.txt")
            split_hint = "test" if split_dir.name.lower() in ("test", "dev") else "train"
            for wav in sorted(waves.rglob("*.wav")):
                utt = wav.stem
                yield SourceItem(
                    key=str(wav.relative_to(root)),
                    audio_path=wav,
                    speaker=wav.parent.name,
                    text=prompts.get(utt, ""),
                    split_hint=split_hint,
                )
