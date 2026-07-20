"""Step 1-3: scan the VIVOS corpus, parse prompts.txt, locate wavs, group by speaker.

VIVOS layout:
    <root>/<split>/prompts.txt          lines: "VIVOSSPK01_R001 <transcript>"
    <root>/<split>/waves/<spk>/<audio_id>.wav
    <root>/<split>/genders.txt          optional: "VIVOSSPK01 m"
"""
from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Utterance:
    audio_id: str          # VIVOSSPK01_R001
    speaker: str           # VIVOSSPK01
    text: str              # transcript
    wav_path: Path         # absolute path to the source wav
    split: str             # train / test


def _speaker_of(audio_id: str) -> str:
    return audio_id.split("_")[0]


def parse_split(dataset_root: str | Path, split: str) -> list[Utterance]:
    """Parse one VIVOS split into Utterance records (missing/empty rows skipped)."""
    root = Path(dataset_root)
    sdir = root / split
    prompts = sdir / "prompts.txt"
    if not prompts.exists():
        log.warning("No prompts.txt for split '%s' at %s — skipping", split, prompts)
        return []

    out: list[Utterance] = []
    missing = 0
    for line in prompts.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        audio_id, _, text = line.partition(" ")
        text = text.strip()
        speaker = _speaker_of(audio_id)
        wav = sdir / "waves" / speaker / f"{audio_id}.wav"
        if not text:
            continue
        if not wav.exists():
            missing += 1
            continue
        out.append(Utterance(audio_id, speaker, text, wav.resolve(), split))
    log.info("Split '%s': %d utterances (%d wavs missing)", split, len(out), missing)
    return out


def scan_dataset(dataset_root: str | Path, splits: list[str]) -> list[Utterance]:
    """Parse all requested splits."""
    records: list[Utterance] = []
    for split in splits:
        records.extend(parse_split(dataset_root, split))
    if not records:
        raise RuntimeError(
            f"No utterances found under {dataset_root} for splits {splits}. "
            f"Expected <root>/<split>/prompts.txt and <root>/<split>/waves/<spk>/*.wav"
        )
    return records


def group_by_speaker(records: list[Utterance]) -> dict[str, list[Utterance]]:
    """Step 3: group utterances by speaker id."""
    groups: dict[str, list[Utterance]] = defaultdict(list)
    for r in records:
        groups[r.speaker].append(r)
    return dict(groups)
