"""metadata.csv writer: incremental + resumable + thread-safe.

Columns: audio_path,label,speaker,text,generator,split
  audio_path  posix path relative to output_root (e.g. real/VIVOSSPK01/..._R001.wav)
  label       0 = real, 1 = fake
  generator   "real" for real clips, else the backend name (e.g. FishSpeechS2)

Rows are appended as soon as an audio file is confirmed on disk, so an interrupted
run leaves a valid partial metadata.csv. On resume, previously-written audio_paths
are loaded and de-duplicated, so re-running never doubles rows.
"""
from __future__ import annotations

import csv
import threading
from pathlib import Path

FIELDS = ["audio_path", "label", "speaker", "text", "generator", "split"]


class MetadataWriter:
    def __init__(self, csv_path: str | Path):
        self.path = Path(csv_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._seen: set[str] = set()
        self._load_existing()

    def _load_existing(self) -> None:
        if not self.path.exists():
            with open(self.path, "w", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=FIELDS).writeheader()
            return
        with open(self.path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("audio_path"):
                    self._seen.add(row["audio_path"])

    def has(self, audio_path: str) -> bool:
        return audio_path in self._seen

    def add(self, audio_path: str, label: int, speaker: str, text: str,
            generator: str, split: str) -> bool:
        """Append one row unless already present. Returns True if written."""
        with self._lock:
            if audio_path in self._seen:
                return False
            with open(self.path, "a", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=FIELDS).writerow({
                    "audio_path": audio_path, "label": label, "speaker": speaker,
                    "text": text, "generator": generator, "split": split,
                })
            self._seen.add(audio_path)
            return True

    def __len__(self) -> int:
        return len(self._seen)
