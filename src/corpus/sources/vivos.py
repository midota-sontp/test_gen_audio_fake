"""VIVOS — lấy tarball chính chủ trên HuggingFace (AILAB-VNUHCM/vivos).

Cùng nội dung với bản Kaggle (kynthesis/vivos-vietnamese-speech-corpus-for-asr)
nhưng không cần Kaggle credential. Cấu trúc: vivos/{train,test}/waves/<SPK>/<SPK>_NNN.wav
-> speaker_id và recording_id lấy trực tiếp từ tên file.
"""
from __future__ import annotations

import gzip
import tarfile
from pathlib import Path
from typing import Iterator

from .base import RawItem, download, hf_url

REPO = "AILAB-VNUHCM/vivos"
TARBALL = "data/vivos.tar.gz"
PROMPTS = {"train": "data/prompts-train.txt.gz", "test": "data/prompts-test.txt.gz"}


class VivosSource:
    name = "vivos"
    label = 0
    generator = None

    def __init__(self, cache_dir: Path, hf_token: str | None = None):
        self.cache = Path(cache_dir) / self.name
        self.token = hf_token
        self.tar_path = self.cache / "vivos.tar.gz"
        self.prompts: dict[str, str] = {}

    def prepare(self) -> None:
        download(hf_url(REPO, TARBALL), self.tar_path, self.token, desc="vivos/tarball")
        for split, rel in PROMPTS.items():
            p = download(hf_url(REPO, rel), self.cache / Path(rel).name,
                         self.token, desc=f"vivos/{split}-prompts")
            with gzip.open(p, "rt", encoding="utf-8", errors="replace") as f:
                for line in f:
                    parts = line.strip().split(" ", 1)
                    if len(parts) == 2:
                        self.prompts[parts[0]] = parts[1]

    def units(self) -> list[str]:
        return ["vivos.tar.gz"]

    def expected_rows(self) -> int | None:
        return 12420      # 11660 train + 760 test

    def iter_unit(self, unit: str, start_row: int) -> Iterator[tuple[int, RawItem]]:
        if not self.prompts:
            self.prepare()
        row_idx = -1
        with tarfile.open(self.tar_path, mode="r|gz") as tf:
            for member in tf:
                if not member.isfile() or not member.name.endswith(".wav"):
                    continue
                row_idx += 1
                if row_idx < start_row:
                    continue            # vẫn phải giải nén tuần tự, nhưng không đọc nội dung
                parts = Path(member.name).parts
                split = parts[1] if len(parts) > 1 else "train"
                stem = Path(member.name).stem
                speaker = parts[3] if len(parts) > 3 else stem.split("_")[0]
                fh = tf.extractfile(member)
                if fh is None:
                    continue
                yield row_idx, RawItem(
                    item_id=f"vivos_{stem}",
                    source=self.name,
                    speaker_id=f"vivos_{speaker}",
                    speaker_known=True,
                    recording_id=f"vivos_{stem}",
                    original_split=split,
                    audio_bytes=fh.read(),
                    extra={"text": self.prompts.get(stem), "tar_member": member.name},
                )
