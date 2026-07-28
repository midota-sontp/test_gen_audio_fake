"""Dataset over the cached WavLM embeddings produced by stage 02."""
from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset


class EmbeddingDataset(Dataset):
    def __init__(self, split_dir: str | Path) -> None:
        self.files = sorted(Path(split_dir).glob("sample_*.pt"))
        if not self.files:
            raise FileNotFoundError(f"No embeddings found in {split_dir}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, i: int):
        rec = torch.load(self.files[i], map_location="cpu")
        emb = rec["emb"].float()
        label = torch.tensor(float(rec["label"]), dtype=torch.float32)
        return emb, label
