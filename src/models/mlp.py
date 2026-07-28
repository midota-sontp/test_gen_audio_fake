"""MLP classifier over pooled WavLM embeddings.

Outputs a raw logit (BCEWithLogitsLoss is used for numerical stability);
call `.probs()` / sigmoid at inference time.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class MLPClassifier(nn.Module):
    def __init__(self, input_dim: int = 768, hidden_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        # LayerNorm on the frozen embedding stabilizes its varying per-dim scale;
        # dropout both before and after the hidden layer curbs memorization.
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Dropout(dropout),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)  # [B] logits

    @staticmethod
    def probs(logits: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(logits)
