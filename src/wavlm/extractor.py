"""Frozen WavLM-Base feature extractor (HuggingFace transformers backend).

Loads `microsoft/wavlm-base` (or any local WavLMModel directory) and produces a
single pooled embedding per clip from a chosen transformer layer. The backbone is
frozen (eval + no grad).

`output_layer=n` selects `hidden_states[n]`, where index 0 is the feature-projection
output (encoder input) and 1..12 are the outputs after each transformer layer. So
`output_layer=6` reproduces the fairseq `extract_features(output_layer=6)` this
pipeline was tuned on (the mid-stack layer 6 separates spoof cues best; the last
layer 12 is content/phonetic and hurts).

The original vendored fairseq loader (WavLM.py/modules.py) is kept in the package for
reference but no longer imported — the HF weights are already cached locally, which
avoids re-downloading the ~360MB fairseq checkpoint.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch


class WavLMExtractor:
    def __init__(
        self,
        checkpoint_path: str | Path,
        device: torch.device | str = "cpu",
        output_layer: Optional[int] = None,
        pooling: str = "mean",
    ) -> None:
        from transformers import WavLMModel  # lazy: keeps import cost off other stages

        self.device = torch.device(device)
        self.output_layer = output_layer
        self.pooling = pooling

        model_id = str(checkpoint_path)
        # `microsoft/wavlm-base` (HF id) or a local WavLMModel dir; resolved from the
        # HF cache offline in Docker (TRANSFORMERS_OFFLINE=1).
        self.model = WavLMModel.from_pretrained(model_id)
        self.model.eval().to(self.device)
        for p in self.model.parameters():
            p.requires_grad_(False)

        self.embed_dim = int(self.model.config.hidden_size)

    @torch.no_grad()
    def extract(self, wav: torch.Tensor) -> torch.Tensor:
        """wav: 1-D float tensor at 16 kHz. Returns a 1-D pooled tensor (CPU).

        Dim is `embed_dim` for mean/max pooling, or `2*embed_dim` for mean_std.
        """
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)  # [1, T]
        wav = wav.to(self.device).float()

        out = self.model(wav, output_hidden_states=True)
        hs = out.hidden_states  # tuple length num_layers+1 (0 = encoder input)
        idx = self.output_layer if self.output_layer is not None else (len(hs) - 1)
        idx = max(0, min(int(idx), len(hs) - 1))
        feat = hs[idx]  # [1, T, D]

        if self.pooling == "max":
            pooled = feat.max(dim=1).values
        elif self.pooling in ("mean_std", "meanstd"):
            # concat time-mean and time-std -> [1, 2D]. Std captures temporal
            # dynamics (where TTS/VC artifacts live) that plain mean-pool discards.
            mean = feat.mean(dim=1)
            std = feat.std(dim=1, unbiased=False)
            pooled = torch.cat([mean, std], dim=-1)
        else:  # "mean"
            pooled = feat.mean(dim=1)
        return pooled.squeeze(0).detach().cpu()
