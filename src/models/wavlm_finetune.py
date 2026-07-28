"""Partial fine-tuning of WavLM for spoof detection (CPU-friendly).

The layer-probe diagnostic showed layer ~6 carries the strongest spoof signal
and layers 7-12 hurt generalization. So we TRUNCATE WavLM at layer `n_keep`
(default 6) and only fine-tune the top `n_trainable` of those kept layers
(default 2 -> layers 5,6). Everything below is frozen and run under no_grad.

To keep this feasible on CPU we cache the *encoder input* (CNN features) once
via `encoder_input`; the relative-position bias in WavLM is produced by layer 0
(only layer 0 has `has_relative_attention_bias`), so the encoder input is the
deepest point we can safely cache — layers 0..n_keep-1 still run each step, but
that is far cheaper than the conv feature extractor + a full 12-layer stack, and
backprop only flows through the top `n_trainable` layers + the head.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


@torch.no_grad()
def encoder_input(wavlm, wav: torch.Tensor) -> torch.Tensor:
    """Raw 16 kHz waveform -> encoder-input features [T, C] (frozen, cacheable)."""
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)  # [1, samples]
    wav = wav.float()
    if getattr(wavlm.cfg, "normalize", False):
        wav = F.layer_norm(wav, wav.shape[1:])
    feats = wavlm.feature_extractor(wav)      # [B, C, T]
    feats = feats.transpose(1, 2)             # [B, T, C]
    feats = wavlm.layer_norm(feats)
    if wavlm.post_extract_proj is not None:
        feats = wavlm.post_extract_proj(feats)
    x = wavlm.dropout_input(feats)            # eval -> identity
    return x.squeeze(0)                       # [T, C]


class WavLMPartialFinetune(nn.Module):
    def __init__(self, wavlm, n_keep: int = 6, n_trainable: int = 2,
                 hidden_dim: int = 256, dropout: float = 0.3, pooling: str = "mean") -> None:
        super().__init__()
        self.encoder = wavlm.encoder
        self.n_keep = min(n_keep, len(self.encoder.layers))
        self.n_frozen = max(0, self.n_keep - n_trainable)
        self.pooling = pooling
        dim = self.encoder.embedding_dim * (2 if pooling in ("mean_std", "meanstd") else 1)
        self.head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Dropout(dropout),
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        # freeze the whole backbone, then unfreeze the top `n_trainable` kept layers
        for p in self.encoder.parameters():
            p.requires_grad_(False)
        for i in range(self.n_frozen, self.n_keep):
            for p in self.encoder.layers[i].parameters():
                p.requires_grad_(True)

    def train(self, mode: bool = True):
        """Keep the whole backbone in eval mode (no backbone dropout/noise); only
        the head follows train/eval. Trainable layers still receive gradients."""
        super().train(mode)
        self.encoder.eval()
        return self

    def backbone_parameters(self):
        for i in range(self.n_frozen, self.n_keep):
            yield from self.encoder.layers[i].parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T, C] cached encoder-input features -> [B] logits."""
        enc = self.encoder
        x_conv = enc.pos_conv(x.transpose(1, 2)).transpose(1, 2)
        x = x + x_conv
        if not enc.layer_norm_first:
            x = enc.layer_norm(x)
        x = F.dropout(x, p=enc.dropout, training=False)  # backbone dropout off
        x = x.transpose(0, 1)                            # [T, B, C]

        pos_bias = None
        with torch.no_grad():                            # frozen prefix: layer 0 makes pos_bias
            for i in range(self.n_frozen):
                x, _, pos_bias = enc.layers[i](
                    x, self_attn_padding_mask=None, need_weights=False, pos_bias=pos_bias)
            x = x.detach()
            if pos_bias is not None:
                pos_bias = pos_bias.detach()
        for i in range(self.n_frozen, self.n_keep):      # trainable top layers
            x, _, pos_bias = enc.layers[i](
                x, self_attn_padding_mask=None, need_weights=False, pos_bias=pos_bias)

        x = x.transpose(0, 1)                            # [B, T, C]
        if self.pooling in ("mean_std", "meanstd"):
            pooled = torch.cat([x.mean(dim=1), x.std(dim=1, unbiased=False)], dim=-1)
        else:
            pooled = x.mean(dim=1)
        return self.head(pooled).squeeze(-1)

    @staticmethod
    def probs(logits: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(logits)
