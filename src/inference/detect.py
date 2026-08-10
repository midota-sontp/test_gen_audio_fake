"""On-demand real/fake inference for a single arbitrary audio clip.

Mirrors the preprocess (VAD + length-normalize) -> extract (frozen WavLM) ->
classify (MLP) chain the pipeline uses for train/test, but for one in-memory
clip instead of a manifest. Used by the dashboard's "tự kiểm tra" upload tab.
"""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import torch

from ..models.mlp import MLPClassifier
from ..preprocessing.audio import _fix_length, _peak_normalize, _vad_concat
from ..utils.config import Config, resolve
from ..wavlm.cache import _resolve_checkpoint
from ..wavlm.extractor import WavLMExtractor


def load_extractor(cfg: Config, device: str = "cpu") -> WavLMExtractor:
    return WavLMExtractor(
        _resolve_checkpoint(cfg.paths.wavlm_checkpoint),
        device=device,
        output_layer=cfg.extract.get_path("output_layer"),
        pooling=cfg.extract.pooling,
    )


def load_classifier(cfg: Config, checkpoint: str | Path, device: str = "cpu") -> MLPClassifier:
    ckpt = torch.load(resolve(checkpoint), map_location=device)
    model = MLPClassifier(cfg.model.input_dim, cfg.model.hidden_dim, cfg.model.dropout).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def predict_bytes(cfg: Config, extractor: WavLMExtractor, model: MLPClassifier,
                   audio_bytes: bytes) -> dict:
    """audio_bytes: raw contents of an uploaded audio file (any librosa-readable format)."""
    import librosa

    pcfg = cfg.preprocess
    sr = int(pcfg.sample_rate)
    y, _ = librosa.load(io.BytesIO(audio_bytes), sr=sr, mono=True)
    y = y.astype(np.float32)
    raw_seconds = round(len(y) / sr, 2)

    if bool(pcfg.vad.enable):
        voiced = _vad_concat(y, int(pcfg.vad.top_db))
        if voiced.size > 0:
            y = voiced

    target_len = int(round(float(pcfg.target_seconds) * sr))
    y = _peak_normalize(_fix_length(y, target_len))

    with torch.no_grad():
        emb = extractor.extract(torch.from_numpy(y)).unsqueeze(0)
        prob_fake = torch.sigmoid(model(emb)).item()

    thr = float(cfg.evaluate.decision_threshold)
    return {
        "probability_fake": prob_fake,
        "prediction": "fake" if prob_fake >= thr else "real",
        "raw_seconds": raw_seconds,
    }
