"""Backbone trích đặc trưng — cắm rời để sau này đổi model khác.

Mặc định là **WavLM** (frozen). Muốn thử XLS-R, HuBERT, Whisper-encoder hay bất kỳ
model SSL nào trên HuggingFace thì chỉ cần đổi `features.backbone.name` /
`checkpoint` trong config; cache đặc trưng tách riêng theo từng backbone nên các
thí nghiệm không đè lên nhau.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from ..utils import get_logger, slugify

log = get_logger("aidetector.features.backbone")

POOLINGS = ("mean", "mean_std", "max", "first")


class Backbone:
    """Lớp cơ sở: nhận danh sách sóng âm 16 kHz → ma trận embedding [B, D]."""

    id: str = "base"
    description: str = ""
    default_checkpoint: str = ""

    def __init__(
        self,
        checkpoint: str | None = None,
        device: str = "cpu",
        output_layer: int = 6,
        pooling: str = "mean",
        sample_rate: int = 16_000,
        **options,
    ) -> None:
        if pooling not in POOLINGS:
            raise ValueError(f"pooling phải thuộc {POOLINGS}, nhận {pooling!r}")
        self.checkpoint = checkpoint or self.default_checkpoint
        self.device = device
        self.output_layer = output_layer
        self.pooling = pooling
        self.sample_rate = sample_rate
        self.options = options
        self._model = None

    # ------------------------------------------------------------------ khoá cache
    @property
    def cache_key(self) -> str:
        """Khoá thư mục cache — đổi bất kỳ tham số nào ⇒ cache mới, không lẫn lộn."""
        return slugify(f"{self.id}-{self.checkpoint}-L{self.output_layer}-{self.pooling}", 96)

    @property
    def output_dim(self) -> int:
        base = self.hidden_size
        return base * 2 if self.pooling == "mean_std" else base

    @property
    def hidden_size(self) -> int:
        raise NotImplementedError

    # ------------------------------------------------------------------ vòng đời
    def load(self) -> None:
        raise NotImplementedError

    def ensure_loaded(self) -> None:
        if self._model is None:
            self.load()

    def embed(self, waveforms: Sequence[np.ndarray]) -> np.ndarray:
        raise NotImplementedError

    # ------------------------------------------------------------------ pooling
    def _pool(self, frames, mask=None):
        """frames: [B, T, D] tensor torch → [B, D'] theo kiểu pooling đã chọn."""
        import torch

        if mask is not None:
            mask = mask.unsqueeze(-1).to(frames.dtype)
            frames = frames * mask
            denom = mask.sum(dim=1).clamp(min=1.0)
        else:
            denom = torch.full((frames.size(0), 1), float(frames.size(1)), device=frames.device)

        if self.pooling == "first":
            return frames[:, 0]
        if self.pooling == "max":
            return frames.max(dim=1).values
        mean = frames.sum(dim=1) / denom
        if self.pooling == "mean":
            return mean
        # mean_std: ghép thêm độ lệch chuẩn theo thời gian — bắt được dao động
        # prosody mà TTS thường làm phẳng.
        var = ((frames - mean.unsqueeze(1)) ** 2).sum(dim=1) / denom
        return torch.cat([mean, var.clamp(min=1e-8).sqrt()], dim=-1)


_REGISTRY: dict[str, type[Backbone]] = {}


def register(cls: type[Backbone]) -> type[Backbone]:
    if cls.id in _REGISTRY:
        raise ValueError(f"Backbone trùng id: {cls.id}")
    _REGISTRY[cls.id] = cls
    return cls


def get_backbone(name: str) -> type[Backbone]:
    if name not in _REGISTRY:
        raise KeyError(f"Không có backbone {name!r}. Hiện có: {', '.join(sorted(_REGISTRY))}")
    return _REGISTRY[name]


def available_backbones() -> dict[str, type[Backbone]]:
    return dict(_REGISTRY)


def build_backbone(config: dict, device: str) -> Backbone:
    cfg = dict(config or {})
    name = cfg.pop("name", "wavlm")
    return get_backbone(name)(device=device, **cfg)


# --------------------------------------------------------------------------- HF SSL
class HFSSLBackbone(Backbone):
    """Bọc chung cho các model SSL dạng transformer trên HuggingFace.

    Dùng được cho WavLM / wav2vec2 / XLS-R / HuBERT / data2vec — chúng chia sẻ cùng
    giao diện `AutoModel(output_hidden_states=True)`.
    """

    def load(self) -> None:
        from transformers import AutoConfig, AutoModel

        log.info("Nạp backbone %s (%s) trên %s", self.id, self.checkpoint, self.device)
        self._config = AutoConfig.from_pretrained(self.checkpoint)
        model = AutoModel.from_pretrained(self.checkpoint)
        model.eval().to(self.device)
        for param in model.parameters():          # backbone đóng băng hoàn toàn
            param.requires_grad_(False)
        self._model = model

        n_layers = getattr(self._config, "num_hidden_layers", 12)
        if not 0 <= self.output_layer <= n_layers:
            raise ValueError(
                f"output_layer={self.output_layer} ngoài khoảng [0, {n_layers}] của {self.checkpoint}"
            )

    @property
    def hidden_size(self) -> int:
        if self._model is None:
            from transformers import AutoConfig

            return int(AutoConfig.from_pretrained(self.checkpoint).hidden_size)
        return int(self._config.hidden_size)

    def embed(self, waveforms: Sequence[np.ndarray]) -> np.ndarray:
        import torch

        self.ensure_loaded()
        assert self._model is not None

        lengths = [len(w) for w in waveforms]
        max_len = max(lengths)
        batch = np.zeros((len(waveforms), max_len), dtype=np.float32)
        for i, wav in enumerate(waveforms):
            batch[i, : len(wav)] = wav

        inputs = torch.from_numpy(batch).to(self.device)
        attention = torch.zeros((len(waveforms), max_len), dtype=torch.long, device=self.device)
        for i, n in enumerate(lengths):
            attention[i, :n] = 1

        with torch.no_grad():
            out = self._model(inputs, attention_mask=attention, output_hidden_states=True)
        # hidden_states[0] là đầu ra CNN feature-extractor, [k] là sau block thứ k.
        frames = out.hidden_states[self.output_layer]

        frame_mask = None
        get_lens = getattr(self._model, "_get_feat_extract_output_lengths", None)
        if get_lens is not None:
            feat_lens = get_lens(torch.tensor(lengths, device=self.device)).to(torch.long)
            frame_mask = (
                torch.arange(frames.size(1), device=self.device)[None, :] < feat_lens[:, None]
            )
        pooled = self._pool(frames, frame_mask)
        return pooled.float().cpu().numpy()


@register
class WavLMBackbone(HFSSLBackbone):
    id = "wavlm"
    description = "WavLM (Microsoft) — mặc định của dự án"
    default_checkpoint = "microsoft/wavlm-base-plus"


@register
class Wav2Vec2Backbone(HFSSLBackbone):
    id = "wav2vec2"
    description = "wav2vec 2.0 / XLS-R (đa ngữ)"
    default_checkpoint = "facebook/wav2vec2-xls-r-300m"


@register
class HubertBackbone(HFSSLBackbone):
    id = "hubert"
    description = "HuBERT"
    default_checkpoint = "facebook/hubert-base-ls960"


# --------------------------------------------------------------------- Whisper
@register
class WhisperEncoderBackbone(Backbone):
    id = "whisper"
    description = "Encoder của Whisper (đặc trưng thiên về nội dung/kênh truyền)"
    default_checkpoint = "openai/whisper-small"

    def load(self) -> None:
        from transformers import AutoFeatureExtractor, WhisperModel

        log.info("Nạp Whisper encoder %s trên %s", self.checkpoint, self.device)
        self._extractor = AutoFeatureExtractor.from_pretrained(self.checkpoint)
        model = WhisperModel.from_pretrained(self.checkpoint).encoder
        model.eval().to(self.device)
        for param in model.parameters():
            param.requires_grad_(False)
        self._model = model
        self._config = model.config

    @property
    def hidden_size(self) -> int:
        if self._model is None:
            from transformers import AutoConfig

            return int(AutoConfig.from_pretrained(self.checkpoint).d_model)
        return int(self._config.d_model)

    def embed(self, waveforms: Sequence[np.ndarray]) -> np.ndarray:
        import torch

        self.ensure_loaded()
        assert self._model is not None
        inputs = self._extractor(
            [np.asarray(w, dtype=np.float32) for w in waveforms],
            sampling_rate=self.sample_rate,
            return_tensors="pt",
        )
        features = inputs.input_features.to(self.device)
        with torch.no_grad():
            out = self._model(features, output_hidden_states=True)
        idx = min(self.output_layer, len(out.hidden_states) - 1)
        return self._pool(out.hidden_states[idx]).float().cpu().numpy()
