"""Đầu phân loại (classifier head) — cũng cắm rời như backbone.

Đầu vào là embedding đã pooling từ backbone [B, D], đầu ra là 1 logit
(>0 ⇒ FAKE). Thêm kiến trúc mới = thêm một lớp con `@register` ở đây rồi đổi
`model.head` trong config.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from .utils import get_logger

log = get_logger("aidetector.models")


class Head(nn.Module):
    """Lớp cơ sở cho mọi classifier head."""

    id: str = "base"
    description: str = ""

    def __init__(self, input_dim: int, **options) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.options = options

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [B, D] -> [B]
        raise NotImplementedError


_REGISTRY: dict[str, type[Head]] = {}


def register(cls: type[Head]) -> type[Head]:
    if cls.id in _REGISTRY:
        raise ValueError(f"Head trùng id: {cls.id}")
    _REGISTRY[cls.id] = cls
    return cls


def get_head(name: str) -> type[Head]:
    if name not in _REGISTRY:
        raise KeyError(f"Không có head {name!r}. Hiện có: {', '.join(sorted(_REGISTRY))}")
    return _REGISTRY[name]


def available_heads() -> dict[str, type[Head]]:
    return dict(_REGISTRY)


def build_head(config: dict, input_dim: int) -> Head:
    cfg = dict(config or {})
    name = cfg.pop("head", "mlp")
    head = get_head(name)(input_dim=input_dim, **cfg)
    n_params = sum(p.numel() for p in head.parameters() if p.requires_grad)
    log.info("Head %s · input_dim=%d · %s tham số", name, input_dim, f"{n_params:,}")
    return head


# --------------------------------------------------------------------------- heads
@register
class LinearHead(Head):
    id = "linear"
    description = "Hồi quy logistic — mốc so sánh cơ sở"

    def __init__(self, input_dim: int, **options) -> None:
        super().__init__(input_dim, **options)
        self.net = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


@register
class MLPHead(Head):
    id = "mlp"
    description = "MLP một lớp ẩn (mặc định)"

    def __init__(self, input_dim: int, hidden_dim: int = 256, dropout: float = 0.3, **options) -> None:
        super().__init__(input_dim, **options)
        # LayerNorm ngay đầu vào giúp hiệu chỉnh (calibration) tốt hơn khi
        # embedding từ các backbone khác nhau có thang đo rất khác nhau.
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


@register
class DeepMLPHead(Head):
    id = "deep_mlp"
    description = "MLP hai lớp ẩn — dùng khi dữ liệu đủ nhiều"

    def __init__(self, input_dim: int, hidden_dims: tuple[int, ...] = (512, 256),
                 dropout: float = 0.3, **options) -> None:
        super().__init__(input_dim, **options)
        layers: list[nn.Module] = [nn.LayerNorm(input_dim)]
        prev = input_dim
        for dim in hidden_dims:
            layers += [nn.Linear(prev, dim), nn.BatchNorm1d(dim), nn.GELU(), nn.Dropout(dropout)]
            prev = dim
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# ------------------------------------------------------------------ checkpoint
def save_checkpoint(path: str | Path, head: Head, meta: dict) -> Path:
    """Lưu head + toàn bộ thông tin cần để tái tạo pipeline lúc suy luận."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": head.state_dict(), "meta": meta}, path)
    return path


def load_checkpoint(path: str | Path, device: str = "cpu") -> tuple[Head, dict]:
    payload = torch.load(str(path), map_location=device, weights_only=False)
    meta = payload["meta"]
    head = build_head(dict(meta["model"]), int(meta["input_dim"]))
    head.load_state_dict(payload["state_dict"])
    head.eval().to(device)
    return head, meta
