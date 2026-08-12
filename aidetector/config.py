"""Đọc/merge config YAML.

Một file config duy nhất điều khiển toàn pipeline. Truy cập lồng nhau bằng
dấu chấm: ``cfg["features.backbone.output_layer"]`` hoặc ``cfg.get("train.lr", 1e-3)``.
Có thể ghi đè từ CLI: ``--set train.lr=1e-4 --set generate.engines=[piper,kokoro_vi]``.
"""

from __future__ import annotations

import ast
import copy
from pathlib import Path
from typing import Any, Iterable

import yaml

DEFAULT_CONFIG_PATH = Path("configs/default.yaml")


class Config:
    """Wrapper mỏng quanh dict, hỗ trợ truy cập theo đường dẫn chấm."""

    def __init__(self, data: dict[str, Any], path: Path | None = None) -> None:
        self._data = data
        self.path = path

    # ---------------------------------------------------------------- factory
    @classmethod
    def load(cls, path: str | Path | None = None, overrides: Iterable[str] = ()) -> "Config":
        path = Path(path or DEFAULT_CONFIG_PATH)
        if not path.exists():
            raise FileNotFoundError(f"Không tìm thấy config: {path}")
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

        # Cho phép kế thừa: `extends: base.yaml` (đường dẫn tương đối với file hiện tại).
        base_ref = data.pop("extends", None)
        if base_ref:
            base = cls.load(path.parent / base_ref)
            data = _deep_merge(base.raw, data)

        cfg = cls(data, path)
        for item in overrides:
            cfg.apply_override(item)
        return cfg

    # ------------------------------------------------------------------ truy cập
    @property
    def raw(self) -> dict[str, Any]:
        return self._data

    def get(self, dotted: str, default: Any = None) -> Any:
        node: Any = self._data
        for key in dotted.split("."):
            if not isinstance(node, dict) or key not in node:
                return default
            node = node[key]
        return node

    def __getitem__(self, dotted: str) -> Any:
        sentinel = object()
        value = self.get(dotted, sentinel)
        if value is sentinel:
            raise KeyError(f"Thiếu khoá config: {dotted}")
        return value

    def section(self, dotted: str) -> dict[str, Any]:
        value = self.get(dotted, {})
        return value if isinstance(value, dict) else {}

    def set(self, dotted: str, value: Any) -> None:
        keys = dotted.split(".")
        node = self._data
        for key in keys[:-1]:
            node = node.setdefault(key, {})
            if not isinstance(node, dict):
                raise TypeError(f"Không thể ghi vào {dotted}: {key} không phải dict")
        node[keys[-1]] = value

    def apply_override(self, item: str) -> None:
        """`train.lr=1e-4` — giá trị được parse theo cú pháp Python/YAML."""
        if "=" not in item:
            raise ValueError(f"--set cần dạng key=value, nhận: {item!r}")
        key, _, raw = item.partition("=")
        self.set(key.strip(), _parse_scalar(raw.strip()))

    def copy(self) -> "Config":
        return Config(copy.deepcopy(self._data), self.path)

    def dump(self) -> str:
        return yaml.safe_dump(self._data, allow_unicode=True, sort_keys=False)

    def __repr__(self) -> str:  # pragma: no cover
        return f"Config(path={self.path}, keys={list(self._data)})"


def _parse_scalar(raw: str) -> Any:
    try:
        return ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        pass
    try:
        loaded = yaml.safe_load(raw)
    except yaml.YAMLError:
        return raw
    return raw if loaded is None and raw not in ("null", "~", "") else loaded


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out
