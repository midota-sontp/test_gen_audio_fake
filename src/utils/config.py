"""Config loading + repo-root resolution shared by all stages."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

# repo root = two levels up from this file (src/utils/config.py -> repo/)
REPO_ROOT = Path(__file__).resolve().parents[2]


class Config(dict):
    """dict with attribute access and dotted-path lookup."""

    def __getattr__(self, name: str) -> Any:
        try:
            val = self[name]
        except KeyError as e:
            raise AttributeError(name) from e
        return Config(val) if isinstance(val, dict) else val

    def get_path(self, dotted: str, default: Any = None) -> Any:
        cur: Any = self
        for part in dotted.split("."):
            if not isinstance(cur, dict) or part not in cur:
                return default
            cur = cur[part]
        return cur


def load_config(path: str | os.PathLike = "configs/mvp.yaml") -> Config:
    cfg_path = Path(path)
    if not cfg_path.is_absolute():
        cfg_path = REPO_ROOT / cfg_path
    with open(cfg_path, "r") as f:
        data = yaml.safe_load(f)
    return Config(data)


def resolve(rel: str | os.PathLike) -> Path:
    """Resolve a possibly-relative path against the repo root."""
    p = Path(rel)
    return p if p.is_absolute() else REPO_ROOT / p
