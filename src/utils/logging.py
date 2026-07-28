"""Rich-backed logger, with a plain fallback if rich is unavailable."""
from __future__ import annotations

import logging

try:
    from rich.logging import RichHandler

    _HANDLER: logging.Handler = RichHandler(rich_tracebacks=True, show_path=False)
    _FMT = "%(message)s"
except Exception:  # pragma: no cover - rich should be installed
    _HANDLER = logging.StreamHandler()
    _FMT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"

_CONFIGURED = False


def get_logger(name: str = "ai-detector") -> logging.Logger:
    global _CONFIGURED
    if not _CONFIGURED:
        logging.basicConfig(level=logging.INFO, format=_FMT, handlers=[_HANDLER])
        _CONFIGURED = True
    return logging.getLogger(name)
