"""Logging setup with a consistent format across collectors and pipelines."""

from __future__ import annotations

import logging
import sys
from typing import Optional

from src.config.settings import get_settings

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_initialized = False


def _initialize_root() -> None:
    global _initialized
    if _initialized:
        return
    settings = get_settings()
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT))
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(settings.log_level)
    _initialized = True


def get_logger(name: Optional[str] = None) -> logging.Logger:
    _initialize_root()
    return logging.getLogger(name if name else "kpx_forecast")
