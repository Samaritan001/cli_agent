"""Shared logging helpers for the cli_client package."""

from __future__ import annotations

import logging

LOG_FORMAT = "\033[32m%(levelname)s\033[0m:    %(message)s"


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def configure_logging(level: int = logging.INFO, fmt: str = LOG_FORMAT) -> None:
    """Configure root logging once (safe to call from entry points)."""
    root = logging.getLogger()
    if root.handlers:
        return
    logging.basicConfig(level=level, format=fmt)
