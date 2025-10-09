"""Logging utilities for experiment tracking."""
from __future__ import annotations

import logging
import sys
from pathlib import Path


def setup_logging(log_file: Path, level: str = "INFO") -> logging.Logger:
    """Configure logging to stream to stdout and file."""
    log_level = getattr(logging, level.upper(), logging.INFO)
    logger = logging.getLogger("federated")
    logger.setLevel(log_level)
    logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def get_logger() -> logging.Logger:
    """Return the experiment logger, creating a default if necessary."""
    logger = logging.getLogger("federated")
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


__all__ = ["setup_logging", "get_logger"]
