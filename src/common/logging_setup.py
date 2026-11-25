"""Shared logging setup for consistent log formatting on host and peers."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path


def setup_logging(role: str, log_dir: str | Path = "logs", level: int = logging.INFO) -> Path:
    """Configure root logger to write to `logs/<role>_<timestamp>.log` and stderr."""
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_path / f"{role}_{timestamp}.log"

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(logging.WARNING)
    root.addHandler(stream_handler)

    return log_file
