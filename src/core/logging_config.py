"""Logging configuration for Argus Surveillance.

Provides dual-destination logging:
- Console (stdout): Clean INFO-level messages.
- File (e.g. logs/topology_matrix.log): Rich DEBUG-level messages with thread names,
  timestamps, detailed request payloads, camera hardware events, and tracebacks.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Optional

DEFAULT_LOG_FILE = "logs/topology_matrix.log"
DEFAULT_CONSOLE_LEVEL = logging.INFO
DEFAULT_FILE_LEVEL = logging.DEBUG

_logging_initialized = False


def setup_logging(
    log_file: Optional[str] = DEFAULT_LOG_FILE,
    console_level: int = DEFAULT_CONSOLE_LEVEL,
    file_level: int = DEFAULT_FILE_LEVEL,
    max_bytes: int = 10 * 1024 * 1024,  # 10 MB
    backup_count: int = 5,
) -> logging.Logger:
    """Configures root and application loggers with console and file handlers."""
    global _logging_initialized

    root_logger = logging.getLogger()
    root_logger.setLevel(min(console_level, file_level))

    # Avoid duplicate handlers if setup_logging is called multiple times
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)

    # 1. Console Handler (Clean, terminal-friendly)
    console_formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # 2. File Handler (Detailed, debug-level with thread info)
    if log_file:
        log_path = Path(log_file)
        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            from logging.handlers import RotatingFileHandler

            file_handler = RotatingFileHandler(
                filename=str(log_path),
                maxBytes=max_bytes,
                backupCount=backup_count,
                encoding="utf-8",
            )
            file_formatter = logging.Formatter(
                fmt="%(asctime)s [%(levelname)s] [%(threadName)s] %(name)s: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            file_handler.setLevel(file_level)
            file_handler.setFormatter(file_formatter)
            root_logger.addHandler(file_handler)

            root_logger.info(
                f"[LOGGING] Detailed logging initialized -> file: '{log_path.resolve()}' "
                f"(FileLevel={logging.getLevelName(file_level)}, ConsoleLevel={logging.getLevelName(console_level)})"
            )
        except Exception as e:
            root_logger.warning(f"[LOGGING] Could not initialize file handler at '{log_file}': {e}")

    _logging_initialized = True
    return logging.getLogger("argus")
