# Copyright 2026 Rahul Reddy
# Licensed under the Apache License, Version 2.0
"""Structured logging configuration with Rich formatting support."""

import logging
from typing import Optional
from rich.logging import RichHandler


def setup_logging(level: str = "INFO", enable_rich: bool = True) -> logging.Logger:
    """Configure and return root logger for DevOps AI."""
    log_level = getattr(logging, level.upper(), logging.INFO)
    logger = logging.getLogger("devops_ai")
    logger.setLevel(log_level)

    # Avoid duplicate handlers on repeated setup calls
    if not logger.handlers:
        if enable_rich:
            handler = RichHandler(
                rich_tracebacks=True,
                show_time=True,
                show_path=False,
                tracebacks_show_locals=False,
            )
        else:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Get a child logger under the devops_ai namespace."""
    if name:
        return logging.getLogger(f"devops_ai.{name}")
    return logging.getLogger("devops_ai")
