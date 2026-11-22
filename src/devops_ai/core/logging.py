# Copyright 2026 Rahul Reddy
# Licensed under the Apache License, Version 2.0
"""Structured logging configuration with JSON and Rich formatting support."""

import datetime
import json
import logging
import re
from typing import Optional
from rich.logging import RichHandler


def _scrub_secrets(text: str) -> str:
    """Best-effort redaction of API keys, bearer tokens, passwords, and secrets."""
    if not isinstance(text, str):
        return text
    try:
        # Scrub Google API key patterns
        scrubbed = re.sub(r"AIza[0-9A-Za-z-_]{20,}", "[REDACTED_API_KEY]", text)
        # Scrub Bearer authorization tokens
        scrubbed = re.sub(
            r"(?i)(bearer\s+)[A-Za-z0-9\-._~+/]+=*",
            r"\g<1>[REDACTED_TOKEN]",
            scrubbed,
        )
        # Scrub password / token / secret / api_key assignment patterns
        scrubbed = re.sub(
            r"(?i)\b(password|passwd|secret|api[_-]?key|token)\b\s*([:=])\s*['\"]?([^\s'\",]+)['\"]?",
            r"\1\2[REDACTED_SECRET]",
            scrubbed,
        )
        return scrubbed
    except Exception:
        return text


class JSONLogFormatter(logging.Formatter):
    """Standardized RFC 8259 JSON log formatter with automated secret scrubbing."""

    def format(self, record: logging.LogRecord) -> str:
        msg = record.getMessage()
        sanitized_msg = _scrub_secrets(msg)

        dt = datetime.datetime.fromtimestamp(record.created, tz=datetime.timezone.utc)
        iso_time = dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"

        payload = {
            "timestamp": iso_time,
            "level": record.levelname,
            "logger": record.name,
            "message": sanitized_msg,
            "module": record.module,
            "line": record.lineno,
            "process_id": record.process,
            "thread_name": record.threadName,
        }

        if record.exc_info:
            payload["exception"] = _scrub_secrets(self.formatException(record.exc_info))

        if hasattr(record, "context") and isinstance(record.context, dict):
            payload["context"] = {
                k: _scrub_secrets(str(v)) if isinstance(v, str) else v
                for k, v in record.context.items()
            }

        return json.dumps(payload, ensure_ascii=False)


def setup_logging(
    level: str = "INFO",
    enable_rich: bool = True,
    log_format: str = "rich",
) -> logging.Logger:
    """Configure and return root logger for DevOps AI."""
    log_level = getattr(logging, level.upper(), logging.INFO)
    logger = logging.getLogger("devops_ai")
    logger.setLevel(log_level)

    # Avoid duplicate handlers on repeated setup calls
    if not logger.handlers:
        if log_format.lower() == "json":
            handler = logging.StreamHandler()
            handler.setFormatter(JSONLogFormatter())
        elif enable_rich:
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
