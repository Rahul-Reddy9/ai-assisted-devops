# Copyright 2026 Rahul Reddy
# Licensed under the Apache License, Version 2.0
"""Structured log ingestion and parsing engine."""

from dataclasses import dataclass, field
import datetime
import json
from pathlib import Path
import re
from typing import Any, Dict, Iterable, List, Optional, Union
from devops_ai.core.exceptions import ObservabilityError
from devops_ai.core.logging import _scrub_secrets, get_logger

logger = get_logger("observability.parser")


@dataclass
class LogEntry:
    """Represents a parsed structured log entry."""

    timestamp: datetime.datetime
    level: str
    message: str
    raw: str
    context: Dict[str, Any] = field(default_factory=dict)

    @property
    def level_score(self) -> int:
        """Numeric severity score for anomaly and threshold analysis."""
        mapping = {
            "DEBUG": 0,
            "INFO": 1,
            "NOTICE": 1,
            "WARNING": 2,
            "WARN": 2,
            "ERROR": 3,
            "ERR": 3,
            "CRITICAL": 4,
            "FATAL": 4,
        }
        return mapping.get(self.level.upper(), 1)


class LogParser:
    """Parser for unstructured and structured log streams."""

    # Matches Day 6 format: 2024-02-09 10:00:00 INFO User logged in
    SYSLOG_PATTERN = re.compile(
        r"^(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}(?:\.\d+)?)\s+(\w+)\s+(.+)$"
    )

    # Matches ISO format: 2024-02-09T10:00:00Z [INFO] User logged in
    ISO_BRACKET_PATTERN = re.compile(r"^(\d{4}-\d{2}-\d{2}T[0-9:.]+Z?)\s+\[?(\w+)\]?\s*:?\s*(.+)$")

    def parse_line(self, line: str) -> Optional[LogEntry]:
        """Parse a single raw log line into a LogEntry with secret scrubbing."""
        stripped = line.strip()
        if not stripped:
            return None

        # 1. Attempt JSON parsing
        if stripped.startswith("{") and stripped.endswith("}"):
            try:
                data = json.loads(stripped)
                if isinstance(data, dict):
                    ts_raw = data.get("timestamp") or data.get("time") or data.get("datetime")
                    lvl_raw = data.get("level") or data.get("levelname") or "INFO"
                    msg_raw = data.get("message") or data.get("msg") or ""

                    parsed_ts = (
                        self._parse_timestamp(str(ts_raw))
                        if ts_raw
                        else datetime.datetime.now(datetime.timezone.utc)
                    )
                    sanitized_msg = _scrub_secrets(str(msg_raw))
                    return LogEntry(
                        timestamp=parsed_ts,
                        level=str(lvl_raw).upper(),
                        message=sanitized_msg,
                        raw=stripped,
                        context={
                            k: v
                            for k, v in data.items()
                            if k not in ("timestamp", "level", "message")
                        },
                    )
            except (json.JSONDecodeError, ValueError):
                pass

        # 2. Attempt Syslog / Day 6 regex parsing
        match = self.SYSLOG_PATTERN.match(stripped)
        if match:
            ts_str, lvl, msg = match.groups()
            parsed_ts = self._parse_timestamp(ts_str)
            if parsed_ts:
                return LogEntry(
                    timestamp=parsed_ts,
                    level=lvl.upper(),
                    message=_scrub_secrets(msg),
                    raw=stripped,
                )

        # 3. Attempt ISO Bracket regex parsing
        match_iso = self.ISO_BRACKET_PATTERN.match(stripped)
        if match_iso:
            ts_str, lvl, msg = match_iso.groups()
            parsed_ts = self._parse_timestamp(ts_str)
            if parsed_ts:
                return LogEntry(
                    timestamp=parsed_ts,
                    level=lvl.upper(),
                    message=_scrub_secrets(msg),
                    raw=stripped,
                )

        # 4. Fallback: unparseable line treated as raw INFO entry
        logger.debug(f"Unmatched log line format: {stripped[:60]}...")
        return None

    def parse_stream(self, stream: Iterable[str]) -> List[LogEntry]:
        """Parse an iterable of log lines, ignoring malformed lines."""
        entries: List[LogEntry] = []
        for line in stream:
            entry = self.parse_line(line)
            if entry is not None:
                entries.append(entry)
        return entries

    def parse_file(self, file_path: Union[str, Path], encoding: str = "utf-8") -> List[LogEntry]:
        """Read and parse all log lines from a file path."""
        path = Path(file_path).resolve()
        if not path.exists():
            raise ObservabilityError(f"Log file not found: '{path}'")
        if not path.is_file():
            raise ObservabilityError(f"Log path is not a file: '{path}'")

        try:
            with open(path, "r", encoding=encoding, errors="replace") as f:
                return self.parse_stream(f)
        except OSError as e:
            raise ObservabilityError(f"Failed to read log file '{path}': {e}") from None

    @staticmethod
    def _parse_timestamp(ts_str: str) -> Optional[datetime.datetime]:
        """Safely parse common timestamp formats."""
        ts_clean = ts_str.strip()
        # ISO formats
        for fmt in (
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d %H:%M:%S.%f",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%dT%H:%M:%S.%fZ",
        ):
            try:
                return datetime.datetime.strptime(ts_clean, fmt)
            except ValueError:
                continue

        # Try fromisoformat (Python 3.11+)
        try:
            return datetime.datetime.fromisoformat(ts_clean.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            return None
