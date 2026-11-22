# Copyright 2026 Rahul Reddy
# Licensed under the Apache License, Version 2.0
"""Unit tests for log parsing, temporal burst detection, and AI anomaly identification."""

import datetime
from pathlib import Path
from unittest.mock import patch
import pytest
from devops_ai.core.exceptions import ObservabilityError
from devops_ai.observability.log_analyzer import LogAnalyzer
from devops_ai.observability.log_parser import LogParser


@pytest.fixture
def sample_log_file(tmp_path: Path) -> Path:
    file = tmp_path / "system_logs.txt"
    content = """2024-02-09 10:00:00 INFO User logged in successfully
2024-02-09 10:00:05 WARNING Database response took longer than 200ms
2024-02-09 10:00:10 ERROR Database connection timeout occurred on port 5432
2024-02-09 10:00:15 ERROR Database retry failed after 3 attempts
2024-02-09 10:00:20 ERROR Pool exhausted: unable to allocate connection
2024-02-09 10:00:25 ERROR Circuit breaker tripped for database service
2024-02-09 10:01:00 INFO Service recovered and health checks passing
"""
    file.write_text(content, encoding="utf-8")
    return file


def test_log_parser_parses_syslog_format(sample_log_file: Path):
    parser = LogParser()
    logs = parser.parse_file(sample_log_file)
    assert len(logs) == 7
    assert logs[0].level == "INFO"
    assert logs[0].message == "User logged in successfully"
    assert logs[0].timestamp == datetime.datetime(2024, 2, 9, 10, 0, 0)
    assert logs[2].level == "ERROR"
    assert logs[2].level_score == 3


def test_log_parser_parses_json_format():
    parser = LogParser()
    json_lines = [
        '{"timestamp": "2024-02-09T10:00:00Z", "level": "INFO", "message": "Starting worker"}',
        '{"timestamp": "2024-02-09T10:00:05Z", "level": "ERROR", "message": "Worker crashed with AIzaSyFakeKey123456789012345678901234", "worker_id": 5}',
    ]
    logs = parser.parse_stream(json_lines)
    assert len(logs) == 2
    assert logs[0].level == "INFO"
    assert logs[1].level == "ERROR"
    assert "AIzaSyFakeKey" not in logs[1].message
    assert "[REDACTED_API_KEY]" in logs[1].message
    assert logs[1].context.get("worker_id") == 5


def test_log_parser_parses_iso_bracket_format():
    parser = LogParser()
    line = "2024-02-09T10:00:00Z [WARNING] Disk usage above 85%"
    entry = parser.parse_line(line)
    assert entry is not None
    assert entry.level == "WARNING"
    assert entry.message == "Disk usage above 85%"


def test_log_parser_handles_empty_and_invalid_lines():
    parser = LogParser()
    assert parser.parse_line("") is None
    assert parser.parse_line("   \n") is None
    assert parser.parse_line("Random non-log text without timestamp") is None


def test_log_parser_missing_file_raises_observability_error():
    parser = LogParser()
    with pytest.raises(ObservabilityError):
        parser.parse_file("nonexistent_log_file_12345.log")


def test_detect_error_bursts_detects_spike(sample_log_file: Path):
    parser = LogParser()
    logs = parser.parse_file(sample_log_file)
    analyzer = LogAnalyzer()

    # 4 errors occur within 15 seconds (10:00:10 to 10:00:25) -> exceeds threshold 3
    bursts = analyzer.detect_error_bursts(logs, window_seconds=30, threshold=3)
    assert len(bursts) == 1
    assert bursts[0].has_burst is True
    assert bursts[0].error_count == 4
    assert "Detected 4 error logs within 30s" in bursts[0].message


def test_detect_error_bursts_no_spike_when_below_threshold(sample_log_file: Path):
    parser = LogParser()
    logs = parser.parse_file(sample_log_file)
    analyzer = LogAnalyzer()

    # If threshold is 10, 4 errors does not constitute a burst
    bursts = analyzer.detect_error_bursts(logs, window_seconds=30, threshold=10)
    assert len(bursts) == 0


def test_detect_anomalies_deterministic(sample_log_file: Path):
    parser = LogParser()
    logs = parser.parse_file(sample_log_file)
    analyzer = LogAnalyzer()

    # Run anomaly detection with fixed random_state
    result = analyzer.detect_anomalies(logs, contamination=0.2, random_state=42)
    assert result.total_logs == 7
    assert result.anomaly_count >= 1
    assert len(result.anomalies) == result.anomaly_count
    assert result.method in ("isolation_forest", "statistical_percentile")


def test_detect_anomalies_statistical_fallback(sample_log_file: Path):
    parser = LogParser()
    logs = parser.parse_file(sample_log_file)
    analyzer = LogAnalyzer()

    # Simulate environment where scikit-learn is not available
    with patch.dict("sys.modules", {"sklearn": None, "sklearn.ensemble": None}):
        result = analyzer.detect_anomalies(logs, contamination=0.2)
        assert result.total_logs == 7
        assert result.anomaly_count >= 1
        assert result.method == "statistical_percentile"
