# Copyright 2026 Rahul Reddy
# Licensed under the Apache License, Version 2.0
"""Unit tests for JSON structured logging and secret scrubbing."""

import json
import logging
from devops_ai.core.logging import JSONLogFormatter, _scrub_secrets


def test_scrub_secrets_masks_google_api_key():
    raw = "Failed connecting to https://generativelanguage.googleapis.com with key AIzaSyFakeKey123456789012345678901234"
    scrubbed = _scrub_secrets(raw)
    assert "AIzaSyFakeKey" not in scrubbed
    assert "[REDACTED_API_KEY]" in scrubbed


def test_scrub_secrets_masks_bearer_token():
    raw = "Authorization: Bearer secret-token-xyz-123456789"
    scrubbed = _scrub_secrets(raw)
    assert "secret-token-xyz" not in scrubbed
    assert "[REDACTED_TOKEN]" in scrubbed


def test_scrub_secrets_masks_password_assignments():
    raw_assign = "Connection failed with password = 'SuperSecretPassword!'"
    scrubbed = _scrub_secrets(raw_assign)
    assert "SuperSecretPassword" not in scrubbed
    assert "[REDACTED_SECRET]" in scrubbed


def test_json_formatter_structure():
    formatter = JSONLogFormatter()
    record = logging.LogRecord(
        name="devops_ai.test",
        level=logging.INFO,
        pathname="test_file.py",
        lineno=42,
        msg="Service initialized successfully",
        args=(),
        exc_info=None,
    )
    formatted = formatter.format(record)
    data = json.loads(formatted)

    assert data["level"] == "INFO"
    assert data["logger"] == "devops_ai.test"
    assert data["message"] == "Service initialized successfully"
    assert data["line"] == 42
    assert "timestamp" in data
    assert data["timestamp"].endswith("Z")
    assert "process_id" in data
    assert "thread_name" in data


def test_json_formatter_scrubs_secrets_in_message_and_context():
    formatter = JSONLogFormatter()
    record = logging.LogRecord(
        name="devops_ai.auth",
        level=logging.ERROR,
        pathname="auth.py",
        lineno=101,
        msg="Auth failed with token AIzaSyFakeKey123456789012345678901234",
        args=(),
        exc_info=None,
    )
    record.context = {
        "auth_header": "Bearer raw-token-secret-999",
        "user_id": 1234,
    }

    formatted = formatter.format(record)
    data = json.loads(formatted)

    assert "AIzaSyFakeKey" not in data["message"]
    assert "[REDACTED_API_KEY]" in data["message"]
    assert "raw-token-secret" not in data["context"]["auth_header"]
    assert "[REDACTED_TOKEN]" in data["context"]["auth_header"]
    assert data["context"]["user_id"] == 1234


def test_json_formatter_handles_exception():
    formatter = JSONLogFormatter()
    try:
        raise ValueError("Critical error with key AIzaSyFakeKey123456789012345678901234")
    except ValueError:
        import sys

        exc_info = sys.exc_info()

    record = logging.LogRecord(
        name="devops_ai.error",
        level=logging.CRITICAL,
        pathname="error.py",
        lineno=99,
        msg="Unhandled exception occurred",
        args=(),
        exc_info=exc_info,
    )
    formatted = formatter.format(record)
    data = json.loads(formatted)

    assert "exception" in data
    assert "AIzaSyFakeKey" not in data["exception"]
    assert "[REDACTED_API_KEY]" in data["exception"]
