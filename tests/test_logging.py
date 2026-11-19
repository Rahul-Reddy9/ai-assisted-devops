# Copyright 2026 Rahul Reddy
# Licensed under the Apache License, Version 2.0
"""Unit tests for structured logging."""

import logging
from devops_ai.core.logging import get_logger, setup_logging


def test_setup_logging_default():
    logger = setup_logging(level="INFO", enable_rich=False)
    assert logger.level == logging.INFO
    assert len(logger.handlers) >= 1


def test_setup_logging_idempotent():
    logger1 = setup_logging(level="DEBUG", enable_rich=False)
    initial_handlers = len(logger1.handlers)
    logger2 = setup_logging(level="DEBUG", enable_rich=False)
    assert len(logger2.handlers) == initial_handlers


def test_get_child_logger():
    child_logger = get_logger("test_module")
    assert child_logger.name == "devops_ai.test_module"
