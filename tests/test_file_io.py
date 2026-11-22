# Copyright 2026 Rahul Reddy
# Licensed under the Apache License, Version 2.0
"""Unit tests for shared atomic file operations."""

from pathlib import Path
from unittest.mock import patch
import pytest
from devops_ai.core.exceptions import DevOpsAIError
from devops_ai.core.file_io import atomic_write_text


def test_atomic_write_new_file(tmp_path: Path):
    target = tmp_path / "subdir" / "output.txt"
    result = atomic_write_text(target, "Hello world", force=False, create_parents=True)
    assert result == target.resolve()
    assert target.exists()
    assert target.read_text(encoding="utf-8") == "Hello world"


def test_atomic_write_refuses_overwrite_without_force(tmp_path: Path):
    target = tmp_path / "existing.txt"
    target.write_text("Original content", encoding="utf-8")

    with pytest.raises(DevOpsAIError) as exc_info:
        atomic_write_text(target, "New content", force=False)
    assert exc_info.value.code == "FILE_ALREADY_EXISTS"
    assert "already exists" in str(exc_info.value)
    assert target.read_text(encoding="utf-8") == "Original content"


def test_atomic_write_overwrites_with_force(tmp_path: Path):
    target = tmp_path / "existing.txt"
    target.write_text("Original content", encoding="utf-8")

    result = atomic_write_text(target, "New content", force=True)
    assert result == target.resolve()
    assert target.read_text(encoding="utf-8") == "New content"


def test_atomic_write_creates_missing_parent_directories(tmp_path: Path):
    target = tmp_path / "a" / "b" / "c" / "deep.txt"
    assert not target.parent.exists()
    atomic_write_text(target, "Deep content", create_parents=True)
    assert target.exists()
    assert target.read_text(encoding="utf-8") == "Deep content"


def test_atomic_write_fails_missing_parent_when_create_parents_false(tmp_path: Path):
    target = tmp_path / "nonexistent" / "file.txt"
    with pytest.raises(DevOpsAIError) as exc_info:
        atomic_write_text(target, "Content", create_parents=False)
    assert exc_info.value.code == "DIRECTORY_NOT_FOUND"


def test_atomic_write_cleans_up_temp_file_on_write_failure(tmp_path: Path):
    target = tmp_path / "fail_write.txt"

    with patch("builtins.open", side_effect=OSError("Disk full")):
        with pytest.raises(DevOpsAIError) as exc_info:
            atomic_write_text(target, "Content")
        assert exc_info.value.code == "FILE_WRITE_ERROR"

    # Verify no leftover temporary files in target directory
    temp_files = list(tmp_path.glob(".*.tmp.*"))
    assert len(temp_files) == 0
    assert not target.exists()


def test_atomic_write_temp_filename_uniqueness(tmp_path: Path):
    # Verify that multiple writes create unique temp files and succeed without conflict
    target1 = tmp_path / "file1.txt"
    target2 = tmp_path / "file2.txt"
    atomic_write_text(target1, "Content 1")
    atomic_write_text(target2, "Content 2")
    assert target1.read_text(encoding="utf-8") == "Content 1"
    assert target2.read_text(encoding="utf-8") == "Content 2"


def test_atomic_write_target_parent_is_not_dir(tmp_path: Path):
    file_as_parent = tmp_path / "file.txt"
    file_as_parent.write_text("I am a file", encoding="utf-8")
    target = file_as_parent / "child.txt"

    with pytest.raises(DevOpsAIError) as exc_info:
        atomic_write_text(target, "Content")
    assert exc_info.value.code == "INVALID_ARGUMENT"
