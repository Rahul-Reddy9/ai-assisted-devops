# Copyright 2026 Rahul Reddy
# Licensed under the Apache License, Version 2.0
"""Shared atomic file operations and safe I/O utilities."""

import os
from pathlib import Path
import shutil
import tempfile
from typing import Union
from devops_ai.core.exceptions import DevOpsAIError
from devops_ai.core.logging import get_logger

logger = get_logger("core.file_io")


def atomic_write_text(
    target_path: Union[str, Path],
    content: str,
    encoding: str = "utf-8",
    force: bool = False,
    create_parents: bool = True,
) -> Path:
    """Safely and atomically write text content to a destination file.

    Guarantees:
    - Writes to an exclusively created, uniquely named temporary file located
      in the destination directory (same filesystem).
    - Under force=False: Strictly refuses to overwrite existing files, raising
      DevOpsAIError(code='FILE_ALREADY_EXISTS').
    - Under force=True: Replaces the destination atomically via os.replace,
      preserving existing permissions where supported.
    - Temporary files are guaranteed to be cleaned up on all failure paths.
    - Flushes and syncs file descriptors (os.fsync) before replacement;
      synchronizes the parent directory descriptor on POSIX systems where supported.
      (On Windows, directory fsync is not supported by the OS and is gracefully omitted).

    Args:
        target_path: Destination file path.
        content: Text content to write.
        encoding: File character encoding (default: utf-8).
        force: If True, overwrite destination if it exists. If False, fail if exists.
        create_parents: If True, automatically create missing parent directories.

    Returns:
        Resolved destination Path object.

    Raises:
        DevOpsAIError: If destination exists and force=False, if parent directory
                       is missing and create_parents=False, or on I/O failure.
    """
    target = Path(target_path).resolve()

    if target.parent.exists():
        if not target.parent.is_dir():
            raise DevOpsAIError(
                f"Destination parent '{target.parent}' is not a directory.",
                code="INVALID_ARGUMENT",
            )
    else:
        if create_parents:
            try:
                target.parent.mkdir(parents=True, exist_ok=True)
            except OSError as e:
                raise DevOpsAIError(
                    f"Failed to create parent directory '{target.parent}': {e}",
                    code="FILE_WRITE_ERROR",
                ) from None
        else:
            raise DevOpsAIError(
                f"Destination parent directory '{target.parent}' does not exist.",
                code="DIRECTORY_NOT_FOUND",
            )

    # Pre-check existence for friendly error reporting
    if target.exists() and not force:
        raise DevOpsAIError(
            f"Destination file '{target}' already exists. Use --force to overwrite.",
            code="FILE_ALREADY_EXISTS",
        )

    # Create exclusive, uniquely named temporary file in the destination directory
    temp_fd = None
    temp_path = None
    try:
        temp_fd, temp_name = tempfile.mkstemp(
            dir=str(target.parent),
            prefix=f".{target.name}.tmp.",
        )
        temp_path = Path(temp_name)

        # Write text and flush to disk
        with open(temp_fd, "w", encoding=encoding, closefd=True) as f:
            temp_fd = None  # closed by context manager
            f.write(content)
            f.flush()
            os.fsync(f.fileno())

        # Preserve existing permissions if replacing
        if target.exists() and force:
            try:
                shutil.copymode(str(target), str(temp_path))
            except (OSError, NotImplementedError):
                pass

        # Atomic replacement / exclusive move
        if force:
            os.replace(str(temp_path), str(target))
        else:
            # On Windows, os.rename fails with FileExistsError if target exists
            # On POSIX, os.link fails with FileExistsError if target exists
            if os.name == "nt":
                if target.exists():
                    raise DevOpsAIError(
                        f"Destination file '{target}' already exists. Use --force to overwrite.",
                        code="FILE_ALREADY_EXISTS",
                    )
                try:
                    os.rename(str(temp_path), str(target))
                except FileExistsError:
                    raise DevOpsAIError(
                        f"Destination file '{target}' already exists. Use --force to overwrite.",
                        code="FILE_ALREADY_EXISTS",
                    ) from None
            else:
                # POSIX atomic exclusive creation via os.link
                try:
                    os.link(str(temp_path), str(target))
                    os.unlink(str(temp_path))
                except FileExistsError:
                    raise DevOpsAIError(
                        f"Destination file '{target}' already exists. Use --force to overwrite.",
                        code="FILE_ALREADY_EXISTS",
                    ) from None

        # Synchronize directory descriptor on POSIX where supported
        if os.name != "nt" and hasattr(os, "O_DIRECTORY"):
            try:
                dir_fd = os.open(str(target.parent), os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(dir_fd)
                finally:
                    os.close(dir_fd)
            except OSError:
                pass

        return target

    except DevOpsAIError:
        raise
    except Exception as e:
        raise DevOpsAIError(
            f"Failed to write destination file '{target}': {e}",
            code="FILE_WRITE_ERROR",
        ) from None
    finally:
        if temp_fd is not None:
            try:
                os.close(temp_fd)
            except OSError:
                pass
        if temp_path is not None and temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass
