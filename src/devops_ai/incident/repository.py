# Copyright 2026 Rahul Reddy
# Licensed under the Apache License, Version 2.0
"""High-reliability, thread-safe, and cross-platform incident storage repository."""

from contextlib import contextmanager
import datetime
import json
import os
from pathlib import Path
import random
import socket
import threading
import time
from typing import Any, Dict, Generator, List, Optional, Union
import uuid

from devops_ai.core.config import get_settings
from devops_ai.core.exceptions import (
    ConcurrencyConflictError,
    IncidentError,
    IncidentNotFoundError,
    RepositoryLockError,
)
from devops_ai.core.file_io import atomic_write_text
from devops_ai.core.logging import get_logger
from devops_ai.incident.models import Incident, IncidentStatus, Severity

logger = get_logger("incident.repository")


def _get_process_create_time(pid: int) -> Optional[float]:
    """Retrieve process creation timestamp to protect against PID reuse."""
    try:
        import psutil

        proc = psutil.Process(pid)
        return float(proc.create_time())
    except Exception:
        return None


def _acquire_os_lock(fd: int) -> bool:
    """Attempt non-blocking acquisition of OS-level advisory lock on file descriptor."""
    if os.name == "nt":
        import msvcrt

        try:
            os.lseek(fd, 0, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
            return True
        except (OSError, PermissionError):
            return False
    else:
        import fcntl

        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except (BlockingIOError, OSError):
            return False


def _release_os_lock(fd: int) -> None:
    """Release OS-level advisory lock on file descriptor."""
    if os.name == "nt":
        import msvcrt

        try:
            os.lseek(fd, 0, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
        except (OSError, PermissionError):
            pass
    else:
        import fcntl

        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        except OSError:
            pass


class IncidentRepository:
    """Thread-safe and cross-process safe persistence layer for operational incidents."""

    def __init__(self, storage_dir: Optional[Union[str, Path]] = None) -> None:
        settings = get_settings()
        self.storage_dir = Path(storage_dir or settings.incident_storage_dir).resolve()
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        self.primary_file = self.storage_dir / "incidents.json"
        self.backup_file = self.storage_dir / "incidents.json.bak"
        self.lock_file = self.storage_dir / ".incidents.lock"

        self._in_memory_lock = threading.RLock()
        self._retention_days = settings.incident_retention_days
        self._max_records = settings.incident_max_records

    def _ensure_lock_file_exists(self) -> None:
        """Ensure the persistent lock file exists with at least 1 byte for byte-range locking."""
        if not self.lock_file.exists():
            try:
                with open(self.lock_file, "a+b") as f:
                    if f.tell() == 0:
                        f.write(b" ")
                        f.flush()
                        os.fsync(f.fileno())
            except OSError:
                pass

    @contextmanager
    def _file_lock(self, timeout: float = 5.0) -> Generator[str, None, None]:
        """Cross-process advisory OS-level lock (msvcrt on Windows, fcntl on Linux).

        The lock file is persistent and NEVER unlinked. Mutual exclusion is enforced
        exclusively by the OS kernel, which automatically frees the lock on process termination.
        """
        self._ensure_lock_file_exists()

        token = str(uuid.uuid4())
        pid = os.getpid()
        hostname = socket.gethostname()
        create_time = _get_process_create_time(pid)

        start_time = time.monotonic()
        lock_file_handle = None
        acquired = False

        while not acquired:
            try:
                # Open in read-write binary mode without truncating
                lock_file_handle = open(self.lock_file, "r+b")
                if _acquire_os_lock(lock_file_handle.fileno()):
                    acquired = True
                    # Write diagnostic metadata (offset 1 onwards, leaving lock byte 0 intact)
                    try:
                        payload = json.dumps(
                            {
                                "pid": pid,
                                "create_time": create_time,
                                "hostname": hostname,
                                "token": token,
                                "acquired_at": time.time(),
                            }
                        ).encode("utf-8")
                        lock_file_handle.seek(1)
                        lock_file_handle.write(payload)
                        lock_file_handle.truncate()
                        lock_file_handle.flush()
                    except Exception as meta_err:
                        logger.debug(
                            f"Failed to write diagnostic lock metadata (non-fatal): {meta_err}"
                        )
                    break
                else:
                    lock_file_handle.close()
                    lock_file_handle = None
            except OSError:
                if lock_file_handle:
                    try:
                        lock_file_handle.close()
                    except OSError:
                        pass
                    lock_file_handle = None

            if time.monotonic() - start_time >= timeout:
                diag_info = ""
                try:
                    if self.lock_file.exists():
                        with open(self.lock_file, "rb") as mf:
                            mf.seek(1)
                            meta_bytes = mf.read()
                            if meta_bytes:
                                meta_obj = json.loads(meta_bytes.decode("utf-8"))
                                diag_info = f" (held by PID {meta_obj.get('pid')} on {meta_obj.get('hostname')})"
                except Exception:
                    pass

                raise RepositoryLockError(
                    f"Timed out after {timeout:.1f}s waiting for repository OS lock at '{self.lock_file}'{diag_info}"
                ) from None

            time.sleep(random.uniform(0.02, 0.08))

        try:
            yield token
        finally:
            if lock_file_handle:
                try:
                    _release_os_lock(lock_file_handle.fileno())
                finally:
                    try:
                        lock_file_handle.close()
                    except OSError:
                        pass

    def _read_data_unlocked(self) -> Dict[str, Dict[str, Any]]:
        """Read and validate incident dictionary from disk with backup recovery."""
        if not self.primary_file.exists():
            return {}

        try:
            raw = self.primary_file.read_text(encoding="utf-8")
            if not raw.strip():
                return {}
            data = json.loads(raw)
            if isinstance(data, dict):
                return data
            raise ValueError("Corrupted repository: root JSON element is not a dictionary.")
        except (json.JSONDecodeError, ValueError, OSError) as primary_err:
            logger.critical(
                f"Primary incident storage '{self.primary_file}' corrupted ({primary_err}). "
                "Attempting automatic recovery from backup..."
            )
            if self.backup_file.exists():
                try:
                    bak_raw = self.backup_file.read_text(encoding="utf-8")
                    data = json.loads(bak_raw)
                    if isinstance(data, dict):
                        # Restore primary file from valid backup
                        atomic_write_text(self.primary_file, bak_raw, force=True)
                        logger.warning(
                            f"Successfully recovered {len(data)} incident records from backup."
                        )
                        return data
                except Exception as bak_err:
                    logger.error(f"Backup restoration also failed: {bak_err}")

            raise IncidentError(
                f"Failed to load incident repository: primary store and backup are unrecoverable ({primary_err})"
            ) from None

    def _write_data_unlocked(self, data: Dict[str, Dict[str, Any]]) -> None:
        """Write incident records atomically, maintaining a backup copy."""
        content = json.dumps(data, indent=2, default=str)

        # Maintain backup copy before replacing primary
        if self.primary_file.exists():
            try:
                existing = self.primary_file.read_text(encoding="utf-8")
                if existing.strip():
                    atomic_write_text(self.backup_file, existing, force=True)
            except OSError as e:
                logger.debug(f"Failed to update backup file before save: {e}")

        # Atomically replace primary store
        atomic_write_text(self.primary_file, content, force=True)

    def save(self, incident: Incident) -> Incident:
        """Persist or update an incident record with optimistic concurrency validation."""
        with self._in_memory_lock, self._file_lock():
            data = self._read_data_unlocked()
            existing_raw = data.get(incident.incident_id)

            if existing_raw:
                expected_version = existing_raw.get("version", 1)
                if incident.version != expected_version:
                    raise ConcurrencyConflictError(
                        incident_id=incident.incident_id,
                        expected=expected_version,
                        actual=incident.version,
                    )
                # Increment version on update
                incident.version += 1
                incident.updated_at = datetime.datetime.now(datetime.timezone.utc)
            else:
                incident.updated_at = datetime.datetime.now(datetime.timezone.utc)

            data[incident.incident_id] = incident.model_dump(mode="json")
            self._write_data_unlocked(data)
            return incident

    def get(self, incident_id: str) -> Optional[Incident]:
        """Retrieve an incident by unique identifier."""
        with self._in_memory_lock, self._file_lock():
            data = self._read_data_unlocked()
            raw = data.get(incident_id)
            if not raw:
                return None
            return Incident.model_validate(raw)

    def get_or_raise(self, incident_id: str) -> Incident:
        """Retrieve an incident by unique identifier or raise IncidentNotFoundError."""
        inc = self.get(incident_id)
        if not inc:
            raise IncidentNotFoundError(incident_id)
        return inc

    def list_all(
        self,
        status: Optional[IncidentStatus] = None,
        severity: Optional[Severity] = None,
        active_only: bool = False,
        limit: int = 100,
    ) -> List[Incident]:
        """Query stored incidents filtered by status, severity, or activity."""
        with self._in_memory_lock, self._file_lock():
            data = self._read_data_unlocked()
            results: List[Incident] = []

            for raw in data.values():
                try:
                    inc = Incident.model_validate(raw)
                    if status and inc.status != status:
                        continue
                    if severity and inc.severity != severity:
                        continue
                    if active_only and inc.status in (
                        IncidentStatus.RESOLVED,
                        IncidentStatus.CLOSED,
                    ):
                        continue
                    results.append(inc)
                except Exception as e:
                    logger.debug(f"Skipping malformed incident record in list_all: {e}")

            # Sort chronologically (most recently updated first)
            results.sort(key=lambda x: x.updated_at, reverse=True)
            return results[:limit]

    def delete(self, incident_id: str) -> bool:
        """Delete an incident record from storage."""
        with self._in_memory_lock, self._file_lock():
            data = self._read_data_unlocked()
            if incident_id not in data:
                return False
            del data[incident_id]
            self._write_data_unlocked(data)
            return True

    def prune_retention(self) -> int:
        """Prune closed incidents older than the configured retention period."""
        cutoff = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(
            days=self._retention_days
        )
        pruned_count = 0

        with self._in_memory_lock, self._file_lock():
            data = self._read_data_unlocked()
            to_delete: List[str] = []

            for inc_id, raw in data.items():
                if raw.get("status") == IncidentStatus.CLOSED.value:
                    closed_at_str = raw.get("closed_at") or raw.get("updated_at")
                    if closed_at_str:
                        try:
                            closed_dt = datetime.datetime.fromisoformat(closed_at_str)
                            if closed_dt < cutoff:
                                to_delete.append(inc_id)
                        except (ValueError, TypeError):
                            pass

            for inc_id in to_delete:
                del data[inc_id]
                pruned_count += 1

            if pruned_count > 0:
                self._write_data_unlocked(data)
                logger.info(
                    f"Pruned {pruned_count} closed incidents older than {self._retention_days} days."
                )

            return pruned_count
