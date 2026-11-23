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


def _is_pid_alive(pid: int, expected_create_time: Optional[float] = None) -> bool:
    """Verify whether a process is currently running, accounting for PID reuse."""
    try:
        import psutil

        if not psutil.pid_exists(pid):
            return False
        if expected_create_time is not None:
            try:
                proc = psutil.Process(pid)
                # If creation times differ by more than 2 seconds, the PID was reused
                if abs(proc.create_time() - expected_create_time) > 2.0:
                    return False
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                return False
        return True
    except ImportError:
        if os.name == "nt":
            import ctypes

            handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)
            if handle == 0:
                return False
            ctypes.windll.kernel32.CloseHandle(handle)
            return True
        else:
            try:
                os.kill(pid, 0)
                return True
            except OSError:
                return False


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

    @contextmanager
    def _file_lock(self, timeout: float = 5.0) -> Generator[str, None, None]:
        """Cross-platform advisory file lock acquisition with PID verification."""
        token = str(uuid.uuid4())
        hostname = socket.gethostname()
        pid = os.getpid()
        create_time = _get_process_create_time(pid)

        payload = {
            "pid": pid,
            "create_time": create_time,
            "hostname": hostname,
            "token": token,
            "acquired_at": time.time(),
        }

        start_time = time.monotonic()
        acquired = False

        while not acquired:
            try:
                # Atomic creation primitive on both Windows and POSIX (O_CREAT | O_EXCL)
                with open(self.lock_file, "x", encoding="utf-8") as f:
                    json.dump(payload, f)
                    f.flush()
                    os.fsync(f.fileno())
                acquired = True
            except FileExistsError:
                # Inspect existing lock for staleness or dead owning process
                try:
                    is_stale = False
                    if self.lock_file.exists():
                        try:
                            content = self.lock_file.read_text(encoding="utf-8").strip()
                            if not content:
                                # Incomplete write: check mtime
                                mtime = self.lock_file.stat().st_mtime
                                if (time.time() - mtime) > 5.0:
                                    is_stale = True
                            else:
                                lock_data = json.loads(content)
                                lock_pid = lock_data.get("pid")
                                lock_time = lock_data.get("create_time")
                                lock_host = lock_data.get("hostname")

                                # Check process liveness if on the same host
                                if lock_host == hostname and isinstance(lock_pid, int):
                                    if not _is_pid_alive(lock_pid, lock_time):
                                        logger.warning(
                                            f"Breaking stale repository lock: owning PID {lock_pid} is dead."
                                        )
                                        is_stale = True
                                elif (time.time() - lock_data.get("acquired_at", 0)) > 60.0:
                                    # Fallback for locks from inaccessible hosts older than 60s
                                    is_stale = True
                        except (json.JSONDecodeError, OSError):
                            mtime = self.lock_file.stat().st_mtime
                            if (time.time() - mtime) > 5.0:
                                is_stale = True

                    if is_stale:
                        try:
                            self.lock_file.unlink(missing_ok=True)
                            continue
                        except OSError:
                            pass
                except OSError:
                    pass

                if time.monotonic() - start_time >= timeout:
                    raise RepositoryLockError(
                        f"Timed out after {timeout:.1f}s waiting for repository lock at '{self.lock_file}'"
                    ) from None

                # Sleep with random jitter to avoid thundering herd problem
                time.sleep(random.uniform(0.05, 0.15))

        try:
            yield token
        finally:
            try:
                if self.lock_file.exists():
                    try:
                        content = self.lock_file.read_text(encoding="utf-8").strip()
                        if content:
                            stored = json.loads(content)
                            if stored.get("token") == token:
                                self.lock_file.unlink(missing_ok=True)
                    except (json.JSONDecodeError, OSError):
                        pass
            except Exception as e:
                logger.debug(f"Failed to release file lock cleanly: {e}")

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
