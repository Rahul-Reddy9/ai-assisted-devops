# Copyright 2026 Rahul Reddy
# Licensed under the Apache License, Version 2.0
"""Multi-process integration tests verifying OS-level repository locking.

Validates that independent operating system processes achieve mutual exclusion
using platform-specific advisory locks (msvcrt on Windows, fcntl on Linux),
with automatic kernel lock release on process termination and zero file deletion.
"""

import subprocess
import sys
from pathlib import Path
import pytest

from devops_ai.core.exceptions import RepositoryLockError
from devops_ai.incident.repository import IncidentRepository


def test_mp_blocking_and_acquisition(tmp_path: Path):
    """Verify that a second process is blocked while the first owns the OS lock,

    and can acquire it once the first process finishes.
    """
    storage_dir = str(tmp_path.resolve())
    repo = IncidentRepository(storage_dir=tmp_path)

    code_child = f"""
import sys, time
from devops_ai.incident.repository import IncidentRepository
repo = IncidentRepository(storage_dir={repr(storage_dir)})
with repo._file_lock(timeout=5.0):
    print("CHILD_LOCKED", flush=True)
    time.sleep(1.5)
print("CHILD_RELEASED", flush=True)
"""

    p = subprocess.Popen(
        [sys.executable, "-c", code_child],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    try:
        # Wait until child confirms it holds the lock
        line = p.stdout.readline().strip()
        assert line == "CHILD_LOCKED"

        # While child holds the lock, parent attempting with short timeout MUST fail
        with pytest.raises(RepositoryLockError) as exc_info:
            with repo._file_lock(timeout=0.2):
                pass
        assert "Timed out after 0.2s" in str(exc_info.value)

        # Parent with sufficient timeout must wait and acquire after child finishes
        with repo._file_lock(timeout=5.0):
            acquired_after = True

        assert acquired_after is True
        p.wait(timeout=5.0)
        assert p.returncode == 0
    finally:
        if p.poll() is None:
            p.kill()


def test_mp_auto_lock_release_on_process_termination(tmp_path: Path):
    """Verify that the operating system kernel automatically releases the lock

    when the owning process is abruptly terminated (crashed/killed).
    """
    storage_dir = str(tmp_path.resolve())
    repo = IncidentRepository(storage_dir=tmp_path)

    # Child acquires lock and sleeps indefinitely until killed
    code_child = f"""
import sys, time
from devops_ai.incident.repository import IncidentRepository
repo = IncidentRepository(storage_dir={repr(storage_dir)})
with repo._file_lock(timeout=5.0):
    print("CHILD_HOLDING_FOREVER", flush=True)
    time.sleep(30.0)
"""

    p = subprocess.Popen(
        [sys.executable, "-c", code_child],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    try:
        line = p.stdout.readline().strip()
        assert line == "CHILD_HOLDING_FOREVER"

        # Verify parent is blocked while child is alive
        with pytest.raises(RepositoryLockError):
            with repo._file_lock(timeout=0.2):
                pass

        # Abruptly terminate the child process (simulating sudden crash/kill -9)
        p.kill()
        p.wait(timeout=5.0)

        # OS kernel must have automatically released the lock; parent must acquire immediately
        with repo._file_lock(timeout=2.0):
            acquired_after_crash = True

        assert acquired_after_crash is True
        assert repo.lock_file.exists(), "Lock file must remain persistent on disk"
    finally:
        if p.poll() is None:
            p.kill()


def test_mp_incomplete_metadata_while_lock_held(tmp_path: Path):
    """Verify that incomplete metadata while the OS lock is held does not compromise

    mutual exclusion, and the lock is never prematurely broken.
    """
    storage_dir = str(tmp_path.resolve())
    repo = IncidentRepository(storage_dir=tmp_path)

    code_child = f"""
import sys, time
from devops_ai.incident.repository import IncidentRepository
repo = IncidentRepository(storage_dir={repr(storage_dir)})
with repo._file_lock(timeout=5.0):
    # Overwrite diagnostic metadata with corrupted/incomplete text while holding the OS lock
    with open(repo.lock_file, "r+b") as f:
        f.seek(1)
        f.write(b'{{"pid": 9999, "corrupted": ')
        f.truncate()
        f.flush()
    print("METADATA_CORRUPTED_WHILE_LOCKED", flush=True)
    time.sleep(1.0)
"""

    p = subprocess.Popen(
        [sys.executable, "-c", code_child],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    try:
        line = p.stdout.readline().strip()
        assert line == "METADATA_CORRUPTED_WHILE_LOCKED"

        # Parent must be blocked by the OS lock, NOT delete the lock based on corrupted metadata
        with pytest.raises(RepositoryLockError):
            with repo._file_lock(timeout=0.2):
                pass

        p.wait(timeout=5.0)
        assert p.returncode == 0

        # After child exits, parent acquires cleanly and writes fresh metadata
        with repo._file_lock(timeout=2.0):
            pass

        assert repo.lock_file.exists()
    finally:
        if p.poll() is None:
            p.kill()


def test_mp_concurrent_incident_writes_no_corruption(tmp_path: Path):
    """Verify multiple independent processes concurrently saving incidents to the same

    repository directory produce zero data loss, zero lost updates, and valid JSON.
    """
    storage_dir = str(tmp_path.resolve())
    num_processes = 4
    incidents_per_process = 5

    processes = []
    for proc_idx in range(num_processes):
        code_worker = f"""
import sys, datetime
from devops_ai.incident.repository import IncidentRepository
from devops_ai.incident.models import Incident, Severity, Category, IncidentStatus

repo = IncidentRepository(storage_dir={repr(storage_dir)})
for i in range({incidents_per_process}):
    now = datetime.datetime.now(datetime.timezone.utc)
    inc = Incident(
        incident_id=f"INC-P{proc_idx}-{{i}}",
        version=1,
        title=f"Incident from Process {proc_idx} item {{i}}",
        category=Category.INFRASTRUCTURE,
        severity=Severity.MEDIUM,
        status=IncidentStatus.DETECTED,
        created_at=now,
        updated_at=now,
    )
    repo.save(inc)
"""
        p = subprocess.Popen(
            [sys.executable, "-c", code_worker],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        processes.append(p)

    # Wait for all processes to complete
    for p in processes:
        stdout, stderr = p.communicate(timeout=30.0)
        assert p.returncode == 0, f"Worker process failed: {stderr}"

    # Verify repository content
    repo = IncidentRepository(storage_dir=tmp_path)
    all_incidents = repo.list_all(limit=100)

    # Total incidents must equal num_processes * incidents_per_process = 20
    assert len(all_incidents) == num_processes * incidents_per_process
    stored_ids = {inc.incident_id for inc in all_incidents}

    for proc_idx in range(num_processes):
        for i in range(incidents_per_process):
            expected_id = f"INC-P{proc_idx}-{i}"
            assert expected_id in stored_ids, f"Missing incident {expected_id}"
