# Copyright 2026 Rahul Reddy
# Licensed under the Apache License, Version 2.0
"""Unit tests verifying IncidentRepository reliability, concurrency, and corruption recovery."""

from concurrent.futures import ThreadPoolExecutor
import datetime
import os
import time
import pytest

from devops_ai.core.exceptions import ConcurrencyConflictError
from devops_ai.incident.models import Category, Incident, IncidentStatus, Severity
from devops_ai.incident.repository import IncidentRepository, _is_pid_alive


def test_repository_save_and_retrieve(tmp_path):
    """Verify basic save, get, and list operations in IncidentRepository."""
    repo = IncidentRepository(storage_dir=tmp_path)
    now = datetime.datetime.now(datetime.timezone.utc)

    inc = Incident(
        incident_id="INC-TEST-001",
        version=1,
        title="Test Incident",
        category=Category.INFRASTRUCTURE,
        severity=Severity.HIGH,
        status=IncidentStatus.DETECTED,
        created_at=now,
        updated_at=now,
    )

    saved = repo.save(inc)
    assert saved.incident_id == "INC-TEST-001"
    assert saved.version == 1

    retrieved = repo.get("INC-TEST-001")
    assert retrieved is not None
    assert retrieved.incident_id == "INC-TEST-001"
    assert retrieved.title == "Test Incident"


def test_optimistic_concurrency_conflict(tmp_path):
    """Verify ConcurrencyConflictError when saving with outdated version."""
    repo = IncidentRepository(storage_dir=tmp_path)
    now = datetime.datetime.now(datetime.timezone.utc)

    inc = Incident(
        incident_id="INC-CONFLICT-001",
        version=1,
        title="Initial version",
        created_at=now,
        updated_at=now,
    )
    repo.save(inc)

    # First update: version increments to 2
    inc.title = "First update"
    repo.save(inc)
    assert inc.version == 2

    # Attempt to save a stale copy with version 1
    stale_copy = Incident(
        incident_id="INC-CONFLICT-001",
        version=1,
        title="Stale update",
        created_at=now,
        updated_at=now,
    )

    with pytest.raises(ConcurrencyConflictError) as exc_info:
        repo.save(stale_copy)

    assert exc_info.value.expected == 2
    assert exc_info.value.actual == 1


def test_corrupted_json_recovery_from_backup(tmp_path):
    """Verify repository automatically restores from .bak if primary JSON is corrupted."""
    repo = IncidentRepository(storage_dir=tmp_path)
    now = datetime.datetime.now(datetime.timezone.utc)

    # 1. Create first valid incident
    inc1 = Incident(
        incident_id="INC-RECOVER-001",
        version=1,
        title="First Incident",
        created_at=now,
        updated_at=now,
    )
    repo.save(inc1)

    # 2. Create second valid incident (triggers backup creation of first state)
    inc2 = Incident(
        incident_id="INC-RECOVER-002",
        version=1,
        title="Second Incident",
        created_at=now,
        updated_at=now,
    )
    repo.save(inc2)

    assert repo.backup_file.exists()

    # 3. Corrupt the primary file with invalid JSON syntax (e.g. truncated file)
    with open(repo.primary_file, "w", encoding="utf-8") as f:
        f.write('{"INC-RECOVER-001": {"incomplete_data": ')

    # 4. Attempt to read -> repository must catch corruption and restore from .bak
    recovered_inc = repo.get("INC-RECOVER-001")
    assert recovered_inc is not None
    assert recovered_inc.incident_id == "INC-RECOVER-001"


def test_concurrent_multithreaded_writes(tmp_path):
    """Verify thread-safety when multiple threads create incidents concurrently."""
    repo = IncidentRepository(storage_dir=tmp_path)

    def create_incident(index: int):
        now = datetime.datetime.now(datetime.timezone.utc)
        inc = Incident(
            incident_id=f"INC-CONCUR-{index:03d}",
            version=1,
            title=f"Thread Incident {index}",
            category=Category.APPLICATION,
            severity=Severity.LOW,
            status=IncidentStatus.DETECTED,
            created_at=now,
            updated_at=now,
        )
        repo.save(inc)
        return inc.incident_id

    # Run 10 concurrent threads
    with ThreadPoolExecutor(max_workers=5) as executor:
        results = list(executor.map(create_incident, range(10)))

    assert len(results) == 10
    all_incidents = repo.list_all(limit=50)
    assert len(all_incidents) == 10


def test_incomplete_stale_lock_recovery(tmp_path):
    """Verify repository safely recovers when an incomplete/empty lock file is older than 5s."""
    repo = IncidentRepository(storage_dir=tmp_path)

    # Create an abandoned empty lock file with old mtime
    repo.lock_file.write_text("", encoding="utf-8")
    old_time = time.time() - 10.0
    os.utime(repo.lock_file, (old_time, old_time))

    now = datetime.datetime.now(datetime.timezone.utc)
    inc = Incident(
        incident_id="INC-LOCK-001",
        version=1,
        title="Lock Recovery Test",
        created_at=now,
        updated_at=now,
    )

    # Saving must detect stale empty lock, evict it, and succeed
    saved = repo.save(inc)
    assert saved.incident_id == "INC-LOCK-001"


def test_pid_liveness_check():
    """Verify _is_pid_alive correctly identifies current process and nonexistent PID."""
    current_pid = os.getpid()
    assert _is_pid_alive(current_pid) is True

    # High PID unlikely to exist
    nonexistent_pid = 99999999
    assert _is_pid_alive(nonexistent_pid) is False


def test_retention_pruning(tmp_path):
    """Verify closed incidents older than retention days are pruned."""
    repo = IncidentRepository(storage_dir=tmp_path)
    now = datetime.datetime.now(datetime.timezone.utc)

    # 1. Closed incident 40 days ago (should be pruned with 30-day retention)
    old_closed = Incident(
        incident_id="INC-OLD-CLOSED",
        version=1,
        title="Old Incident",
        status=IncidentStatus.CLOSED,
        created_at=now - datetime.timedelta(days=45),
        updated_at=now - datetime.timedelta(days=40),
        closed_at=now - datetime.timedelta(days=40),
    )
    repo.save(old_closed)

    # 2. Closed incident 5 days ago (should NOT be pruned)
    recent_closed = Incident(
        incident_id="INC-RECENT-CLOSED",
        version=1,
        title="Recent Closed",
        status=IncidentStatus.CLOSED,
        created_at=now - datetime.timedelta(days=10),
        updated_at=now - datetime.timedelta(days=5),
        closed_at=now - datetime.timedelta(days=5),
    )
    repo.save(recent_closed)

    # 3. Active incident (should NEVER be pruned)
    active_inc = Incident(
        incident_id="INC-ACTIVE",
        version=1,
        title="Active Incident",
        status=IncidentStatus.DETECTED,
        created_at=now - datetime.timedelta(days=60),
        updated_at=now,
    )
    repo.save(active_inc)

    pruned = repo.prune_retention()
    assert pruned == 1

    remaining = repo.list_all()
    remaining_ids = [i.incident_id for i in remaining]
    assert "INC-OLD-CLOSED" not in remaining_ids
    assert "INC-RECENT-CLOSED" in remaining_ids
    assert "INC-ACTIVE" in remaining_ids
