# Copyright 2026 Rahul Reddy
# Licensed under the Apache License, Version 2.0
"""Read-only Docker container monitoring with graceful offline degradation."""

from dataclasses import dataclass, field
import datetime
import json
import subprocess
from typing import List, Optional

from devops_ai.core.logging import _scrub_secrets, get_logger
from devops_ai.observability.metrics import record_docker_status

logger = get_logger("observability.docker_monitor")


@dataclass(frozen=True)
class ContainerInfo:
    """Summary of a single Docker container."""

    container_id: str
    name: str
    image: str
    status: str
    state: str
    created: str
    ports: str


@dataclass(frozen=True)
class DockerStatus:
    """Consolidated Docker daemon connectivity and container inventory."""

    is_available: bool
    daemon_version: Optional[str] = None
    containers: List[ContainerInfo] = field(default_factory=list)
    error: Optional[str] = None
    timestamp: str = field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()
    )


class DockerMonitor:
    """Provides safe, read-only inspection of local Docker daemon and containers."""

    def __init__(self, cli_command: str = "docker") -> None:
        self.cli_command = cli_command

    def check_availability(self, timeout: float = 3.0) -> bool:
        """Probe Docker daemon availability without modifying state."""
        try:
            cmd = [self.cli_command, "version", "--format", "{{.Server.Version}}"]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
            available = result.returncode == 0 and bool(result.stdout.strip())
            record_docker_status(available)
            return available
        except (subprocess.TimeoutExpired, FileNotFoundError, Exception) as exc:
            logger.debug(f"Docker availability check failed: {exc}")
            record_docker_status(False)
            return False

    def get_status(
        self,
        all_containers: bool = True,
        timeout: float = 5.0,
    ) -> DockerStatus:
        """Inspect running or all containers in a strictly read-only manner."""
        # 1. Check daemon version
        try:
            ver_cmd = [self.cli_command, "version", "--format", "{{.Server.Version}}"]
            ver_res = subprocess.run(
                ver_cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
            if ver_res.returncode != 0:
                err_msg = _scrub_secrets(ver_res.stderr.strip() or "Docker daemon unreachable")
                record_docker_status(False)
                return DockerStatus(
                    is_available=False,
                    error=f"Docker daemon unreachable: {err_msg}",
                )
            daemon_version = ver_res.stdout.strip()
        except FileNotFoundError:
            record_docker_status(False)
            return DockerStatus(
                is_available=False,
                error="Docker CLI executable not found on system PATH",
            )
        except subprocess.TimeoutExpired:
            record_docker_status(False)
            return DockerStatus(
                is_available=False,
                error="Docker daemon probe timed out",
            )
        except Exception as exc:
            record_docker_status(False)
            return DockerStatus(
                is_available=False,
                error=f"Failed to query Docker daemon: {_scrub_secrets(str(exc))}",
            )

        # 2. Query container list
        cmd = [self.cli_command, "ps", "--format", "{{json .}}"]
        if all_containers:
            cmd.insert(2, "-a")

        try:
            ps_res = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
            if ps_res.returncode != 0:
                err_msg = _scrub_secrets(ps_res.stderr.strip() or "Failed to query containers")
                record_docker_status(False)
                return DockerStatus(
                    is_available=False,
                    daemon_version=daemon_version,
                    error=f"Failed to query containers: {err_msg}",
                )

            containers: List[ContainerInfo] = []
            for line in ps_res.stdout.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    raw_id = str(data.get("ID", ""))
                    containers.append(
                        ContainerInfo(
                            container_id=raw_id[:12] if raw_id else "",
                            name=str(data.get("Names", "")),
                            image=str(data.get("Image", "")),
                            status=str(data.get("Status", "")),
                            state=str(data.get("State", "")),
                            created=str(data.get("CreatedAt", "")),
                            ports=str(data.get("Ports", "")),
                        )
                    )
                except json.JSONDecodeError:
                    logger.debug(f"Skipping unparseable docker ps output line: {line}")
                    continue

            record_docker_status(True)
            return DockerStatus(
                is_available=True,
                daemon_version=daemon_version,
                containers=containers,
            )
        except subprocess.TimeoutExpired:
            record_docker_status(False)
            return DockerStatus(
                is_available=False,
                daemon_version=daemon_version,
                error="Docker container listing timed out",
            )
        except Exception as exc:
            record_docker_status(False)
            return DockerStatus(
                is_available=False,
                daemon_version=daemon_version,
                error=f"Error inspecting containers: {_scrub_secrets(str(exc))}",
            )
