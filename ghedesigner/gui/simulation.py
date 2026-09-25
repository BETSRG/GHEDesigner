from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
import tempfile
import threading
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ghedesigner.gui.log import LOGGER

SimulationCommandFactory = Callable[[Path, Path], Sequence[str]]
TERMINAL_STATUSES = {"completed", "failed", "cancelled"}
ACTIVE_STATUSES = {"queued", "running", "cancelling"}
MAX_OUTPUT_LENGTH = 200_000


class SimulationBusyError(RuntimeError):
    """Raised when a simulation is already running in this GUI process."""


def _timestamp() -> str:
    return datetime.now(UTC).isoformat()


def _default_command(input_path: Path, output_directory: Path) -> Sequence[str]:
    return [sys.executable, "-m", "ghedesigner.main", str(input_path), str(output_directory)]


@dataclass
class SimulationJob:
    id: str
    output_directory: Path
    input_name: str
    status: str = "queued"
    output: str = ""
    error: str | None = None
    return_code: int | None = None
    created_at: str = field(default_factory=_timestamp)
    started_at: str | None = None
    finished_at: str | None = None
    cancel_requested: bool = False
    process: subprocess.Popen[str] | None = field(default=None, repr=False)

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "status": self.status,
            "output_directory": str(self.output_directory),
            "input_name": self.input_name,
            "output": self.output,
            "error": self.error,
            "return_code": self.return_code,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }


class SimulationManager:
    """Run one GUI-launched simulation at a time without blocking request threads."""

    def __init__(self, command_factory: SimulationCommandFactory | None = None) -> None:
        self._command_factory = command_factory or _default_command
        self._jobs: dict[str, SimulationJob] = {}
        self._lock = threading.RLock()

    def start(
        self,
        document: dict[str, Any],
        output_directory: str,
        input_name: str = "ghedesigner-input.json",
    ) -> dict[str, Any]:
        output_path = Path(output_directory).expanduser().resolve()
        if output_path.exists() and not output_path.is_dir():
            raise ValueError("The output path exists and is not a directory.")
        try:
            output_path.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            raise ValueError(f"Unable to create the output directory: {error}") from error

        safe_name = Path(input_name).name or "ghedesigner-input.json"
        if not safe_name.lower().endswith(".json"):
            safe_name += ".json"

        with self._lock:
            if any(job.status in ACTIVE_STATUSES for job in self._jobs.values()):
                raise SimulationBusyError("Another simulation is already running.")
            job = SimulationJob(uuid.uuid4().hex, output_path, safe_name)
            self._jobs[job.id] = job
            self._prune_jobs()

        worker = threading.Thread(
            target=self._run,
            args=(job.id, copy.deepcopy(document)),
            name=f"simulation-{job.id[:8]}",
            daemon=True,
        )
        worker.start()
        LOGGER.info(
            "simulation_queued job_id=%s input_name=%s output_directory=%s",
            job.id,
            job.input_name,
            job.output_directory,
        )
        return job.as_dict()

    def get(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return copy.deepcopy(job.as_dict()) if job is not None else None

    def cancel(self, job_id: str) -> dict[str, Any] | None:
        process: subprocess.Popen[str] | None = None
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            if job.status in TERMINAL_STATUSES:
                return copy.deepcopy(job.as_dict())
            job.cancel_requested = True
            process = job.process
            if process is None and job.status == "queued":
                job.status = "cancelled"
                job.finished_at = _timestamp()
            else:
                job.status = "cancelling"
            snapshot = copy.deepcopy(job.as_dict())

        if process is not None and process.poll() is None:
            process.terminate()
        LOGGER.info("simulation_cancel_requested job_id=%s", job_id)
        return snapshot

    def shutdown(self) -> None:
        with self._lock:
            active_ids = [job.id for job in self._jobs.values() if job.status in ACTIVE_STATUSES]
        for job_id in active_ids:
            self.cancel(job_id)

    def _append_output(self, job: SimulationJob, value: str) -> None:
        combined = job.output + value
        if len(combined) > MAX_OUTPUT_LENGTH:
            marker = "[Earlier process output truncated]\n"
            combined = marker + combined[-(MAX_OUTPUT_LENGTH - len(marker)) :]
        job.output = combined

    def _run(self, job_id: str, document: dict[str, Any]) -> None:
        with self._lock:
            job = self._jobs[job_id]
            if job.cancel_requested:
                return
            job.status = "running"
            job.started_at = _timestamp()
            self._append_output(job, f"Starting GHEDesigner simulation\nOutput directory: {job.output_directory}\n\n")

        try:
            with tempfile.TemporaryDirectory(prefix="ghedesigner-gui-") as temporary_directory:
                input_path = Path(temporary_directory) / job.input_name
                input_path.write_text(json.dumps(document, indent=2) + "\n")
                command = list(self._command_factory(input_path, job.output_directory))
                LOGGER.info("simulation_started job_id=%s command=%s", job_id, command[0])
                process = subprocess.Popen(  # noqa: S603
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    env={**os.environ, "PYTHONUNBUFFERED": "1"},
                )
                with self._lock:
                    job.process = process
                    should_cancel = job.cancel_requested
                if should_cancel and process.poll() is None:
                    process.terminate()
                if process.stdout is not None:
                    for line in process.stdout:
                        with self._lock:
                            self._append_output(job, line)
                return_code = process.wait()

            with self._lock:
                job.process = None
                job.return_code = return_code
                job.finished_at = _timestamp()
                if job.cancel_requested:
                    job.status = "cancelled"
                    self._append_output(job, "\nSimulation cancelled.\n")
                elif return_code == 0:
                    job.status = "completed"
                    self._append_output(job, "\nSimulation completed successfully.\n")
                else:
                    job.status = "failed"
                    job.error = f"GHEDesigner exited with status {return_code}."
                    self._append_output(job, f"\n{job.error}\n")
                LOGGER.info(
                    "simulation_finished job_id=%s status=%s return_code=%s",
                    job_id,
                    job.status,
                    return_code,
                )
        except Exception as error:  # noqa: BLE001
            with self._lock:
                job.process = None
                job.status = "cancelled" if job.cancel_requested else "failed"
                job.error = None if job.cancel_requested else str(error)
                job.finished_at = _timestamp()
                self._append_output(job, f"\nSimulation {job.status}: {error}\n")
            LOGGER.exception("simulation_failed job_id=%s", job_id)

    def _prune_jobs(self) -> None:
        terminal_jobs = [job for job in self._jobs.values() if job.status in TERMINAL_STATUSES]
        for job in sorted(terminal_jobs, key=lambda item: item.created_at)[:-20]:
            self._jobs.pop(job.id, None)
