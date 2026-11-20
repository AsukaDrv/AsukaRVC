"""Background job manager for the custom WebUI REST API.

Long-running tasks (AI cover inference, model downloads, training steps) run
in worker threads so HTTP requests stay fast; the frontend polls the job
endpoint for status, progress log and the final result.

Only ONE job runs at a time: training subprocesses use the module-global
process handle in ``ui.utils.jobs`` and inference needs the whole GPU, so
concurrent heavy tasks are rejected with :class:`BusyError` (HTTP 409).
"""

import logging
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field

import gradio as gr

from ui.utils.jobs import request_stop, run_job

MAX_LOG_CHARS = 20000  # Keep the log sent to the browser bounded
MAX_JOBS_KEPT = 40
JOB_TTL_SECONDS = 3600

# Job kinds backed by a subprocess (stoppable immediately via SIGTERM/SIGKILL).
SUBPROCESS_KINDS = {"train-slice", "train-extract", "train-index", "train-start"}


class BusyError(RuntimeError):
    """Raised when a new job is requested while another one is still running."""


@dataclass
class Job:
    id: str
    kind: str
    title: str
    status: str = "pending"  # pending -> running -> finished | stopped | error
    log: str = ""
    result: dict | None = None
    error: str | None = None
    stop_requested: bool = False
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def public(self) -> dict:
        """JSON-safe view returned to the frontend."""
        return {
            "id": self.id,
            "kind": self.kind,
            "title": self.title,
            "status": self.status,
            "log": self.log,
            "result": self.result,
            "error": self.error,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


class _JobLogHandler(logging.Handler):
    """Captures log records emitted by the running task into the job log.

    The pipeline reports progress via ``display_progress`` which, outside the
    Gradio UI, logs through the "core" logger — so capturing log records gives
    the web frontend the same progress feedback the Gradio UI gets.
    """

    def __init__(self, job: Job, manager: "JobManager"):
        super().__init__(level=logging.INFO)
        self._job = job
        self._manager = manager

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = record.getMessage()
        except Exception:  # noqa: BLE001 — never break the app on a bad record
            message = str(record.msg)
        if message.strip():
            self._manager._append_log(self._job, message)


class JobManager:
    """Registry and runner for background jobs (single active job)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jobs: dict[str, Job] = {}
        self._active_id: str | None = None

    # ------------------------------------------------------------------ state

    def _touch(self, job: Job) -> None:
        job.updated_at = time.time()

    def _append_log(self, job: Job, text: str) -> None:
        with self._lock:
            job.log = f"{job.log}\n{text}"[-MAX_LOG_CHARS:] if job.log else text
            self._touch(job)

    def _replace_log(self, job: Job, text: str) -> None:
        with self._lock:
            job.log = text[-MAX_LOG_CHARS:]
            self._touch(job)

    def set_log(self, job: Job, text: str) -> None:
        """Public hook for runners that stream their own progress log."""
        self._replace_log(job, text)

    def _update(self, job: Job, **fields) -> None:
        with self._lock:
            for key, value in fields.items():
                setattr(job, key, value)
            self._touch(job)

    def _prune(self) -> None:
        now = time.time()
        stale = [
            job_id
            for job_id, job in self._jobs.items()
            if job.status in ("finished", "stopped", "error")
            and now - job.updated_at > JOB_TTL_SECONDS
        ]
        for job_id in stale:
            self._jobs.pop(job_id, None)
        while len(self._jobs) > MAX_JOBS_KEPT:
            self._jobs.pop(next(iter(self._jobs)), None)

    # ----------------------------------------------------------------- public

    def active(self) -> Job | None:
        with self._lock:
            job = self._jobs.get(self._active_id) if self._active_id else None
        if job is not None and job.status in ("finished", "stopped", "error"):
            return None
        return job

    def active_public(self) -> dict | None:
        job = self.active()
        return job.public() if job else None

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def register(self, kind: str, title: str) -> Job:
        """Create a job slot; refuse when another heavy task is still running."""
        with self._lock:
            self._prune()
            active = self._jobs.get(self._active_id) if self._active_id else None
            if active is not None and active.status in ("pending", "running"):
                raise BusyError(
                    f"Another task is still running: '{active.title}'. "
                    "Wait for it to finish or stop it first."
                )
            job = Job(id=uuid.uuid4().hex[:12], kind=kind, title=title)
            self._jobs[job.id] = job
            self._active_id = job.id
            return job

    def _release(self, job: Job) -> None:
        with self._lock:
            if self._active_id == job.id:
                self._active_id = None

    def stop(self) -> str:
        """Stop the active job. Subprocess jobs die immediately; in-process
        inference tasks cannot be killed safely, so they are flagged and the
        result is discarded when they eventually finish."""
        job = self.active()
        if job is None:
            return "No task is running."
        with self._lock:
            job.stop_requested = True
        if job.kind in SUBPROCESS_KINDS:
            return request_stop()
        return (
            "Stop requested — the running task cannot be interrupted immediately; "
            "it will be marked as stopped when it finishes."
        )

    # ----------------------------------------------------------------- launch

    def start_callable(self, kind: str, title: str, fn, *args, pass_job: bool = False, **kwargs) -> Job:
        """Run ``fn(*args, **kwargs)`` in a background thread, capturing logs.

        With ``pass_job=True`` the :class:`Job` instance is passed as the first
        argument so the runner can stream its own log (used by the training
        endpoints that consume ``ui.utils.jobs.run_job`` generators).
        """
        job = self.register(kind, title)
        call_args = (job, *args) if pass_job else args
        worker = threading.Thread(
            target=self._thread_callable,
            args=(job, fn, call_args, kwargs),
            name=f"webui-job-{job.id}",
            daemon=True,
        )
        self._update(job, status="pending")
        worker.start()
        return job

    def start_commands(self, kind: str, title: str, commands: list, prefix: str = "") -> Job:
        """Run pipeline module commands as subprocesses (same as the Gradio
        Training tab) by consuming the ``ui.utils.jobs.run_job`` generator."""
        job = self.register(kind, title)
        worker = threading.Thread(
            target=self._thread_commands,
            args=(job, commands, prefix),
            name=f"webui-job-{job.id}",
            daemon=True,
        )
        self._update(job, status="pending")
        worker.start()
        return job

    # ----------------------------------------------------------------- threads

    def _finish(self, job: Job, stopped_marker: str | None = None) -> None:
        if stopped_marker and stopped_marker in job.log:
            self._update(job, status="stopped")
        elif job.stop_requested:
            self._update(job, status="stopped")
        else:
            self._update(job, status="finished")
        self._release(job)

    def _thread_callable(self, job: Job, fn, args, kwargs) -> None:
        root_logger = logging.getLogger()
        handler = _JobLogHandler(job, self)
        root_logger.addHandler(handler)
        self._update(job, status="running")
        try:
            result = fn(*args, **kwargs)
            payload = {"output": str(result)} if result is not None else None
            self._update(job, result=payload)
            self._finish(job)
        except gr.Error as error:  # validation errors raised by shared helpers
            message = getattr(error, "message", None) or str(error)
            self._update(job, status="error", error=str(message))
            self._release(job)
        except Exception as error:  # noqa: BLE001 — report everything to the UI
            self._update(job, status="error", error=f"{type(error).__name__}: {error}")
            self._release(job)
        finally:
            root_logger.removeHandler(handler)

    def _thread_commands(self, job: Job, commands: list, prefix: str) -> None:
        self._update(job, status="running")
        try:
            last_text = ""
            for text in run_job(job.title, commands, prefix):
                last_text = text
                self._replace_log(job, text)
            self._finish(job, stopped_marker="Stopped by user")
        except gr.Error as error:
            message = getattr(error, "message", None) or str(error)
            self._update(job, status="error", error=str(message))
            self._release(job)
        except Exception as error:  # noqa: BLE001 — report everything to the UI
            self._update(job, status="error", error=f"{type(error).__name__}: {error}")
            self._release(job)


manager = JobManager()
