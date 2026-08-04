"""Job storage.

:class:`JobStore` is the interface the API depends on. :class:`InMemoryJobStore`
implements it for local development and tests.

The production implementation is Postgres-backed against the schema in
``sql/``; it is not written yet, and the interface is kept deliberately narrow
so that it can be. Nothing in the API layer assumes storage is in-process —
that is what lets the worker be a separate service later.
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Protocol, runtime_checkable

from .schemas import JobStatus

__all__ = ["Job", "JobStore", "InMemoryJobStore"]


def _now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True, slots=True)
class Job:
    job_id: str
    status: JobStatus
    created_at: datetime
    updated_at: datetime
    video_uri: str
    match_id: str | None = None
    progress: float = 0.0
    error: str | None = None
    #: The report as a plain dict, present once the job has succeeded.
    report: dict | None = None


@runtime_checkable
class JobStore(Protocol):
    def create(self, video_uri: str, match_id: str | None) -> Job: ...

    def get(self, job_id: str) -> Job | None: ...

    def update(self, job_id: str, **changes) -> Job | None: ...

    def list(self, limit: int = 50) -> list[Job]: ...


class InMemoryJobStore:
    """Thread-safe in-process job store.

    Suitable for a single-process deployment and for tests. State is lost on
    restart, which is why it is not the production path.
    """

    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def create(self, video_uri: str, match_id: str | None = None) -> Job:
        now = _now()
        job = Job(
            job_id=uuid.uuid4().hex,
            status=JobStatus.QUEUED,
            created_at=now,
            updated_at=now,
            video_uri=video_uri,
            match_id=match_id,
        )
        with self._lock:
            self._jobs[job.job_id] = job
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def update(self, job_id: str, **changes) -> Job | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            updated = replace(job, updated_at=_now(), **changes)
            self._jobs[job_id] = updated
            return updated

    def list(self, limit: int = 50) -> list[Job]:
        with self._lock:
            jobs = sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)
        return jobs[:limit]
