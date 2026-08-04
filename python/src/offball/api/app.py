"""FastAPI application.

Analysis is asynchronous by necessity: a full match takes minutes to hours
depending on stride and hardware, so the API accepts a job, returns immediately,
and the caller polls. The endpoints are:

===============================  =========================================
``POST   /v1/analyses``          Submit footage, get a job id back.
``GET    /v1/analyses``          List recent jobs.
``GET    /v1/analyses/{id}``     Job status.
``GET    /v1/analyses/{id}/report``  The finished report.
``GET    /healthz``              Liveness, plus which kernel backend is active.
===============================  =========================================

The worker here runs in a FastAPI background task, which is right for a single
node and wrong for anything larger — see ``docs/01-architecture.md`` for the
queue-backed version.
"""

from __future__ import annotations

import logging
from typing import Annotated

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Query, status

from .. import __version__
from ..kernels import BACKEND
from .schemas import (
    AnalysisRequest,
    ErrorResponse,
    JobResponse,
    JobStatus,
    ReportResponse,
)
from .store import InMemoryJobStore, Job, JobStore

logger = logging.getLogger(__name__)

_store = InMemoryJobStore()


def get_store() -> JobStore:
    """Dependency seam: override in tests, swap for Postgres in production."""
    return _store


app = FastAPI(
    title="offball",
    version=__version__,
    summary="Automated off-the-ball positioning analysis from match footage.",
)


def _to_job_response(job: Job) -> JobResponse:
    return JobResponse(
        job_id=job.job_id,
        status=job.status,
        match_id=job.match_id,
        created_at=job.created_at,
        updated_at=job.updated_at,
        progress=job.progress,
        error=job.error,
    )


def run_analysis(job_id: str, request: AnalysisRequest, store: JobStore) -> None:
    """Execute one analysis job.

    Wired up but not yet connected to a real detector: constructing the vision
    stack needs model weights, which are a deployment concern rather than a
    library one. The job therefore fails loudly with a clear message instead of
    returning a fabricated report — see ``docs/06-roadmap.md`` for the
    remaining work.
    """
    store.update(job_id, status=JobStatus.RUNNING, progress=0.0)
    try:
        from ..pipeline import Pipeline, PipelineConfig  # noqa: F401
        from ..video import read_frames  # noqa: F401

        raise NotImplementedError(
            "No detector is configured. Set OFFBALL_DETECTOR_WEIGHTS and install "
            "the 'vision' extra to enable video analysis."
        )
    except Exception as exc:
        logger.exception("analysis job %s failed", job_id)
        store.update(job_id, status=JobStatus.FAILED, error=str(exc))


@app.get("/healthz", tags=["ops"])
def healthz() -> dict:
    """Liveness probe. Reports which numeric backend is compiled in."""
    return {"status": "ok", "version": __version__, "kernel_backend": BACKEND}


@app.post(
    "/v1/analyses",
    response_model=JobResponse,
    status_code=status.HTTP_202_ACCEPTED,
    tags=["analyses"],
)
def create_analysis(
    request: AnalysisRequest,
    background: BackgroundTasks,
    store: Annotated[JobStore, Depends(get_store)],
) -> JobResponse:
    """Queue a match for analysis.

    Returns 202 with a job id; poll ``GET /v1/analyses/{id}`` for status.
    """
    job = store.create(video_uri=request.video_uri, match_id=request.match_id)
    background.add_task(run_analysis, job.job_id, request, store)
    return _to_job_response(job)


@app.get("/v1/analyses", response_model=list[JobResponse], tags=["analyses"])
def list_analyses(
    store: Annotated[JobStore, Depends(get_store)],
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
) -> list[JobResponse]:
    """Most recent jobs, newest first."""
    return [_to_job_response(j) for j in store.list(limit)]


@app.get(
    "/v1/analyses/{job_id}",
    response_model=JobResponse,
    responses={404: {"model": ErrorResponse}},
    tags=["analyses"],
)
def get_analysis(
    job_id: str, store: Annotated[JobStore, Depends(get_store)]
) -> JobResponse:
    job = store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"no such job: {job_id}")
    return _to_job_response(job)


@app.get(
    "/v1/analyses/{job_id}/report",
    response_model=ReportResponse,
    responses={404: {"model": ErrorResponse}, 409: {"model": ErrorResponse}},
    tags=["analyses"],
)
def get_report(
    job_id: str, store: Annotated[JobStore, Depends(get_store)]
) -> ReportResponse:
    """The finished report.

    Returns 409 while the job is still running, so callers can distinguish
    "not ready yet" from "no such job".
    """
    job = store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"no such job: {job_id}")
    if job.status is not JobStatus.SUCCEEDED or job.report is None:
        raise HTTPException(
            status_code=409,
            detail=f"job {job_id} is {job.status.value}; no report available",
        )
    return ReportResponse(job_id=job.job_id, match_id=job.match_id, **job.report)
