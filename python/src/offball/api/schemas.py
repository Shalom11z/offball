"""Pydantic schemas for the HTTP API.

These are the wire contract. They are kept separate from the internal
dataclasses in :mod:`offball.types` on purpose: the internal shapes change as
the model evolves, and the API should not break every time they do.

The TypeScript client in ``ts/sdk`` mirrors these types; ``npm run codegen``
regenerates it from the OpenAPI schema this module produces.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, field_validator

__all__ = [
    "JobStatus",
    "AnalysisRequest",
    "JobResponse",
    "PlayerSummarySchema",
    "TeamSummarySchema",
    "ReportResponse",
    "ErrorResponse",
]


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class AnalysisRequest(BaseModel):
    """Request to analyse one match video."""

    video_uri: str = Field(
        ..., description="Location of the source footage (s3://, gs://, or a local path)."
    )
    match_id: str | None = Field(None, description="Caller's own identifier for the fixture.")
    fps: float = Field(25.0, gt=0, le=120, description="Source frame rate.")
    stride: int = Field(
        1,
        ge=1,
        le=25,
        description=(
            "Analyse every Nth frame. Raising this trades tracking robustness for "
            "throughput; see docs/02-vision-pipeline.md."
        ),
    )
    pitch_length: float = Field(105.0, ge=90.0, le=120.0, description="Metres.")
    pitch_width: float = Field(68.0, ge=45.0, le=90.0, description="Metres.")

    @field_validator("video_uri")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("video_uri must not be empty")
        return v


class JobResponse(BaseModel):
    job_id: str
    status: JobStatus
    match_id: str | None = None
    created_at: datetime
    updated_at: datetime
    #: 0.0-1.0, best-effort.
    progress: float = 0.0
    error: str | None = None


class PlayerSummarySchema(BaseModel):
    track_id: int
    frames: int
    duration: float
    median_space_owned: float = Field(..., description="Pitch area owned, m^2.")
    median_position_value: float = Field(..., description="Threat value of ground held, 0-1.")
    availability_rate: float = Field(..., description="Share of frames offering a clear pass.")
    offside_rate: float
    median_offside_margin: float | None = Field(
        None, description="Metres beyond the offside line; negative is onside."
    )
    mean_lines_broken: float
    median_separation: float | None = Field(
        None, description="Median nearest-opponent distance, metres."
    )
    mean_pressure: float


class TeamSummarySchema(BaseModel):
    team: Literal["home", "away", "referee", "unknown"]
    frames: int
    duration: float
    median_controlled_space: float
    median_dangerous_space: float
    median_attacking_hull: float
    median_defending_hull: float
    mean_passing_options: float


class ReportResponse(BaseModel):
    job_id: str
    match_id: str | None = None
    frames_scored: int
    frames_total: int
    coverage: float = Field(
        ...,
        description=(
            "Share of frames that could be scored. Below ~0.6 the vision stage "
            "struggled and the figures are provisional."
        ),
    )
    teams: list[TeamSummarySchema]
    players: list[PlayerSummarySchema]


class ErrorResponse(BaseModel):
    detail: str
