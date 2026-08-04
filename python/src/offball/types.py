"""Core data structures shared across the pipeline.

Everything here is a frozen dataclass with no heavy dependencies, so the
tactics layer can be imported and tested without OpenCV, torch, or the Rust
extension present.

Two coordinate spaces are in play and are never mixed silently:

``image``
    Pixels, origin top-left, ``y`` increasing downward. Produced by detection
    and tracking.
``pitch``
    Metres, origin at the bottom-left corner, ``x`` along the length, ``y``
    across the width. Produced by applying the frame homography. All tactical
    metrics operate here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Iterable, Iterator, Sequence

__all__ = [
    "Team",
    "BBox",
    "Detection",
    "PlayerObservation",
    "FrameState",
    "Point",
]

Point = tuple[float, float]


class Team(str, Enum):
    """Which side an observation belongs to.

    ``UNKNOWN`` is a first-class value, not an error: team assignment is
    unreliable for the first few frames of a track and for players in a ruck,
    and forcing a guess corrupts downstream metrics far worse than abstaining.
    """

    HOME = "home"
    AWAY = "away"
    REFEREE = "referee"
    UNKNOWN = "unknown"

    @property
    def is_player(self) -> bool:
        return self in (Team.HOME, Team.AWAY)

    def opponent(self) -> "Team":
        if self is Team.HOME:
            return Team.AWAY
        if self is Team.AWAY:
            return Team.HOME
        return Team.UNKNOWN


@dataclass(frozen=True, slots=True)
class BBox:
    """Axis-aligned box in image pixels, ``(x1, y1)`` top-left."""

    x1: float
    y1: float
    x2: float
    y2: float

    def __post_init__(self) -> None:
        if self.x2 < self.x1 or self.y2 < self.y1:
            raise ValueError(f"degenerate bbox: {self}")

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def centre(self) -> Point:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    @property
    def ground_point(self) -> Point:
        """Bottom-centre of the box: where the player meets the turf.

        This, not the box centre, is what gets projected to pitch coordinates.
        Projecting the centre puts every player roughly a metre further from
        the camera than they really are, and the error grows with distance.
        """
        return ((self.x1 + self.x2) / 2.0, self.y2)

    def iou(self, other: "BBox") -> float:
        ix1, iy1 = max(self.x1, other.x1), max(self.y1, other.y1)
        ix2, iy2 = min(self.x2, other.x2), min(self.y2, other.y2)
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        inter = iw * ih
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0.0

    def scaled(self, fx: float, fy: float) -> "BBox":
        """Scale about the centre, used to crop the torso for team assignment."""
        cx, cy = self.centre
        hw, hh = self.width * fx / 2.0, self.height * fy / 2.0
        return BBox(cx - hw, cy - hh, cx + hw, cy + hh)


@dataclass(frozen=True, slots=True)
class Detection:
    """One raw detector output in a single frame."""

    bbox: BBox
    confidence: float
    label: str = "player"

    @property
    def is_ball(self) -> bool:
        return self.label == "ball"


@dataclass(frozen=True, slots=True)
class PlayerObservation:
    """A tracked player in one frame, in both coordinate spaces.

    ``pitch_xy`` and ``velocity`` are ``None`` until calibration and smoothing
    have run; the tactics layer skips observations that lack them rather than
    substituting zeros.
    """

    track_id: int
    bbox: BBox
    team: Team = Team.UNKNOWN
    confidence: float = 1.0
    pitch_xy: Point | None = None
    velocity: Point | None = None

    @property
    def speed(self) -> float | None:
        if self.velocity is None:
            return None
        vx, vy = self.velocity
        return math.hypot(vx, vy)

    def with_pitch(self, xy: Point, velocity: Point | None = None) -> "PlayerObservation":
        return replace(self, pitch_xy=xy, velocity=velocity)


@dataclass(frozen=True, slots=True)
class FrameState:
    """Everything known about one frame after the vision stage.

    ``attacking_team`` is the side in possession, which is what makes the
    off-ball metrics directional. When it is ``None`` the frame is scored for
    neither side (a loose ball, or a possession gap we could not resolve).
    """

    frame_index: int
    timestamp: float
    players: tuple[PlayerObservation, ...] = ()
    ball_pitch_xy: Point | None = None
    homography: tuple[float, ...] | None = None
    attacking_team: Team | None = None

    def team(self, team: Team) -> tuple[PlayerObservation, ...]:
        """Players of one side that have usable pitch coordinates."""
        return tuple(p for p in self.players if p.team is team and p.pitch_xy is not None)

    @property
    def is_calibrated(self) -> bool:
        return self.homography is not None

    @property
    def has_possession(self) -> bool:
        return self.attacking_team is not None and self.attacking_team.is_player


def positions(players: Iterable[PlayerObservation]) -> list[Point]:
    """Pitch positions of observations, skipping uncalibrated ones."""
    return [p.pitch_xy for p in players if p.pitch_xy is not None]


def velocities(players: Iterable[PlayerObservation]) -> list[Point]:
    """Pitch velocities, defaulting to stationary where unknown."""
    return [p.velocity or (0.0, 0.0) for p in players if p.pitch_xy is not None]
