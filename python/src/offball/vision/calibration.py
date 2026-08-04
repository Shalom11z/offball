"""Camera calibration: image pixels to pitch metres.

Broadcast footage is a moving view of a flat plane, so one 3x3 homography per
frame is enough — no full camera model, no intrinsics, no lens calibration.

The hard part is not the algebra (see :mod:`offball.kernels`) but robustness:

* A pitch has rotational and reflective symmetry. A keypoint model will
  confidently label the *wrong* penalty box perhaps a few percent of the time,
  and one such frame throws every player 50 metres across the pitch. RANSAC
  catches the single-keypoint version of this; :class:`HomographySmoother`
  catches the whole-frame version by rejecting solutions that disagree with
  recent history.
* Close-ups, replays and crowd shots have no usable pitch lines at all. Those
  frames must be reported as uncalibrated, never guessed at.

:class:`KeypointSource` is the seam where a trained pitch-keypoint model plugs
in. :class:`ScriptedKeypoints` provides deterministic correspondences for tests.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from ..kernels import fit_homography, project
from ..types import Point

__all__ = [
    "Calibration",
    "CalibrationConfig",
    "HomographySmoother",
    "KeypointSource",
    "ScriptedKeypoints",
]


@runtime_checkable
class KeypointSource(Protocol):
    """Finds named pitch landmarks in a frame.

    Returns ``(image_point, pitch_point)`` correspondences. The pitch side comes
    from ``Pitch::template_keypoints`` in the Rust crate (mirrored in
    :mod:`offball.pitch`), so implementations only need to locate landmarks and
    name them.
    """

    def keypoints(self, frame) -> list[tuple[Point, Point]]:
        ...


class ScriptedKeypoints:
    """Replays fixed correspondences, one entry per frame.

    ``None`` entries represent frames with no usable pitch view (a close-up or
    a crowd shot), which the pipeline must handle rather than crash on.
    """

    def __init__(self, script: Sequence[list[tuple[Point, Point]] | None]) -> None:
        self._script = list(script)
        self._index = 0

    def reset(self) -> None:
        self._index = 0

    def keypoints(self, frame=None) -> list[tuple[Point, Point]]:
        if self._index >= len(self._script):
            return []
        out = self._script[self._index]
        self._index += 1
        return list(out) if out else []


@dataclass(frozen=True, slots=True)
class Calibration:
    """A fitted homography for one frame, with its quality measures."""

    matrix: tuple[float, ...]
    #: Mean reprojection error over the inliers, in metres.
    error: float
    inliers: int
    total_keypoints: int

    @property
    def inlier_ratio(self) -> float:
        return self.inliers / self.total_keypoints if self.total_keypoints else 0.0

    def to_pitch(self, points: Sequence[Point]) -> list[Point | None]:
        """Project image points onto the pitch."""
        return project(self.matrix, points)


@dataclass(frozen=True, slots=True)
class CalibrationConfig:
    #: RANSAC inlier threshold, in metres on the pitch.
    ransac_threshold: float = 1.5
    ransac_iterations: int = 1000
    seed: int = 0
    #: Reject a fit whose mean inlier error exceeds this, in metres.
    max_error: float = 3.0
    #: Reject a fit supported by fewer than this many inliers.
    min_inliers: int = 5
    #: How far the projected pitch centre may jump between consecutive
    #: calibrated frames, in metres. A real camera pans smoothly; a 30m jump in
    #: 40ms is a mis-solve, not a camera move.
    max_centre_jump: float = 15.0
    #: Frames of history kept for the continuity check.
    history: int = 5


def calibrate_frame(
    correspondences: Sequence[tuple[Point, Point]], config: CalibrationConfig | None = None
) -> Calibration | None:
    """Fit one frame's homography from image/pitch correspondences.

    Returns ``None`` when there are too few keypoints, the fit fails, or the
    result does not meet the quality bars in ``config``.
    """
    config = config or CalibrationConfig()
    if len(correspondences) < 4:
        return None

    src = [c[0] for c in correspondences]
    dst = [c[1] for c in correspondences]
    try:
        matrix, inliers = fit_homography(
            src, dst, config.ransac_threshold, config.ransac_iterations, config.seed
        )
    except ValueError:
        return None

    if len(inliers) < config.min_inliers:
        return None

    projected = project(matrix, [src[i] for i in inliers])
    errors = []
    for p, i in zip(projected, inliers, strict=True):
        if p is None:
            return None
        errors.append(math.hypot(p[0] - dst[i][0], p[1] - dst[i][1]))
    error = sum(errors) / len(errors)
    if error > config.max_error:
        return None

    return Calibration(
        matrix=matrix, error=error, inliers=len(inliers), total_keypoints=len(correspondences)
    )


class HomographySmoother:
    """Temporal gate over per-frame calibrations.

    Rejects solutions that are geometrically fine in isolation but inconsistent
    with the recent camera path — the symmetry mis-solve described in the module
    docstring. On rejection (or on an uncalibrated frame) it can coast on the
    last good homography for a short while, which carries the pipeline through
    brief losses of pitch markings.
    """

    def __init__(self, config: CalibrationConfig | None = None, max_coast: int = 10) -> None:
        self.config = config or CalibrationConfig()
        self._history: deque[Calibration] = deque(maxlen=self.config.history)
        self._last_good: Calibration | None = None
        self._coasted = 0
        self.max_coast = max_coast
        self.rejected = 0

    @staticmethod
    def _reference_point(cal: Calibration) -> Point | None:
        """Where the image centre-ish point lands on the pitch.

        Any fixed image point works as a continuity probe; what matters is that
        the same point is used every frame.
        """
        out = project(cal.matrix, [(960.0, 540.0)])
        return out[0]

    def push(self, cal: Calibration | None) -> Calibration | None:
        """Feed one frame's calibration; returns the one to actually use."""
        if cal is not None and self._last_good is not None:
            prev = self._reference_point(self._last_good)
            cur = self._reference_point(cal)
            if prev is not None and cur is not None:
                jump = math.hypot(cur[0] - prev[0], cur[1] - prev[1])
                if jump > self.config.max_centre_jump:
                    self.rejected += 1
                    cal = None

        if cal is not None:
            self._history.append(cal)
            self._last_good = cal
            self._coasted = 0
            return cal

        # Coast on the last good solution for a bounded number of frames.
        if self._last_good is not None and self._coasted < self.max_coast:
            self._coasted += 1
            return self._last_good
        return None

    def reset(self) -> None:
        self._history.clear()
        self._last_good = None
        self._coasted = 0
        self.rejected = 0
