"""Measure calibration quality on real footage.

The classical pitch-line detector has only ever been verified against
synthetically rendered pitches, where the answer is known and the images are
clean. This module is how it gets measured on real video, which is the number
that actually decides whether the platform works.

Two modes:

``survey``
    No ground truth needed. Runs the detector over a video or a directory of
    frames and reports how often it calibrates, with what line support, and how
    often the temporal gate rejects a solution. This is the first thing to run
    on any new source, and it is enough to tell a usable camera angle from an
    unusable one.

``evaluate``
    With ground-truth homographies, reports reprojection error in metres at
    points spread across the pitch — the figure to quote. Ground truth comes
    from a dataset such as SoccerNet-Calibration; see
    :func:`load_ground_truth`.

Both report **distributions, not means**. A calibrator that is excellent on 80%
of frames and catastrophic on 20% has a respectable mean and is unusable, and
only the tail shows that.
"""

from __future__ import annotations

import json
import math
import statistics
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .types import Point
from .vision.calibration import CalibrationConfig, calibrate_frame
from .vision.lines import ClassicalKeypointSource, PitchLineConfig

__all__ = [
    "CalibrationSurvey",
    "FrameOutcome",
    "evaluate_frames",
    "load_ground_truth",
    "survey_frames",
]

#: Points spread over the pitch at which reprojection error is measured.
#: Deliberately includes the corners, where perspective error is worst and a
#: centre-weighted sample would flatter the result.
PROBES: tuple[Point, ...] = tuple(
    (x, y) for x in (2.0, 26.0, 52.5, 79.0, 103.0) for y in (2.0, 34.0, 66.0)
)


@dataclass(frozen=True, slots=True)
class FrameOutcome:
    """What happened on one frame."""

    index: int
    calibrated: bool
    #: Fraction of the projected template landing on detected line pixels.
    support: float = 0.0
    #: Distinct lines found before template matching.
    lines: int = 0
    #: Mean inlier reprojection error of the fit, in metres.
    fit_error: float | None = None
    #: True when several candidate solutions tied — usually the pitch's own
    #: symmetry.
    ambiguous: bool = False
    #: Mean error against ground truth, in metres. Only in `evaluate` mode.
    truth_error: float | None = None


@dataclass
class CalibrationSurvey:
    """Aggregate result over a sequence of frames."""

    frames: int = 0
    calibrated: int = 0
    outcomes: list[FrameOutcome] = field(default_factory=list)

    @property
    def rate(self) -> float:
        """Share of frames that produced a usable homography."""
        return self.calibrated / self.frames if self.frames else 0.0

    def _values(self, attr: str) -> list[float]:
        return [
            getattr(o, attr)
            for o in self.outcomes
            if o.calibrated and getattr(o, attr) is not None
        ]

    def percentiles(self, attr: str) -> dict[str, float]:
        """Median / p90 / worst for a per-frame quantity.

        Reported instead of a mean because the tail is what makes a calibrator
        unusable, and a mean hides it.
        """
        values = sorted(self._values(attr))
        if not values:
            return {}
        return {
            "median": statistics.median(values),
            "p90": values[min(len(values) - 1, int(len(values) * 0.9))],
            "worst": values[-1],
        }

    @property
    def ambiguous_frames(self) -> int:
        return sum(1 for o in self.outcomes if o.ambiguous)

    def to_dict(self) -> dict:
        return {
            "frames": self.frames,
            "calibrated": self.calibrated,
            "rate": self.rate,
            "ambiguous_frames": self.ambiguous_frames,
            "support": self.percentiles("support"),
            "fit_error_m": self.percentiles("fit_error"),
            "truth_error_m": self.percentiles("truth_error"),
            "outcomes": [asdict(o) for o in self.outcomes],
        }

    def summary(self) -> str:
        """Human-readable report."""
        lines = [
            f"frames analysed   {self.frames}",
            f"calibrated        {self.calibrated} ({self.rate:.0%})",
            f"ambiguous         {self.ambiguous_frames}",
        ]
        support = self.percentiles("support")
        if support:
            lines.append(
                f"line support      median {support['median']:.2f} | "
                f"p90 {support['p90']:.2f}"
            )
        fit = self.percentiles("fit_error")
        if fit:
            lines.append(
                f"fit error (m)     median {fit['median']:.2f} | "
                f"p90 {fit['p90']:.2f} | worst {fit['worst']:.2f}"
            )
        truth = self.percentiles("truth_error")
        if truth:
            lines.append(
                f"TRUTH error (m)   median {truth['median']:.2f} | "
                f"p90 {truth['p90']:.2f} | worst {truth['worst']:.2f}"
            )
        else:
            lines.append("TRUTH error (m)   n/a - no ground truth supplied")
        return "\n".join(lines)


def _iter_images(source: str | Path, limit: int | None = None, stride: int = 1) -> Iterator:
    """Yield frames from a video file or a directory of images."""
    import cv2

    path = Path(source)
    if path.is_dir():
        files = sorted(
            p for p in path.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        for i, f in enumerate(files[:: max(1, stride)]):
            if limit is not None and i >= limit:
                return
            image = cv2.imread(str(f))
            if image is not None:
                yield image
        return

    from .video import read_frames

    yield from read_frames(path, stride=stride, limit=limit)


def survey_frames(
    frames: Iterable,
    config: PitchLineConfig | None = None,
    calibration: CalibrationConfig | None = None,
    ground_truth: list[tuple[float, ...] | None] | None = None,
) -> CalibrationSurvey:
    """Run the classical detector over frames and record what happened.

    Args:
        frames: Images, in order.
        config: Line-detection settings.
        calibration: Quality gates applied to each fit.
        ground_truth: Optional per-frame **image -> pitch** homographies. When
            supplied, per-frame error against truth is measured too.

    The detector keeps its own previous solution as a prior, exactly as it does
    in the pipeline, so this measures sequence behaviour rather than isolated
    frames.
    """
    source = ClassicalKeypointSource(config)
    calibration = calibration or CalibrationConfig()
    result = CalibrationSurvey()

    for i, frame in enumerate(frames):
        result.frames += 1
        pairs = source.keypoints(frame)
        support, lines = source.last_support, source.last_line_count
        ambiguous = source.last_ambiguous

        cal = calibrate_frame(pairs, calibration) if pairs else None
        if cal is None:
            result.outcomes.append(
                FrameOutcome(i, False, support, lines, None, ambiguous)
            )
            continue

        result.calibrated += 1
        truth_error = None
        if ground_truth is not None and i < len(ground_truth) and ground_truth[i]:
            truth_error = _error_against_truth(cal.matrix, ground_truth[i])

        result.outcomes.append(
            FrameOutcome(i, True, support, lines, cal.error, ambiguous, truth_error)
        )

    return result


def _error_against_truth(
    estimate: tuple[float, ...], truth: tuple[float, ...]
) -> float | None:
    """Mean distance, in metres, between two image->pitch homographies.

    Compares where each maps a spread of *image* points, then measures the gap
    in pitch metres — which is the error a user actually experiences, unlike a
    matrix norm.
    """
    from .kernels import project
    from .vision.lines import _invert

    truth_inv = _invert(truth)
    if truth_inv is None:
        return None
    # Take image points by pushing pitch probes through the true camera.
    image_points = project(truth_inv, list(PROBES))
    if any(p is None for p in image_points):
        return None

    got = project(estimate, [p for p in image_points if p is not None])
    errors = [
        math.dist(g, t) for g, t in zip(got, PROBES, strict=True) if g is not None
    ]
    return sum(errors) / len(errors) if errors else None


def load_ground_truth(path: str | Path) -> list[tuple[float, ...] | None]:
    """Load per-frame image->pitch homographies from a JSON file.

    Expected format — a list, one entry per frame, each either ``null`` or nine
    row-major numbers::

        [[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], null, ...]

    Datasets publish calibration in many shapes (SoccerNet annotates pitch
    lines; others give camera parameters). Converting a given dataset into this
    format is a small adapter, kept out of here so this module does not grow a
    dependency on any one dataset's schema.
    """
    data = json.loads(Path(path).read_text())
    out: list[tuple[float, ...] | None] = []
    for entry in data:
        if entry is None:
            out.append(None)
        elif len(entry) == 9:
            out.append(tuple(float(v) for v in entry))
        else:
            raise ValueError(f"homography must have 9 elements, got {len(entry)}")
    return out


def evaluate_frames(
    source: str | Path,
    ground_truth: str | Path | None = None,
    limit: int | None = None,
    stride: int = 1,
    config: PitchLineConfig | None = None,
) -> CalibrationSurvey:
    """Survey a video file or image directory, optionally against ground truth."""
    truth = load_ground_truth(ground_truth) if ground_truth else None
    return survey_frames(
        _iter_images(source, limit=limit, stride=stride),
        config=config,
        ground_truth=truth,
    )
