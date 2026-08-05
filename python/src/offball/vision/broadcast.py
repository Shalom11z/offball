"""Calibration for the broadcast centre view.

.. warning::

   **Measured against SoccerNet ground truth, this is not accurate.** On
   centre-view frames it produces a homography for 83% of images with a
   **median error of 51 metres** — roughly half a pitch. It should not be
   relied on for measurement.

   The overlays look convincing, and that is the trap: an overlay shows the
   projected template sitting on the paint, which a badly wrong homography can
   also do. Only ground truth revealed this, and it did so immediately.

   Three hypotheses for the cause were tested and **all rejected**: a penalty
   arc mistaken for the centre circle, the pitch's mirror symmetry, and
   unverified fits passing without support. None changed the median. The cause
   is not yet known.

   Kept because the machinery (evidence extraction, camera solve, symmetry
   handling) is sound and the failure is worth diagnosing rather than deleting.
   See ``docs/02-vision-pipeline.md``.


The straight-line detector calibrates 0% of a real Premier League half
(``docs/02-vision-pipeline.md``): the dominant shot holds one straight pitch
line and a circle, and template matching needs two lines per direction.

This module solves the same frame from the evidence that *is* present, and
raises the rate to roughly half the frames on that footage.

Three sources, each contributing something the others cannot:

**Far touchline, from the mask boundary.** Where grass meets the advertising
hoardings. Nothing is painted there, so Hough can never see it, but it is one of
the strongest and most reliably present cues in broadcast framing.

**Halfway line, from paint.** The single straight line the centre view reliably
shows.

**Centre circle, as an ellipse.** A circle maps to an ellipse under a
homography, and its 9.15m radius is a powerful constraint. A straight-line
transform actively destroys this by shattering the arc into chords — the exact
failure measured on real footage.

These are fed to :func:`~offball.vision.camera.fit_camera`, which fits a
physical camera rather than a free homography. See that module for why the
reduced degrees of freedom are what make the shot solvable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from ..types import Point
from .camera import CameraFit, fit_camera
from .lines import PitchLineConfig, detect_lines, line_mask, pitch_mask
from .lines import _invert as _invert_h

__all__ = ["BroadcastCalibrator", "BroadcastConfig", "PitchEvidence", "extract_evidence"]


@dataclass(frozen=True, slots=True)
class BroadcastConfig:
    lines: PitchLineConfig = field(default_factory=PitchLineConfig)
    #: Sample every Nth image column when tracing the pitch boundary.
    boundary_step: int = 6
    #: A column needs this many pitch pixels before its topmost one is trusted
    #: as touchline; fewer means the column clips scoreboard or crowd.
    min_column_pitch_pixels: int = 40
    #: Boundary points further than this from a robust line fit are dropped.
    boundary_inlier_px: float = 6.0
    #: A line steeper than this from horizontal is a halfway-line candidate.
    halfway_max_angle_deg: float = 35.0
    #: Minimum major axis of a candidate centre-circle ellipse, in pixels.
    min_ellipse_major: float = 120.0
    min_ellipse_minor: float = 20.0
    #: Reject near-degenerate ellipses, which are usually a straight edge.
    min_axis_ratio: float = 0.10
    #: A contour point this close to the fitted ellipse counts as an inlier.
    ellipse_inlier_px: float = 3.0
    #: Fraction of contour points that must be inliers for the contour to be
    #: accepted as a genuine ellipse. This single check is what separates a
    #: real centre circle from an arbitrary blob, and raised the calibration
    #: rate on real footage from 10% to ~50%.
    min_ellipse_inlier_ratio: float = 0.65
    min_ellipse_points: int = 70
    #: Reject a camera fit above this RMS residual, in metres.
    max_error: float = 3.0
    #: Fraction of the projected pitch template that must land on detected line
    #: pixels for a fit to be accepted.
    #:
    #: **This is the guard against being confidently wrong.** A low residual
    #: only says the camera explains the evidence it was given; it says nothing
    #: about whether that evidence was what we assumed. On a penalty-area view
    #: the penalty arc is a perfectly good ellipse, gets taken for the centre
    #: circle, and yields a self-consistent camera that puts the pitch ~40m from
    #: where it is. Measured against SoccerNet ground truth, that failure
    #: accounted for a 51m median error across 34% of frames. Checking the whole
    #: projected template against the actual paint catches it.
    min_support: float = 0.45
    #: Pixels within which a projected template point counts as supported.
    support_tolerance: float = 12.0
    pitch_length: float = 105.0
    pitch_width: float = 68.0


@dataclass(frozen=True, slots=True)
class PitchEvidence:
    """Image points gathered from one frame, by source."""

    circle: list[Point]
    touchline: list[Point]
    halfway: list[Point]

    @property
    def sources(self) -> int:
        """How many of the three sources supplied usable evidence."""
        return sum(1 for s in (self.circle, self.touchline, self.halfway) if len(s) >= 2)


def _mirror(
    homography: tuple[float, ...],
    pitch_length: float,
    pitch_width: float,
    flip_x: bool,
    flip_y: bool,
) -> tuple[float, ...] | None:
    """Compose an image->pitch homography with a pitch symmetry.

    The markings are invariant under ``x -> length - x`` and
    ``y -> width - y``, so these four variants all explain a symmetric view
    equally well and only asymmetric evidence distinguishes them.
    """
    sx, tx = (-1.0, pitch_length) if flip_x else (1.0, 0.0)
    sy, ty = (-1.0, pitch_width) if flip_y else (1.0, 0.0)
    h = homography
    # M @ H, with M the affine mirror in pitch space.
    return (
        sx * h[0] + tx * h[6], sx * h[1] + tx * h[7], sx * h[2] + tx * h[8],
        sy * h[3] + ty * h[6], sy * h[4] + ty * h[7], sy * h[5] + ty * h[8],
        h[6], h[7], h[8],
    )


def _ellipse_inliers(points: list[Point], ellipse, tolerance: float) -> list[Point]:
    """Contour points lying on the fitted ellipse.

    Distance is measured in the ellipse's own frame and scaled by the minor
    axis, which approximates true geometric distance closely enough to
    discriminate an ellipse from a blob without an iterative foot-point solve.
    """
    (ex, ey), (major, minor), angle = ellipse
    a, b = major / 2.0, minor / 2.0
    if a < 1.0 or b < 1.0:
        return []
    t = math.radians(angle)
    cos_t, sin_t = math.cos(t), math.sin(t)

    keep: list[Point] = []
    for x, y in points:
        dx, dy = x - ex, y - ey
        u = dx * cos_t + dy * sin_t
        v = -dx * sin_t + dy * cos_t
        r = math.hypot(u / a, v / b)
        if abs(r - 1.0) * min(a, b) <= tolerance:
            keep.append((x, y))
    return keep


def extract_evidence(frame, config: BroadcastConfig | None = None) -> PitchEvidence:
    """Gather touchline, halfway-line and centre-circle points from a frame."""
    import cv2
    import numpy as np

    config = config or BroadcastConfig()
    height, width = frame.shape[:2]
    mask = pitch_mask(frame, config.lines)
    paint = line_mask(frame, mask, config.lines)
    lines = detect_lines(paint, config.lines)

    # --- far touchline: the top edge of the grass ------------------------
    touchline: list[Point] = []
    for x in range(0, width, config.boundary_step):
        column = np.nonzero(mask[:, x])[0]
        if column.size < config.min_column_pitch_pixels:
            continue
        y = int(column.min())
        # Skip columns clipped by the frame edge: their topmost pitch pixel is
        # the edge of the image, not the touchline.
        if 3 < y < height - 4:
            touchline.append((float(x), float(y)))

    if len(touchline) >= 10:
        pts = np.array(touchline, np.float32)
        vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
        distance = np.abs((pts[:, 0] - x0) * vy - (pts[:, 1] - y0) * vx)
        inlier = distance < config.boundary_inlier_px
        touchline = [p for p, keep in zip(touchline, inlier, strict=True) if keep]
    else:
        touchline = []

    # --- halfway line: the strongest steep line --------------------------
    halfway: list[Point] = []
    candidates = [
        line
        for line in lines
        if min(abs(math.degrees(line.theta)), 180 - abs(math.degrees(line.theta)))
        < config.halfway_max_angle_deg
    ]
    if candidates:
        best = max(candidates, key=lambda line: line.strength)
        cos_t, sin_t = math.cos(best.theta), math.sin(best.theta)
        for t in np.linspace(-height, height, 60):
            x = cos_t * best.rho - sin_t * t
            y = sin_t * best.rho + cos_t * t
            if 0 <= x < width and 0 <= y < height and mask[int(y), int(x)] > 0:
                halfway.append((float(x), float(y)))

    # --- centre circle ----------------------------------------------------
    # Erase every straight line first: the circle touches the halfway line, and
    # without this their contours merge into one unfittable blob.
    without_lines = paint.copy()
    for line in lines:
        cos_t, sin_t = math.cos(line.theta), math.sin(line.theta)
        x0, y0 = cos_t * line.rho, sin_t * line.rho
        cv2.line(
            without_lines,
            (int(x0 + 3000 * -sin_t), int(y0 + 3000 * cos_t)),
            (int(x0 - 3000 * -sin_t), int(y0 - 3000 * cos_t)),
            0,
            9,
        )

    contours, _ = cv2.findContours(
        without_lines, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE
    )
    circle: list[Point] = []
    for contour in contours:
        if len(contour) < 60:
            continue
        try:
            ellipse = cv2.fitEllipse(contour)
        except cv2.error:
            continue
        (ex, ey), (major, minor), _ = ellipse
        if major < config.min_ellipse_major or minor < config.min_ellipse_minor:
            continue
        if minor / major < config.min_axis_ratio:
            continue
        if not (0 < ex < width and 0 < ey < height):
            continue

        points = [(float(p[0][0]), float(p[0][1])) for p in contour]
        inliers = _ellipse_inliers(points, ellipse, config.ellipse_inlier_px)
        if (
            len(inliers) < config.min_ellipse_inlier_ratio * len(points)
            or len(inliers) < config.min_ellipse_points
        ):
            continue
        if len(inliers) > len(circle):
            circle = inliers

    return PitchEvidence(circle=circle[::2], touchline=touchline, halfway=halfway)


class BroadcastCalibrator:
    """A :class:`~offball.vision.calibration.KeypointSource` for broadcast video.

    Emits correspondences sampled from the fitted camera rather than raw
    observations, because the underlying evidence cannot be paired point-by-point
    with pitch coordinates: a circle point is known only to lie 9.15m from the
    centre spot, and a touchline point only to lie on ``y = width``.

    A consequence worth knowing: the reprojection error
    :func:`~offball.vision.calibration.calibrate_frame` computes from these
    correspondences is **not** an independent check, since they were generated
    from the very homography being fitted. The real quality signal is
    :attr:`last_error`, the RMS residual of the camera fit against the observed
    evidence, in metres.
    """

    def __init__(self, config: BroadcastConfig | None = None) -> None:
        self.config = config or BroadcastConfig()
        #: RMS residual of the last camera fit, in metres.
        self.last_error: float = float("inf")
        #: Evidence counts from the last frame, for tuning.
        self.last_evidence: PitchEvidence | None = None
        #: Fraction of the projected template that landed on paint.
        self.last_support: float = 0.0
        self.last_fit: CameraFit | None = None

    def _support(self, homography, paint, shape) -> float:
        """Fraction of the projected pitch template landing on real paint.

        Independent of the fit: the camera was solved from three specific cues,
        and this asks whether the *rest* of the pitch then lands where paint
        actually is.
        """
        import cv2

        from ..kernels import project
        from ..viz import _pitch_polylines

        config = self.config
        distance = cv2.distanceTransform(cv2.bitwise_not(paint), cv2.DIST_L2, 3)
        height, width = shape[:2]

        samples: list[Point] = []
        for polyline in _pitch_polylines(config.pitch_length, config.pitch_width):
            samples.extend(polyline)

        pitch_to_image = _invert_h(homography)
        if pitch_to_image is None:
            return 0.0
        projected = project(pitch_to_image, samples)

        hits = visible = 0
        for point in projected:
            if point is None:
                continue
            x, y = point
            if not (0 <= x < width and 0 <= y < height):
                continue
            visible += 1
            if distance[int(y), int(x)] <= config.support_tolerance:
                hits += 1
        if visible < 20:
            return 0.0
        return hits / visible

    def calibrate(self, frame) -> CameraFit | None:
        """Fit a camera to one frame, or return ``None``."""
        config = self.config
        evidence = extract_evidence(frame, config)
        self.last_evidence = evidence
        self.last_error = float("inf")
        self.last_fit = None

        if evidence.sources < 2:
            # One source leaves the camera underdetermined; the optimiser would
            # return a confident wrong answer.
            return None

        height, width = frame.shape[:2]
        fit = fit_camera(
            width,
            height,
            circle_points=evidence.circle,
            touchline_points=evidence.touchline,
            halfway_points=evidence.halfway,
            pitch_length=config.pitch_length,
            pitch_width=config.pitch_width,
            max_error=config.max_error,
        )
        if fit is None:
            return None

        # Verify against the paint before trusting it. Without this the fit is
        # only self-consistent, which is how a penalty arc mistaken for the
        # centre circle produces a confident 40m error.
        paint = line_mask(frame, pitch_mask(frame, config.lines), config.lines)

        # Resolve the pitch's own symmetry. The evidence this calibrator uses —
        # centre circle, halfway line, a touchline — is invariant under
        # x -> length - x and y -> width - y, so the fit is equally good at
        # either end. Measured against SoccerNet ground truth, taking the raw
        # fit gave a 51m median error: almost exactly half a pitch, the
        # signature of a mirrored solution.
        #
        # An overlay cannot reveal this, because a mirrored homography puts the
        # template on the paint just as neatly. Only asymmetric evidence breaks
        # the tie, so each variant is scored against the *whole* template,
        # penalty boxes included.
        best, best_support = fit, self._support(fit.homography, paint, frame.shape)
        for flip_x, flip_y in ((True, False), (False, True), (True, True)):
            variant = _mirror(
                fit.homography, config.pitch_length, config.pitch_width, flip_x, flip_y
            )
            if variant is None:
                continue
            score = self._support(variant, paint, frame.shape)
            if score > best_support:
                best_support = score
                best = CameraFit(fit.camera, variant, fit.error, fit.observations)

        fit = best
        support = best_support
        self.last_support = support
        if support < config.min_support:
            return None

        self.last_error = fit.error
        self.last_fit = fit
        return fit

    def keypoints(self, frame) -> list[tuple[Point, Point]]:
        """Image/pitch correspondences implied by the fitted camera."""
        import numpy as np  # noqa: F401  (kept for symmetry with other sources)

        from ..kernels import project

        fit = self.calibrate(frame)
        if fit is None:
            return []

        height, width = frame.shape[:2]
        pitch_to_image = fit.camera.homography()
        if pitch_to_image is None:
            return []

        config = self.config
        samples = [
            (config.pitch_length * i / 6.0, config.pitch_width * j / 4.0)
            for i in range(7)
            for j in range(5)
        ]
        projected = project(pitch_to_image, samples)
        return [
            ((p[0], p[1]), pitch)
            for p, pitch in zip(projected, samples, strict=True)
            if p is not None and 0 <= p[0] < width and 0 <= p[1] < height
        ]
