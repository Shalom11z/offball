"""Camera-model calibration: solve the shot that line matching cannot.

The straight-line detector fails on the dominant broadcast view, and the reason
is counted in ``docs/02-vision-pipeline.md``: a centre shot contains exactly one
straight pitch line — the halfway line — plus the centre circle. Matching lines
to a template needs two lines in each pitch direction, and that information is
not in the picture.

This module solves the same frame from what *is* there:

* the **halfway line**, from paint;
* the **far touchline**, taken from the boundary of the pitch mask rather than
  from paint — it is where grass meets the advertising hoardings, which Hough
  cannot see because nothing is painted there;
* the **centre circle**, whose 9.15m radius is a strong constraint that a
  straight-line transform actively destroys by fragmenting the arc into chords.

## Why a camera and not a homography

A homography has 8 degrees of freedom. The evidence above supplies roughly 9
constraints, but they are badly conditioned — the three sources are nearly
degenerate with respect to a free 8-DOF fit, and it wanders off into
projectively valid nonsense.

A real broadcast camera has far fewer: position, orientation and focal length,
with square pixels and a principal point at the image centre. That is 7, and
crucially they are *physically* constrained — a camera cannot be underground or
have negative focal length. Fitting the camera and deriving the homography from
it is what makes the centre view tractable.

## Residuals are in metres

Every residual is expressed by un-projecting detected image points onto the
pitch and comparing against known geometry: circle points should sit 9.15m from
the centre spot, touchline points on ``y = width``, halfway points on
``x = length / 2``. Errors are therefore in metres and directly comparable, so
no arbitrary weighting between terms is needed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..types import Point

__all__ = ["CameraFit", "CameraModel", "broadcast_prior", "fit_camera"]


@dataclass(frozen=True, slots=True)
class CameraModel:
    """A pinhole camera viewing the pitch plane.

    Angles in radians. Position in pitch metres, with ``z`` up. The pitch plane
    is ``z = 0``, so the camera sits at positive ``z`` and negative ``y`` for a
    main-gantry view along one touchline.
    """

    #: Focal length in pixels.
    focal: float
    #: Camera centre in pitch coordinates.
    x: float
    y: float
    z: float
    #: Rotation: pan about the world z axis, tilt down from horizontal, roll
    #: about the optical axis.
    pan: float
    tilt: float
    roll: float
    #: Principal point, normally the image centre.
    cx: float = 640.0
    cy: float = 360.0

    def rotation(self) -> list[list[float]]:
        """World -> camera rotation.

        Composed as roll * tilt * pan. The camera looks along +y in world terms
        at zero pan, which matches a gantry on the near touchline.
        """
        cp, sp = math.cos(self.pan), math.sin(self.pan)
        ct, st = math.cos(self.tilt), math.sin(self.tilt)
        cr, sr = math.cos(self.roll), math.sin(self.roll)

        # Pan about world z.
        rz = [[cp, -sp, 0.0], [sp, cp, 0.0], [0.0, 0.0, 1.0]]
        # Camera axes: x right, y down, z forward. Start looking along +y.
        base = [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]
        # Tilt about the camera x axis.
        rt = [[1.0, 0.0, 0.0], [0.0, ct, -st], [0.0, st, ct]]
        # Roll about the optical axis.
        rr = [[cr, -sr, 0.0], [sr, cr, 0.0], [0.0, 0.0, 1.0]]

        def mul(a, b):
            return [
                [sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)]
                for i in range(3)
            ]

        return mul(rr, mul(rt, mul(base, rz)))

    def homography(self) -> tuple[float, ...] | None:
        """Pitch -> image homography, row-major.

        For a plane at ``z = 0`` the projection reduces to
        ``K [r1 r2 t]`` where ``t = -R C``.
        """
        r = self.rotation()
        c = (self.x, self.y, self.z)
        t = [-sum(r[i][k] * c[k] for k in range(3)) for i in range(3)]

        # Columns r1, r2 and t form the plane-to-image map before intrinsics.
        m = [
            [r[0][0], r[0][1], t[0]],
            [r[1][0], r[1][1], t[1]],
            [r[2][0], r[2][1], t[2]],
        ]
        k = [[self.focal, 0.0, self.cx], [0.0, self.focal, self.cy], [0.0, 0.0, 1.0]]
        out = [
            sum(k[i][p] * m[p][j] for p in range(3)) for i in range(3) for j in range(3)
        ]
        if abs(out[8]) < 1e-12:
            return None
        return tuple(v / out[8] for v in out)

    def to_params(self) -> list[float]:
        return [self.focal, self.x, self.y, self.z, self.pan, self.tilt, self.roll]

    @staticmethod
    def from_params(p, cx: float, cy: float) -> CameraModel:
        return CameraModel(
            focal=p[0], x=p[1], y=p[2], z=p[3], pan=p[4], tilt=p[5], roll=p[6],
            cx=cx, cy=cy,
        )


def broadcast_prior(
    image_width: int, image_height: int, pitch_length: float = 105.0
) -> CameraModel:
    """A plausible main-gantry camera, used to initialise the fit.

    Main cameras sit near the halfway line, elevated, set back behind the near
    touchline. These numbers are not measured from any venue; they only need to
    be close enough for the optimiser to descend from, and the physical bounds
    in :func:`fit_camera` keep it honest.
    """
    return CameraModel(
        focal=1.3 * image_width,
        x=pitch_length / 2.0,
        y=-32.0,
        z=16.0,
        pan=0.0,
        tilt=math.radians(14.0),
        roll=0.0,
        cx=image_width / 2.0,
        cy=image_height / 2.0,
    )


@dataclass(frozen=True, slots=True)
class CameraFit:
    camera: CameraModel
    homography: tuple[float, ...]
    #: RMS residual in metres over all supplied evidence.
    error: float
    #: Number of residual terms the fit was built from.
    observations: int


def _invert3(h):
    m = h
    c = (
        m[4] * m[8] - m[5] * m[7], m[2] * m[7] - m[1] * m[8], m[1] * m[5] - m[2] * m[4],
        m[5] * m[6] - m[3] * m[8], m[0] * m[8] - m[2] * m[6], m[2] * m[3] - m[0] * m[5],
        m[3] * m[7] - m[4] * m[6], m[1] * m[6] - m[0] * m[7], m[0] * m[4] - m[1] * m[3],
    )
    det = m[0] * c[0] + m[1] * c[3] + m[2] * c[6]
    if abs(det) < 1e-12:
        return None
    return tuple(v / det for v in c)


def fit_camera(
    image_width: int,
    image_height: int,
    circle_points: list[Point] | None = None,
    touchline_points: list[Point] | None = None,
    halfway_points: list[Point] | None = None,
    pitch_length: float = 105.0,
    pitch_width: float = 68.0,
    initial: CameraModel | None = None,
    max_error: float = 3.0,
    touchline_y: float | None = None,
) -> CameraFit | None:
    """Fit a camera to whatever pitch evidence a frame offers.

    Args:
        circle_points: Image points lying on the centre circle.
        touchline_points: Image points on the **far** touchline (``y = width``),
            typically taken from the pitch-mask boundary rather than from paint.
        halfway_points: Image points on the halfway line (``x = length / 2``).
        initial: Starting guess; defaults to :func:`broadcast_prior`.
        max_error: Reject the fit above this RMS residual, in metres.

    Returns:
        A :class:`CameraFit`, or ``None`` when there is too little evidence, the
        optimiser fails, or the result does not meet ``max_error``.

    Requires evidence from at least two of the three sources: any single source
    leaves the camera badly underdetermined, and the optimiser will happily
    return a confident wrong answer from one.
    """
    try:
        import numpy as np
        from scipy.optimize import least_squares
    except ImportError:  # pragma: no cover - depends on environment
        return None

    circle_points = list(circle_points or [])
    touchline_points = list(touchline_points or [])
    halfway_points = list(halfway_points or [])

    sources = sum(
        1 for s in (circle_points, touchline_points, halfway_points) if len(s) >= 2
    )
    if sources < 2:
        return None

    # Which touchline the pitch-mask boundary corresponds to depends on which
    # side of the pitch the camera sits. Asserting one of them is a 68m lie
    # half the time, so when unspecified both are tried and the better fit
    # kept. Measured against ground truth, the boundary landed at y~0 in some
    # frames and y~68 in others.
    if touchline_y is None and touchline_points:
        best_fit = None
        for candidate in (0.0, pitch_width):
            attempt = fit_camera(
                image_width, image_height, circle_points, touchline_points,
                halfway_points, pitch_length, pitch_width, initial, max_error,
                touchline_y=candidate,
            )
            if attempt is not None and (best_fit is None or attempt.error < best_fit.error):
                best_fit = attempt
        return best_fit
    target_y = pitch_width if touchline_y is None else touchline_y

    centre = (pitch_length / 2.0, pitch_width / 2.0)
    init = initial or broadcast_prior(image_width, image_height, pitch_length)
    cx, cy = init.cx, init.cy

    def residuals(p):
        camera = CameraModel.from_params(p, cx, cy)
        h = camera.homography()
        if h is None:
            return np.full(
                len(circle_points) + len(touchline_points) + len(halfway_points), 1e3
            )
        inv = _invert3(h)
        if inv is None:
            return np.full(
                len(circle_points) + len(touchline_points) + len(halfway_points), 1e3
            )

        def to_pitch(pt):
            w = inv[6] * pt[0] + inv[7] * pt[1] + inv[8]
            if abs(w) < 1e-9:
                return None
            return (
                (inv[0] * pt[0] + inv[1] * pt[1] + inv[2]) / w,
                (inv[3] * pt[0] + inv[4] * pt[1] + inv[5]) / w,
            )

        out = []
        # Circle: every point should sit 9.15m from the centre spot.
        for pt in circle_points:
            q = to_pitch(pt)
            out.append(1e3 if q is None else math.dist(q, centre) - 9.15)
        # Touchline: whichever side the camera is on.
        for pt in touchline_points:
            q = to_pitch(pt)
            out.append(1e3 if q is None else q[1] - target_y)
        # Halfway line: x = pitch_length / 2.
        for pt in halfway_points:
            q = to_pitch(pt)
            out.append(1e3 if q is None else q[0] - pitch_length / 2.0)
        return np.asarray(out, dtype=float)

    # Physical bounds. These are what stop the optimiser reaching a
    # mathematically valid but impossible camera - underground, behind the
    # pitch, or with a focal length no broadcast lens has.
    lower = [0.3 * image_width, -60.0, -220.0, 3.0, -1.2, -0.15, -0.6]
    upper = [8.0 * image_width, pitch_length + 60.0, -3.0, 90.0, 1.2, 1.3, 0.6]
    start = [min(max(v, lo), hi) for v, lo, hi in zip(init.to_params(), lower, upper, strict=True)]

    try:
        result = least_squares(
            residuals, start, bounds=(lower, upper), max_nfev=400, xtol=1e-8, ftol=1e-8
        )
    except Exception:  # pragma: no cover - optimiser blow-up
        return None

    camera = CameraModel.from_params(result.x, cx, cy)
    h = camera.homography()
    if h is None:
        return None
    inv = _invert3(h)
    if inv is None:
        return None

    n = len(result.fun)
    rms = float((result.fun @ result.fun / n) ** 0.5) if n else float("inf")
    if not math.isfinite(rms) or rms > max_error:
        return None
    return CameraFit(camera=camera, homography=inv, error=rms, observations=n)
