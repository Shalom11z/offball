"""SoccerNet-Calibration adapter.

Turns SoccerNet's pitch-line annotations into the ground-truth homographies
:mod:`offball.benchmark` needs, so the classical detector can be scored against
real broadcast footage.

The data is behind a short NDA form at <https://www.soccer-net.org/data>. With
the password::

    from SoccerNet.Downloader import SoccerNetDownloader as D
    d = D(LocalDirectory="data/soccernet")
    d.downloadDataTask(task="calibration-2023", split=["test"], password="...")

Then::

    offball benchmark data/soccernet/calibration-2023/test --soccernet

## The annotation format

Each frame has a JSON alongside it mapping a pitch-element name to a polyline::

    {"Side line top": [{"x": 0.13, "y": 0.41}, {"x": 0.88, "y": 0.39}], ...}

Coordinates are **normalised to [0, 1]** and scaled by ``width - 1`` /
``height - 1``. Names come from ``SoccerPitch.lines_classes``.

## Why only the straight lines are used

SoccerNet annotates circles and goal frames too. Circles are not lines, and the
goal frame is not on the ground plane at all — a homography maps the *pitch
plane*, so a crossbar 2.44m in the air would corrupt the fit. Only the 13
straight ground lines are used, which is exactly the set the detector itself
matches against.

## Coordinate conversion

SoccerNet puts the origin at the centre spot, x in [-L/2, L/2] and y in
[-W/2, W/2] with "top" negative. This package puts the origin at a corner, x in
[0, L] and y in [0, W]. The conversion is applied once, here.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterator
from pathlib import Path

from ..kernels import fit_homography
from ..types import Point

__all__ = [
    "SOCCERNET_LINE_CLASSES",
    "homography_from_annotation",
    "iter_annotated_frames",
    "pitch_lines",
]

#: The straight ground lines, as ``(class name, axis, offset)`` where ``axis``
#: is the constant pitch coordinate. Offsets are given in SoccerNet's
#: centre-origin frame and converted below.
#:
#: Penalty area is 40.32m wide (±20.16), goal area 18.32m (±9.16); both are
#: fixed by the laws of the game and do not scale with pitch size.
_LINE_SPECS: tuple[tuple[str, str, float], ...] = (
    ("Side line left", "x", -0.5),          # fraction of pitch length
    ("Side line right", "x", 0.5),
    ("Middle line", "x", 0.0),
    ("Big rect. left main", "x", -0.5),      # + 16.5m, handled below
    ("Big rect. right main", "x", 0.5),      # - 16.5m
    ("Small rect. left main", "x", -0.5),    # + 5.5m
    ("Small rect. right main", "x", 0.5),    # - 5.5m
    ("Side line top", "y", -0.5),            # fraction of pitch width
    ("Side line bottom", "y", 0.5),
    ("Big rect. left top", "y", 0.0),        # -20.16m
    ("Big rect. left bottom", "y", 0.0),     # +20.16m
    ("Big rect. right top", "y", 0.0),
    ("Big rect. right bottom", "y", 0.0),
    ("Small rect. left top", "y", 0.0),      # -9.16m
    ("Small rect. left bottom", "y", 0.0),
    ("Small rect. right top", "y", 0.0),
    ("Small rect. right bottom", "y", 0.0),
)

SOCCERNET_LINE_CLASSES: tuple[str, ...] = tuple(name for name, _, _ in _LINE_SPECS)


def pitch_lines(
    pitch_length: float = 105.0, pitch_width: float = 68.0
) -> dict[str, tuple[str, float]]:
    """Map each SoccerNet line class to ``(axis, coordinate)`` in our frame.

    ``axis`` is ``"x"`` for lines parallel to the goal lines and ``"y"`` for
    lines parallel to the touchlines; ``coordinate`` is the constant value of
    that axis, in metres from the corner origin.
    """
    half_w = pitch_width / 2.0
    return {
        # Lines of constant x.
        "Side line left": ("x", 0.0),
        "Side line right": ("x", pitch_length),
        "Middle line": ("x", pitch_length / 2.0),
        "Big rect. left main": ("x", 16.5),
        "Big rect. right main": ("x", pitch_length - 16.5),
        "Small rect. left main": ("x", 5.5),
        "Small rect. right main": ("x", pitch_length - 5.5),
        # Lines of constant y. SoccerNet's "top" is negative y (our 0 side).
        "Side line top": ("y", 0.0),
        "Side line bottom": ("y", pitch_width),
        "Big rect. left top": ("y", half_w - 20.16),
        "Big rect. left bottom": ("y", half_w + 20.16),
        "Big rect. right top": ("y", half_w - 20.16),
        "Big rect. right bottom": ("y", half_w + 20.16),
        "Small rect. left top": ("y", half_w - 9.16),
        "Small rect. left bottom": ("y", half_w + 9.16),
        "Small rect. right top": ("y", half_w - 9.16),
        "Small rect. right bottom": ("y", half_w + 9.16),
    }


def _fit_image_line(points: list[Point]) -> tuple[float, float, float] | None:
    """Total-least-squares line ``ax + by = c`` through annotated points.

    Orthogonal rather than vertical regression: an annotated touchline in a
    broadcast frame can be near-vertical in image space, where ordinary
    least squares blows up.
    """
    if len(points) < 2:
        return None
    n = len(points)
    mx = sum(p[0] for p in points) / n
    my = sum(p[1] for p in points) / n

    sxx = sum((p[0] - mx) ** 2 for p in points)
    syy = sum((p[1] - my) ** 2 for p in points)
    sxy = sum((p[0] - mx) * (p[1] - my) for p in points)

    # Principal direction of the scatter; the normal is perpendicular to it.
    theta = 0.5 * math.atan2(2.0 * sxy, sxx - syy)
    a, b = -math.sin(theta), math.cos(theta)
    norm = math.hypot(a, b)
    if norm < 1e-12:
        return None
    a, b = a / norm, b / norm
    return (a, b, a * mx + b * my)


def _intersect(
    l1: tuple[float, float, float], l2: tuple[float, float, float]
) -> Point | None:
    a1, b1, c1 = l1
    a2, b2, c2 = l2
    det = a1 * b2 - a2 * b1
    if abs(det) < 1e-9:
        return None  # parallel
    return ((c1 * b2 - c2 * b1) / det, (a1 * c2 - a2 * c1) / det)


def homography_from_annotation(
    annotation: dict,
    image_width: int,
    image_height: int,
    pitch_length: float = 105.0,
    pitch_width: float = 68.0,
    min_lines: int = 4,
) -> tuple[float, ...] | None:
    """Derive an image -> pitch homography from one SoccerNet annotation.

    Uses the same trick as the detector: a homography maps lines to lines, so
    the intersection of two annotated image lines corresponds to the
    intersection of the two pitch lines they denote — including intersections
    where no paint exists, such as the halfway line crossed with a penalty-area
    edge.

    Args:
        annotation: The parsed JSON for one frame.
        image_width: Pixel width the normalised coordinates refer to.
        image_height: Pixel height.
        pitch_length: Metres.
        pitch_width: Metres.
        min_lines: Minimum usable straight lines; fewer returns ``None``.

    Returns:
        A flat row-major 3x3 homography, or ``None`` when the frame does not
        carry enough straight-line annotation to determine one. Frames showing
        only a goalmouth or a centre circle legitimately fall in this category.
    """
    geometry = pitch_lines(pitch_length, pitch_width)

    x_lines: list[tuple[float, tuple[float, float, float]]] = []
    y_lines: list[tuple[float, tuple[float, float, float]]] = []

    for name, points in annotation.items():
        spec = geometry.get(name)
        if spec is None or not points:
            continue  # circles, goal frames, unknown classes
        try:
            pixels = [
                (float(p["x"]) * (image_width - 1), float(p["y"]) * (image_height - 1))
                for p in points
            ]
        except (KeyError, TypeError, ValueError):
            continue
        line = _fit_image_line(pixels)
        if line is None:
            continue
        axis, coordinate = spec
        (x_lines if axis == "x" else y_lines).append((coordinate, line))

    if len(x_lines) + len(y_lines) < min_lines:
        return None
    # Need both directions: intersections of parallel lines are useless.
    if len(x_lines) < 2 or len(y_lines) < 2:
        return None

    src: list[Point] = []
    dst: list[Point] = []
    for px, xline in x_lines:
        for py, yline in y_lines:
            point = _intersect(xline, yline)
            if point is None:
                continue
            src.append(point)
            dst.append((px, py))

    if len(src) < 4:
        return None
    try:
        # RANSAC rather than a plain fit: an annotator's stray polyline shows
        # up as a whole bad line, and every intersection involving it is wrong.
        matrix, _ = fit_homography(src, dst, threshold=2.0, iterations=200, seed=3)
    except ValueError:
        return None
    return matrix


def iter_annotated_frames(
    directory: str | Path,
    pitch_length: float = 105.0,
    pitch_width: float = 68.0,
) -> Iterator[tuple[Path, tuple[float, ...] | None]]:
    """Yield ``(image_path, homography)`` for a SoccerNet split directory.

    Pairs each image with its same-stem ``.json`` annotation. Images whose
    annotation is missing or too sparse yield ``None``, so the benchmark can
    count them as unscoreable rather than skipping them silently — a detector
    should not be credited for frames nobody could calibrate.
    """
    import cv2

    directory = Path(directory)
    images = sorted(
        p for p in directory.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    for image_path in images:
        annotation_path = image_path.with_suffix(".json")
        if not annotation_path.exists():
            yield image_path, None
            continue
        try:
            annotation = json.loads(annotation_path.read_text())
        except json.JSONDecodeError:
            yield image_path, None
            continue

        image = cv2.imread(str(image_path))
        if image is None:
            yield image_path, None
            continue
        height, width = image.shape[:2]
        yield image_path, homography_from_annotation(
            annotation, width, height, pitch_length, pitch_width
        )
