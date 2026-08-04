"""SoccerNet annotation adapter.

Verified by generating annotations *from a known homography* in SoccerNet's own
format, then checking the adapter recovers that homography. Real SoccerNet data
is NDA-gated, so this is how the adapter is validated before the data exists.
"""

from __future__ import annotations

import json
import math

import pytest

from offball import kernels
from offball.datasets.soccernet import (
    homography_from_annotation,
    iter_annotated_frames,
    pitch_lines,
)

WIDTH, HEIGHT = 1920, 1080
PITCH_L, PITCH_W = 105.0, 68.0

# A plausible pitch -> image transform (gantry camera at the halfway line).
H_PITCH_TO_IMAGE = (12.0, 5.3, 250.0, 0.2, -6.5, 980.0, 0.0, 0.004, 1.0)


def to_image(p):
    q = kernels.project(H_PITCH_TO_IMAGE, [p])[0]
    assert q is not None
    return q


def make_annotation(classes=None, samples=6) -> dict:
    """Render pitch lines into a SoccerNet-format annotation.

    Coordinates are normalised to [0, 1] and scaled by (size - 1), matching
    SoccerNet's own convention.
    """
    geometry = pitch_lines(PITCH_L, PITCH_W)
    classes = classes or list(geometry)
    out: dict[str, list[dict[str, float]]] = {}

    for name in classes:
        axis, coord = geometry[name]
        points = []
        for i in range(samples):
            t = i / (samples - 1)
            pitch = (coord, t * PITCH_W) if axis == "x" else (t * PITCH_L, coord)
            px, py = to_image(pitch)
            points.append({"x": px / (WIDTH - 1), "y": py / (HEIGHT - 1)})
        out[name] = points
    return out


def recovery_error(homography) -> float:
    """Mean pitch-space error, in metres, over probes across the pitch."""
    probes = [(x, y) for x in (5.0, 52.5, 100.0) for y in (5.0, 34.0, 63.0)]
    total = 0.0
    for p in probes:
        got = kernels.project(homography, [to_image(p)])[0]
        assert got is not None
        total += math.dist(got, p)
    return total / len(probes)


def test_recovers_a_known_homography():
    h = homography_from_annotation(make_annotation(), WIDTH, HEIGHT)
    assert h is not None
    assert recovery_error(h) < 0.5


def test_works_with_only_four_lines():
    """A frame showing one penalty area and both touchlines is enough."""
    h = homography_from_annotation(
        make_annotation(
            [
                "Side line top",
                "Side line bottom",
                "Big rect. left main",
                "Side line left",
            ]
        ),
        WIDTH,
        HEIGHT,
    )
    assert h is not None
    assert recovery_error(h) < 1.0


def test_needs_both_line_directions():
    """Parallel lines alone cannot determine a homography."""
    only_x = make_annotation(["Side line left", "Side line right", "Middle line"])
    assert homography_from_annotation(only_x, WIDTH, HEIGHT) is None

    only_y = make_annotation(["Side line top", "Side line bottom"])
    assert homography_from_annotation(only_y, WIDTH, HEIGHT) is None


def test_too_few_lines_returns_none():
    assert homography_from_annotation({}, WIDTH, HEIGHT) is None
    assert homography_from_annotation(make_annotation(["Side line top"]), WIDTH, HEIGHT) is None


def test_circles_and_goal_frames_are_ignored():
    """The goal frame is 2.44m off the ground and would corrupt a plane fit."""
    annotation = make_annotation()
    annotation["Circle central"] = [{"x": 0.5, "y": 0.5}, {"x": 0.6, "y": 0.55}]
    annotation["Goal left crossbar"] = [{"x": 0.1, "y": 0.2}, {"x": 0.15, "y": 0.2}]
    annotation["Line unknown"] = [{"x": 0.0, "y": 0.0}, {"x": 1.0, "y": 1.0}]

    h = homography_from_annotation(annotation, WIDTH, HEIGHT)
    assert h is not None
    assert recovery_error(h) < 0.5, "unusable classes must not affect the fit"


def test_a_stray_annotation_is_rejected_by_ransac():
    annotation = make_annotation()
    # An annotator mislabelling one line puts every intersection using it wrong.
    annotation["Small rect. right main"] = [
        {"x": 0.05, "y": 0.9},
        {"x": 0.9, "y": 0.05},
    ]
    h = homography_from_annotation(annotation, WIDTH, HEIGHT)
    assert h is not None
    assert recovery_error(h) < 1.5


def test_malformed_points_are_skipped_not_raised():
    annotation = make_annotation()
    annotation["Side line left"] = [{"bad": 1}, {"bad": 2}]
    annotation["Middle line"] = "not a list"
    h = homography_from_annotation(annotation, WIDTH, HEIGHT)
    assert h is not None


def test_pitch_lines_geometry_matches_the_laws():
    g = pitch_lines(105.0, 68.0)
    assert g["Side line left"] == ("x", 0.0)
    assert g["Side line right"] == ("x", 105.0)
    assert g["Middle line"] == ("x", 52.5)
    assert g["Big rect. left main"] == ("x", 16.5)
    # Penalty area 40.32m wide, centred: 34 +/- 20.16
    assert g["Big rect. left top"][1] == pytest.approx(13.84)
    assert g["Big rect. left bottom"][1] == pytest.approx(54.16)
    # Goal area 18.32m wide: 34 +/- 9.16
    assert g["Small rect. left top"][1] == pytest.approx(24.84)
    assert g["Small rect. left bottom"][1] == pytest.approx(43.16)


def test_pitch_lines_scale_with_pitch_size():
    g = pitch_lines(100.0, 64.0)
    assert g["Side line right"] == ("x", 100.0)
    assert g["Big rect. right main"] == ("x", 83.5)
    # Fixed distances do not scale with the pitch.
    assert g["Big rect. left top"][1] == pytest.approx(32.0 - 20.16)


def test_iter_annotated_frames(tmp_path):
    cv2 = pytest.importorskip("cv2")
    import numpy as np

    image = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
    cv2.imwrite(str(tmp_path / "frame_000.jpg"), image)
    (tmp_path / "frame_000.json").write_text(json.dumps(make_annotation()))

    # A frame with no annotation at all.
    cv2.imwrite(str(tmp_path / "frame_001.jpg"), image)

    results = list(iter_annotated_frames(tmp_path))
    assert len(results) == 2
    assert results[0][1] is not None
    assert results[1][1] is None, "an unannotated frame must yield None, not be skipped"
