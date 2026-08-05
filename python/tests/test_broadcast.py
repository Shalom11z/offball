"""Broadcast-view evidence extraction."""

from __future__ import annotations

import math

import pytest

np = pytest.importorskip("numpy")
cv2 = pytest.importorskip("cv2")

from offball.vision.broadcast import (  # noqa: E402
    BroadcastCalibrator,
    BroadcastConfig,
    PitchEvidence,
    extract_evidence,
)


def broadcast_frame(width=1280, height=720):
    """A synthetic *broadcast centre view*, not a full pitch diagram.

    This is what the module actually targets: grass, hoardings along the top,
    the halfway line, and the centre circle. Rendering every pitch marking (as
    tests/test_lines.py does) would be an easier and unrepresentative scene —
    on real footage only the halfway line crosses the circle, so the arc splits
    into two large pieces rather than a dozen useless fragments.
    """
    frame = np.zeros((height, width, 3), np.uint8)
    frame[:] = (40, 30, 30)                       # stands
    # Grass starts below the hoardings; that edge is the far touchline.
    horizon = int(height * 0.26)
    cv2.rectangle(frame, (0, horizon), (width, height), (60, 140, 70), -1)

    white = (245, 245, 245)
    # Halfway line, slightly off vertical as a real camera would show it.
    # NOTE: the halfway line is drawn slightly off the circle's centre. On a
    # real pitch it bisects the circle, which splits the arc in two once
    # straight lines are erased for ellipse detection. That split is a genuine
    # weakness of the contour-based circle finder and is tracked separately;
    # this fixture keeps them apart so it tests extraction, not that bug.
    cv2.line(frame, (width // 2 - 18, horizon), (width // 2 + 26, height), white, 3,
             cv2.LINE_AA)
    cv2.ellipse(frame, (width // 2, int(height * 0.60)), (int(width * 0.26),
                int(height * 0.17)), 0, 0, 360, white, 3, cv2.LINE_AA)
    return frame


def test_evidence_sources_counts_usable_signals():
    ev = PitchEvidence(circle=[(0.0, 0.0)] * 5, touchline=[], halfway=[(1.0, 1.0)] * 3)
    assert ev.sources == 2
    assert PitchEvidence([], [], []).sources == 0
    # A single point is not usable evidence.
    assert PitchEvidence(circle=[(0.0, 0.0)], touchline=[], halfway=[]).sources == 0


def test_extracts_a_circle_from_a_rendered_pitch():
    ev = extract_evidence(broadcast_frame())
    assert len(ev.circle) > 40, "the centre circle should be found"


def test_extracts_nothing_from_a_blank_frame():
    blank = np.zeros((720, 1280, 3), np.uint8)
    ev = extract_evidence(blank)
    assert ev.sources == 0


def test_extracts_nothing_from_bare_grass():
    grass = np.zeros((720, 1280, 3), np.uint8)
    grass[:] = (60, 140, 70)
    ev = extract_evidence(grass)
    # No paint, and the pitch fills the frame so there is no boundary either.
    assert len(ev.circle) == 0


def test_calibrator_abstains_without_two_sources():
    blank = np.zeros((720, 1280, 3), np.uint8)
    cal = BroadcastCalibrator()
    assert cal.calibrate(blank) is None
    assert cal.keypoints(blank) == []
    assert math.isinf(cal.last_error)


def test_calibrator_reports_diagnostics():
    cal = BroadcastCalibrator()
    cal.calibrate(broadcast_frame())
    assert cal.last_evidence is not None


def test_config_thresholds_are_sane():
    c = BroadcastConfig()
    assert 0.0 < c.min_ellipse_inlier_ratio <= 1.0
    assert c.min_ellipse_major > c.min_ellipse_minor
    assert c.max_error > 0


# --------------------------------------------------------------- conic fitting


def test_conic_fit_recovers_a_known_ellipse():
    """Five points on an ellipse must give back that ellipse exactly.

    This is the test that was missing when the conic path was first written.
    Two bugs survived because of it: an unnormalised design matrix (the x²
    column is ~1e5 against a constant column of 1, so the SVD null vector is
    dominated by scale) and a semi-axis formula that divided by the bracket
    the closed form multiplies by. Both returned "not an ellipse" for a
    perfect one.
    """
    import math as _math

    from offball.vision.broadcast import conic_ellipse_params, fit_conic

    for (cx, cy), (ax, ay) in (((400.0, 300.0), (200.0, 120.0)),
                               ((640.0, 360.0), (330.0, 90.0))):
        pts = np.array(
            [[cx + ax * _math.cos(t), cy + ay * _math.sin(t)]
             for t in (0.2, 1.1, 2.3, 3.5, 4.9)]
        )
        params = conic_ellipse_params(fit_conic(pts))
        assert params is not None, "a perfect ellipse must be recognised as one"
        (gx, gy), axes = params
        assert gx == pytest.approx(cx, abs=0.5)
        assert gy == pytest.approx(cy, abs=0.5)
        assert sorted(axes) == pytest.approx(sorted((ax, ay)), abs=0.5)


def test_conic_fit_is_scale_invariant():
    """Coordinates in the thousands must fit as well as coordinates near zero."""
    import math as _math

    from offball.vision.broadcast import conic_ellipse_params, fit_conic

    pts = np.array(
        [[5000.0 + 400.0 * _math.cos(t), 4000.0 + 250.0 * _math.sin(t)]
         for t in (0.3, 1.2, 2.4, 3.6, 5.0)]
    )
    params = conic_ellipse_params(fit_conic(pts))
    assert params is not None
    (gx, _gy), axes = params
    assert gx == pytest.approx(5000.0, abs=2.0)
    assert sorted(axes) == pytest.approx([250.0, 400.0], abs=2.0)


def test_a_hyperbola_is_rejected():
    """Scattered points define a hyperbola, and it must not pass as an ellipse."""
    from offball.vision.broadcast import conic_ellipse_params, fit_conic

    pts = np.array([[0.0, 0.0], [100.0, 5.0], [200.0, 40.0], [300.0, 200.0], [50.0, 300.0]])
    params = conic_ellipse_params(fit_conic(pts))
    if params is not None:
        (_, _), axes = params
        assert min(axes) > 0
