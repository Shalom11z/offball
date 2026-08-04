"""Calibration: fitting, quality gates, and the temporal continuity check."""

from __future__ import annotations

import math

import pytest

from offball.vision.calibration import (
    Calibration,
    CalibrationConfig,
    HomographySmoother,
    ScriptedKeypoints,
    calibrate_frame,
)

# A plausible broadcast image -> pitch transform.
H = (0.055, 0.004, -6.0, -0.002, 0.030, -8.0, -0.00002, 0.00048, 1.0)


def apply(h, p):
    w = h[6] * p[0] + h[7] * p[1] + h[8]
    return ((h[0] * p[0] + h[1] * p[1] + h[2]) / w, (h[3] * p[0] + h[4] * p[1] + h[5]) / w)


def correspondences(n: int = 8) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """Image points spread over the frame, with their true pitch positions."""
    image_points = [
        (200.0, 800.0), (1700.0, 820.0), (300.0, 400.0), (1600.0, 420.0),
        (960.0, 600.0), (700.0, 950.0), (1200.0, 300.0), (500.0, 650.0),
    ][:n]
    return [(p, apply(H, p)) for p in image_points]


def test_calibrate_recovers_the_transform():
    cal = calibrate_frame(correspondences())
    assert cal is not None
    assert cal.error < 0.01
    assert cal.inlier_ratio == 1.0

    projected = cal.to_pitch([(960.0, 600.0)])[0]
    expected = apply(H, (960.0, 600.0))
    assert projected is not None
    assert math.dist(projected, expected) < 0.01


def test_too_few_keypoints_returns_none():
    assert calibrate_frame(correspondences(3)) is None
    assert calibrate_frame([]) is None


def test_min_inliers_gate_rejects_thin_support():
    config = CalibrationConfig(min_inliers=7)
    assert calibrate_frame(correspondences(5), config) is None


def test_a_badly_corrupted_keypoint_set_is_rejected():
    # Half the correspondences are wrong: no consensus reaches min_inliers.
    corr = correspondences()
    corrupted = corr[:4] + [(p, (pitch[0] + 60.0, pitch[1] - 40.0)) for p, pitch in corr[4:]]
    config = CalibrationConfig(min_inliers=7, ransac_threshold=1.0)
    assert calibrate_frame(corrupted, config) is None


def test_outliers_are_excluded_but_a_good_fit_survives():
    corr = correspondences()
    corr.append(((900.0, 500.0), (999.0, -999.0)))
    cal = calibrate_frame(corr, CalibrationConfig(ransac_threshold=1.0, min_inliers=6))
    assert cal is not None
    assert cal.inliers == 8
    assert cal.total_keypoints == 9
    assert cal.inlier_ratio == pytest.approx(8 / 9)


def test_smoother_passes_a_steady_camera_through():
    smoother = HomographySmoother()
    cal = calibrate_frame(correspondences())
    for _ in range(5):
        assert smoother.push(cal) is cal
    assert smoother.rejected == 0


def test_smoother_rejects_an_implausible_jump():
    smoother = HomographySmoother(CalibrationConfig(max_centre_jump=15.0))
    good = calibrate_frame(correspondences())
    assert smoother.push(good) is good

    # The classic symmetry mis-solve: geometrically self-consistent, but it
    # puts the image centre 50m from where it was 40ms ago.
    shifted = [(p, (pitch[0] + 50.0, pitch[1])) for p, pitch in correspondences()]
    bad = calibrate_frame(shifted)
    assert bad is not None, "the bad fit is internally consistent - that is the point"

    result = smoother.push(bad)
    assert smoother.rejected == 1
    assert result is good, "rejection coasts on the last good solution"


def test_smoother_coasts_then_gives_up():
    smoother = HomographySmoother(max_coast=3)
    good = calibrate_frame(correspondences())
    smoother.push(good)
    # Three uncalibrated frames still return the last good homography.
    for _ in range(3):
        assert smoother.push(None) is good
    # Beyond the coast limit it admits it does not know.
    assert smoother.push(None) is None


def test_smoother_reset_clears_history():
    smoother = HomographySmoother()
    smoother.push(calibrate_frame(correspondences()))
    smoother.reset()
    assert smoother.push(None) is None
    assert smoother.rejected == 0


def test_scripted_keypoints_handle_frames_with_no_pitch_view():
    corr = correspondences()
    source = ScriptedKeypoints([corr, None, corr])
    assert len(source.keypoints()) == 8
    assert source.keypoints() == [], "a close-up yields no keypoints, not a crash"
    assert len(source.keypoints()) == 8
    assert source.keypoints() == [], "past the end of the script"


def test_calibration_inlier_ratio_with_no_keypoints():
    cal = Calibration(matrix=H, error=0.0, inliers=0, total_keypoints=0)
    assert cal.inlier_ratio == 0.0
