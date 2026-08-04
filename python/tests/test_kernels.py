"""Tests for the numeric kernels.

These run against whichever backend is installed, so the same suite validates
the pure-Python reference and the compiled Rust extension.
"""

from __future__ import annotations

import math

import pytest

from offball import kernels
from offball.kernels import ControlParams

# ---------------------------------------------------------------- homography


def _sample_h() -> tuple[float, ...]:
    return (1.2, 0.35, 40.0, -0.15, 0.9, 25.0, 0.0006, 0.0011, 1.0)


def _apply(h, p):
    w = h[6] * p[0] + h[7] * p[1] + h[8]
    return ((h[0] * p[0] + h[1] * p[1] + h[2]) / w, (h[3] * p[0] + h[4] * p[1] + h[5]) / w)


def test_fit_homography_recovers_a_known_transform():
    h = _sample_h()
    src = [(0.0, 0.0), (105.0, 0.0), (105.0, 68.0), (0.0, 68.0), (52.5, 34.0), (16.5, 13.85)]
    dst = [_apply(h, p) for p in src]

    fitted, inliers = kernels.fit_homography(src, dst, threshold=0.5, seed=1)
    assert len(inliers) == len(src)
    for p, expected in zip(src, dst):
        got = kernels.project(fitted, [p])[0]
        assert got is not None
        assert math.dist(got, expected) < 1e-6


def test_fit_homography_rejects_outliers():
    h = _sample_h()
    src = [
        (0.0, 0.0), (105.0, 0.0), (105.0, 68.0), (0.0, 68.0),
        (52.5, 34.0), (16.5, 13.85), (88.5, 54.15), (52.5, 0.0),
    ]
    dst = [_apply(h, p) for p in src]
    # A keypoint model confusing the two penalty boxes.
    src = [*src, (20.0, 20.0), (70.0, 50.0)]
    dst = [*dst, (900.0, -400.0), (-250.0, 700.0)]

    _, inliers = kernels.fit_homography(src, dst, threshold=1.0, iterations=500, seed=42)
    assert set(inliers) == set(range(8))


def test_fit_homography_is_deterministic():
    h = _sample_h()
    src = [(i * 8.0, 34.0 + 30.0 * math.sin(i * 1.7)) for i in range(12)]
    dst = [_apply(h, p) for p in src]
    a = kernels.fit_homography(src, dst, threshold=0.5, iterations=200, seed=7)
    b = kernels.fit_homography(src, dst, threshold=0.5, iterations=200, seed=7)
    assert a[1] == b[1]


def test_fit_homography_validates_input():
    with pytest.raises(ValueError, match="same length"):
        kernels.fit_homography([(0.0, 0.0)], [(0.0, 0.0), (1.0, 1.0)])
    with pytest.raises(ValueError, match="at least 4"):
        kernels.fit_homography([(0.0, 0.0)] * 3, [(0.0, 0.0)] * 3)


def test_collinear_correspondences_fail_rather_than_return_nonsense():
    # Every point on one line: the homography is genuinely undetermined.
    src = [(float(i), float(i)) for i in range(8)]
    dst = [(2.0 * i, 2.0 * i) for i in range(8)]
    with pytest.raises(ValueError):
        kernels.fit_homography(src, dst, threshold=0.01, iterations=50, seed=3)


def test_project_returns_none_on_the_horizon():
    h = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, -5.0)
    assert kernels.project(h, [(5.0, 2.0)])[0] is None


def test_project_validates_matrix_size():
    with pytest.raises(ValueError, match="9 elements"):
        kernels.project((1.0, 0.0, 0.0), [(0.0, 0.0)])


# -------------------------------------------------------------- pitch control


def test_lone_team_controls_the_whole_pitch():
    g = kernels.pitch_control([(50.0, 34.0)], [(0.0, 0.0)], [True], nx=20, ny=20)
    assert all(abs(v - 1.0) < 1e-9 for v in g.values)
    assert g.total() == pytest.approx(105.0 * 68.0)


def test_no_players_gives_a_uniform_half_field():
    g = kernels.pitch_control([], [], [], nx=10, ny=10)
    assert all(abs(v - 0.5) < 1e-9 for v in g.values)


def test_control_is_bounded_and_mirrors_under_team_swap():
    pos = [(30.0, 20.0), (70.0, 50.0)]
    vel = [(2.0, 0.0), (-1.0, 1.0)]
    a = kernels.pitch_control(pos, vel, [True, False], nx=16, ny=16)
    b = kernels.pitch_control(pos, vel, [False, True], nx=16, ny=16)
    for va, vb in zip(a.values, b.values):
        assert 0.0 <= va <= 1.0
        assert va + vb == pytest.approx(1.0)


def test_control_validates_shapes():
    with pytest.raises(ValueError, match="same length"):
        kernels.pitch_control([(0.0, 0.0)], [], [True])
    with pytest.raises(ValueError, match="positive"):
        kernels.pitch_control([], [], [], nx=0)


def test_running_at_a_target_shortens_arrival_time():
    params = ControlParams()
    target = (70.0, 34.0)
    still = params.time_to_intercept((50.0, 34.0), (0.0, 0.0), target)
    sprinting = params.time_to_intercept((50.0, 34.0), (6.0, 0.0), target)
    assert sprinting < still


def test_space_ownership_tiles_the_pitch():
    pos = [(20.0, 20.0), (80.0, 20.0), (50.0, 55.0)]
    vel = [(0.0, 0.0)] * 3
    owned = kernels.space_ownership(pos, vel, [True, False, True], nx=60, ny=40)
    assert sum(owned) == pytest.approx(105.0 * 68.0)
    assert all(a > 0 for a in owned)


def test_space_ownership_of_nobody_is_empty():
    assert kernels.space_ownership([], [], []) == []


def test_pitch_value_rises_toward_goal():
    far = kernels.pitch_value((10.0, 34.0))
    mid = kernels.pitch_value((60.0, 34.0))
    near = kernels.pitch_value((98.0, 34.0))
    assert far < mid < near
    assert 0.0 <= near <= 1.0
    # Central beats the byline corner at a similar distance.
    assert near > kernels.pitch_value((98.0, 2.0))


def test_dangerous_space_prefers_the_final_third():
    vel = [(0.0, 0.0), (0.0, 0.0)]
    deep = kernels.pitch_control([(15.0, 34.0), (90.0, 34.0)], vel, [True, False], nx=40, ny=30)
    high = kernels.pitch_control([(90.0, 34.0), (15.0, 34.0)], vel, [True, False], nx=40, ny=30)
    assert kernels.dangerous_space(high) > kernels.dangerous_space(deep)


# ------------------------------------------------------------------- metrics


def test_offside_line_uses_the_second_last_defender():
    assert kernels.offside_line([2.0, 40.0, 42.0, 45.0], 30.0) == pytest.approx(42.0)


def test_ball_ahead_of_the_defence_pushes_the_line_forward():
    assert kernels.offside_line([2.0, 40.0, 42.0, 45.0], 60.0) == pytest.approx(60.0)


def test_offside_line_abstains_without_two_defenders():
    assert kernels.offside_line([40.0], 10.0) is None
    assert kernels.offside_line([], 10.0) is None


def test_passing_lane_detects_a_blocker():
    v = kernels.passing_lane((0.0, 0.0), (20.0, 0.0), [(10.0, 0.5)], corridor=1.2)
    assert not v.open
    assert v.blockers == 1
    assert v.clearance == pytest.approx(0.5)


def test_defender_behind_the_passer_does_not_block():
    v = kernels.passing_lane((10.0, 0.0), (30.0, 0.0), [(0.0, 0.0)], corridor=1.2)
    assert v.open


def test_empty_defence_leaves_the_lane_open():
    v = kernels.passing_lane((0.0, 0.0), (10.0, 0.0), [], corridor=1.2)
    assert v.open and v.blockers == 0
    assert math.isinf(v.clearance)


def test_team_shape_distinguishes_compact_from_stretched():
    compact = kernels.team_shape([(40.0, 20.0), (45.0, 30.0), (42.0, 40.0), (48.0, 48.0)])
    stretched = kernels.team_shape([(10.0, 5.0), (50.0, 30.0), (90.0, 60.0), (70.0, 10.0)])
    assert compact.hull_area < stretched.hull_area
    assert compact.depth < stretched.depth
    assert compact.centroid[0] == pytest.approx(43.75)


def test_team_shape_of_nobody_is_none():
    assert kernels.team_shape([]) is None


def test_defensive_lines_recover_a_442():
    xs = [29.0, 30.0, 31.0, 30.5, 49.0, 50.0, 51.0, 50.5, 70.0, 71.0]
    lines = kernels.defensive_lines(xs, k=3)
    assert len(lines) == 3
    assert lines == sorted(lines)
    assert lines[0] == pytest.approx(30.125, abs=1.0)
    assert lines[1] == pytest.approx(50.125, abs=1.0)
    assert lines[2] == pytest.approx(70.5, abs=1.0)


def test_defensive_lines_edge_cases():
    assert kernels.defensive_lines([], 3) == []
    assert kernels.defensive_lines([30.0, 50.0], 0) == []
    assert len(kernels.defensive_lines([30.0, 31.0, 50.0, 70.0], 99)) == 4


def test_lines_broken_counts_banks_passed():
    lines = [30.0, 50.0, 70.0]
    assert kernels.lines_broken(20.0, lines) == 0
    assert kernels.lines_broken(60.0, lines) == 2
    assert kernels.lines_broken(80.0, lines) == 3


def test_marking_pressure_decays_with_distance():
    p = (50.0, 34.0)
    assert kernels.marking_pressure(p, [(50.5, 34.0)]) > 0.8
    assert kernels.marking_pressure(p, [(65.0, 34.0)]) < 0.1
    assert kernels.marking_pressure(p, []) == 0.0


def test_nearest_opponent_distance():
    d = kernels.nearest_opponent_distance((0.0, 0.0), [(10.0, 0.0), (3.0, 4.0)])
    assert d == pytest.approx(5.0)
    assert kernels.nearest_opponent_distance((0.0, 0.0), []) is None


def test_fit_dlt_uses_every_correspondence():
    """Exact fit on a consistent set, with no RANSAC sampling involved."""
    h = _sample_h()
    src = [(x, y) for x in (0.0, 16.5, 52.5, 105.0) for y in (0.0, 13.84, 54.16, 68.0)]
    dst = [_apply(h, p) for p in src]

    fitted = kernels.fit_dlt(src, dst)
    for p, expected in zip(src, dst):
        got = kernels.project(fitted, [p])[0]
        assert got is not None
        assert math.dist(got, expected) < 1e-6


def test_fit_dlt_succeeds_where_a_single_ransac_sample_would_fail():
    """The reason fit_dlt exists.

    On a grid of intersections many 4-point subsets are collinear, so RANSAC
    with few iterations can fail outright on a set that is perfectly
    well-conditioned as a whole.
    """
    h = _sample_h()
    src = [(x, y) for x in (0.0, 16.5, 52.5, 105.0) for y in (0.0, 13.84, 54.16, 68.0)]
    dst = [_apply(h, p) for p in src]
    assert kernels.fit_dlt(src, dst) is not None


def test_fit_dlt_validates_input():
    with pytest.raises(ValueError, match="same length"):
        kernels.fit_dlt([(0.0, 0.0)] * 4, [(0.0, 0.0)] * 5)
    with pytest.raises(ValueError, match="at least 4"):
        kernels.fit_dlt([(0.0, 0.0)] * 3, [(0.0, 0.0)] * 3)
