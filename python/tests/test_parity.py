"""Rust and pure-Python kernels must agree.

The pure-Python implementation is the specification. When the compiled
extension is installed, every kernel is evaluated both ways here and the
results compared. When it is not, the suite skips — CI builds the extension so
the comparison actually runs.

This is the test that makes the dual-backend design safe: without it, a
divergence would silently change match reports depending on whether a wheel
happened to be installed.
"""

from __future__ import annotations

import math

import pytest

from offball import kernels
from offball.kernels import ControlParams

pytestmark = pytest.mark.skipif(
    not kernels.using_rust(),
    reason="compiled kernels not installed; build with 'maturin develop --release'",
)

# Tolerances: both backends do the same f64 arithmetic in the same order, so
# agreement should be near-exact. These allow for the last bit or two.
ABS = 1e-9
REL = 1e-9


@pytest.fixture(scope="module")
def py():
    """An independent copy of the kernels module, pinned to the Python backend.

    ``importlib.reload`` will not do: it re-executes in the *same* module
    object, so clearing ``_rs`` there would also disable the compiled backend
    for :mod:`offball.kernels` and every comparison below would compare pure
    Python against itself. This loads a second module object from the same
    source file instead, leaving the real one untouched.
    """
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location(
        "offball._kernels_pyref", kernels.__file__
    )
    module = importlib.util.module_from_spec(spec)
    # Relative imports inside the module (``from .types import Point``) resolve
    # against this.
    module.__package__ = "offball"
    sys.modules["offball._kernels_pyref"] = module
    spec.loader.exec_module(module)
    module._rs = None
    try:
        yield module
    finally:
        sys.modules.pop("offball._kernels_pyref", None)


def test_the_two_backends_are_actually_distinct(py):
    """Guard against this whole suite silently becoming a no-op."""
    assert kernels is not py
    assert kernels._rs is not None, "the compiled backend must still be live"
    assert py._rs is None, "the reference copy must be on pure Python"
    assert py.using_rust() is False
    assert kernels.using_rust() is True


PLAYERS = (
    [(20.0, 20.0), (35.0, 40.0), (55.0, 34.0), (70.0, 15.0), (80.0, 50.0), (95.0, 34.0)],
    [(1.5, -0.5), (0.0, 2.0), (3.0, 0.0), (-2.0, 1.0), (0.5, -1.5), (0.0, 0.0)],
    [True, True, True, False, False, False],
)


def test_pitch_control_matches(py):
    pos, vel, att = PLAYERS
    a = kernels.pitch_control(pos, vel, att, nx=30, ny=20)
    b = py.pitch_control(pos, vel, att, nx=30, ny=20)
    assert (a.nx, a.ny) == (b.nx, b.ny)
    assert a.cell_area == pytest.approx(b.cell_area, rel=REL)
    for va, vb in zip(a.values, b.values):
        assert va == pytest.approx(vb, abs=ABS)


def test_pitch_control_with_custom_params_matches(py):
    pos, vel, att = PLAYERS
    params = ControlParams(reaction_time=0.4, max_speed=9.1, tau=0.2)
    a = kernels.pitch_control(pos, vel, att, nx=24, ny=16, params=params)
    b = py.pitch_control(pos, vel, att, nx=24, ny=16, params=params)
    for va, vb in zip(a.values, b.values):
        assert va == pytest.approx(vb, abs=ABS)


def test_degenerate_control_cases_match(py):
    for pos, vel, att in (
        ([], [], []),
        ([(50.0, 34.0)], [(0.0, 0.0)], [True]),
        ([(50.0, 34.0)], [(0.0, 0.0)], [False]),
    ):
        a = kernels.pitch_control(pos, vel, att, nx=8, ny=8)
        b = py.pitch_control(pos, vel, att, nx=8, ny=8)
        assert list(a.values) == pytest.approx(list(b.values), abs=ABS)


def test_space_ownership_matches(py):
    pos, vel, att = PLAYERS
    a = kernels.space_ownership(pos, vel, att, nx=40, ny=26)
    b = py.space_ownership(pos, vel, att, nx=40, ny=26)
    assert a == pytest.approx(b, abs=1e-7)


def test_dangerous_space_matches(py):
    pos, vel, att = PLAYERS
    grid = kernels.pitch_control(pos, vel, att, nx=30, ny=20)
    assert kernels.dangerous_space(grid) == pytest.approx(py.dangerous_space(grid), rel=REL)


def test_pitch_value_matches(py):
    for p in [(0.0, 0.0), (52.5, 34.0), (98.0, 34.0), (98.0, 2.0), (105.0, 68.0)]:
        assert kernels.pitch_value(p) == pytest.approx(py.pitch_value(p), abs=ABS)


def test_offside_line_matches(py):
    for defs, ball in (
        ([2.0, 40.0, 42.0, 45.0], 30.0),
        ([2.0, 40.0, 42.0, 45.0], 60.0),
        ([40.0], 10.0),
        ([], 10.0),
    ):
        assert kernels.offside_line(defs, ball) == py.offside_line(defs, ball)


def test_passing_lane_matches(py):
    cases = [
        ((0.0, 0.0), (20.0, 0.0), [(10.0, 0.5)]),
        ((0.0, 0.0), (20.0, 0.0), [(10.0, 6.0)]),
        ((10.0, 0.0), (30.0, 0.0), [(0.0, 0.0)]),
        ((0.0, 0.0), (10.0, 0.0), []),
        ((30.0, 20.0), (70.0, 50.0), [(50.0, 35.0), (45.0, 20.0), (60.0, 44.0)]),
    ]
    for ball, receiver, defenders in cases:
        a = kernels.passing_lane(ball, receiver, defenders)
        b = py.passing_lane(ball, receiver, defenders)
        assert a.open == b.open
        assert a.blockers == b.blockers
        if math.isinf(a.clearance):
            assert math.isinf(b.clearance)
        else:
            assert a.clearance == pytest.approx(b.clearance, abs=ABS)


def test_team_shape_matches(py):
    for pts in (
        [(40.0, 20.0), (45.0, 30.0), (42.0, 40.0), (48.0, 48.0)],
        [(10.0, 5.0), (50.0, 30.0), (90.0, 60.0), (70.0, 10.0)],
        [(1.0, 1.0), (2.0, 2.0), (3.0, 3.0)],  # collinear
        [],
    ):
        a = kernels.team_shape(pts)
        b = py.team_shape(pts)
        if a is None or b is None:
            assert a is None and b is None
            continue
        assert a.hull_area == pytest.approx(b.hull_area, abs=1e-7)
        assert a.depth == pytest.approx(b.depth, abs=ABS)
        assert a.width == pytest.approx(b.width, abs=ABS)
        assert a.centroid == pytest.approx(b.centroid, abs=ABS)


def test_defensive_lines_match(py):
    for xs, k in (
        ([29.0, 30.0, 31.0, 30.5, 49.0, 50.0, 51.0, 50.5, 70.0, 71.0], 3),
        ([30.0, 31.0, 50.0, 70.0], 2),
        ([30.0, 31.0, 50.0, 70.0], 99),
        ([], 3),
    ):
        assert kernels.defensive_lines(xs, k) == pytest.approx(py.defensive_lines(xs, k), abs=ABS)


def test_marking_pressure_matches(py):
    p = (50.0, 34.0)
    for opponents in ([(50.5, 34.0)], [(65.0, 34.0)], [(51.0, 35.0), (60.0, 20.0)], []):
        assert kernels.marking_pressure(p, opponents) == pytest.approx(
            py.marking_pressure(p, opponents), abs=ABS
        )


def test_homography_fit_and_projection_match(py):
    h = (1.2, 0.35, 40.0, -0.15, 0.9, 25.0, 0.0006, 0.0011, 1.0)

    def apply(p):
        w = h[6] * p[0] + h[7] * p[1] + h[8]
        return ((h[0] * p[0] + h[1] * p[1] + h[2]) / w, (h[3] * p[0] + h[4] * p[1] + h[5]) / w)

    src = [(0.0, 0.0), (105.0, 0.0), (105.0, 68.0), (0.0, 68.0), (52.5, 34.0), (16.5, 13.85)]
    dst = [apply(p) for p in src]

    ha, ia = kernels.fit_homography(src, dst, threshold=0.5, iterations=200, seed=11)
    hb, ib = py.fit_homography(src, dst, threshold=0.5, iterations=200, seed=11)
    assert ia == ib, "both backends must select the same consensus set"
    assert ha == pytest.approx(hb, abs=1e-7)

    pts = [(10.0, 10.0), (60.0, 40.0), (100.0, 60.0)]
    for a, b in zip(kernels.project(ha, pts), py.project(hb, pts)):
        assert a == pytest.approx(b, abs=1e-7)


def test_horizon_projection_matches(py):
    h = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, -5.0)
    assert kernels.project(h, [(5.0, 2.0)])[0] is None
    assert py.project(h, [(5.0, 2.0)])[0] is None


def test_fit_dlt_matches(py):
    h = (1.2, 0.35, 40.0, -0.15, 0.9, 25.0, 0.0006, 0.0011, 1.0)

    def apply(p):
        w = h[6] * p[0] + h[7] * p[1] + h[8]
        return ((h[0] * p[0] + h[1] * p[1] + h[2]) / w, (h[3] * p[0] + h[4] * p[1] + h[5]) / w)

    # A grid of intersections, the shape the line matcher actually produces.
    src = [(x, y) for x in (0.0, 16.5, 52.5, 105.0) for y in (0.0, 13.84, 54.16, 68.0)]
    dst = [apply(p) for p in src]
    assert kernels.fit_dlt(src, dst) == pytest.approx(py.fit_dlt(src, dst), abs=1e-7)


def test_fit_dlt_rejects_degenerate_input_in_both_backends(py):
    collinear_src = [(float(i), float(i)) for i in range(6)]
    collinear_dst = [(2.0 * i, 2.0 * i) for i in range(6)]
    for mod in (kernels, py):
        with pytest.raises(ValueError):
            mod.fit_dlt(collinear_src, collinear_dst)
        with pytest.raises(ValueError):
            mod.fit_dlt([(0.0, 0.0)] * 3, [(0.0, 0.0)] * 3)
        with pytest.raises(ValueError, match="same length"):
            mod.fit_dlt([(0.0, 0.0)] * 4, [(0.0, 0.0)] * 5)
