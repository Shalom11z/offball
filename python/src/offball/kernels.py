"""Numeric kernels, dispatched to Rust when available.

The Rust extension (``offball_core``, built from ``rust/offball-core``) is
roughly two orders of magnitude faster on the pitch-control grid and is what
production runs use. It is *not* required: every kernel has a pure-Python
reference implementation below with identical semantics, so the package
installs and runs anywhere, and CI checks the two agree.

Use :data:`BACKEND` to see which is active.

The pure-Python versions are the specification; if the two ever disagree, the
Rust one is wrong. ``tests/test_parity.py`` enforces this.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from .types import Point

__all__ = [
    "BACKEND",
    "ControlGrid",
    "ControlParams",
    "LaneVerdict",
    "TeamShape",
    "dangerous_space",
    "defensive_lines",
    "fit_homography",
    "marking_pressure",
    "offside_line",
    "passing_lane",
    "pitch_control",
    "pitch_value",
    "project",
    "space_ownership",
    "team_shape",
    "using_rust",
]

try:  # pragma: no cover - exercised by whichever branch the environment takes
    import offball_core as _rs

    BACKEND: Literal["rust", "python"] = "rust"
except ImportError:  # pragma: no cover
    _rs = None
    BACKEND = "python"


def using_rust() -> bool:
    """True when the compiled kernels are in use."""
    return _rs is not None


# --------------------------------------------------------------------------
# Parameters and result types
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ControlParams:
    """Tunables for the time-to-intercept control model.

    Defaults are league-average figures from the tracking literature rather
    than values fitted to any one competition. See
    ``docs/03-tactical-metrics.md``.
    """

    reaction_time: float = 0.7
    max_speed: float = 7.8
    tau: float = 0.45

    def time_to_intercept(self, pos: Point, vel: Point, target: Point) -> float:
        """Reaction-then-sprint arrival time, in seconds."""
        dx = pos[0] + vel[0] * self.reaction_time - target[0]
        dy = pos[1] + vel[1] * self.reaction_time - target[1]
        return self.reaction_time + math.hypot(dx, dy) / max(self.max_speed, 1e-6)


@dataclass(frozen=True, slots=True)
class ControlGrid:
    """A scalar field over the pitch, row-major, ``ny`` rows of ``nx``."""

    values: tuple[float, ...]
    nx: int
    ny: int
    cell_area: float

    def at(self, ix: int, iy: int) -> float:
        return self.values[iy * self.nx + ix]

    def total(self) -> float:
        """Integral over the pitch; m^2 when values are probabilities."""
        return sum(self.values) * self.cell_area


@dataclass(frozen=True, slots=True)
class LaneVerdict:
    open: bool
    clearance: float
    blockers: int


@dataclass(frozen=True, slots=True)
class TeamShape:
    hull_area: float
    depth: float
    width: float
    centroid: Point


# --------------------------------------------------------------------------
# Pure-Python reference geometry
# --------------------------------------------------------------------------


def _cell_centre(ix: int, iy: int, nx: int, ny: int, length: float, width: float) -> Point:
    return ((ix + 0.5) * length / nx, (iy + 0.5) * width / ny)


def _point_segment_distance(p: Point, a: Point, b: Point) -> float:
    abx, aby = b[0] - a[0], b[1] - a[1]
    len2 = abx * abx + aby * aby
    if len2 <= 1e-18:
        return math.hypot(p[0] - a[0], p[1] - a[1])
    t = ((p[0] - a[0]) * abx + (p[1] - a[1]) * aby) / len2
    t = min(1.0, max(0.0, t))
    return math.hypot(p[0] - (a[0] + abx * t), p[1] - (a[1] + aby * t))


def _convex_hull(points: Sequence[Point]) -> list[Point]:
    """Andrew's monotone chain, counter-clockwise."""
    pts = sorted(set(points))
    if len(pts) < 3:
        return pts

    def half(seq: Sequence[Point]) -> list[Point]:
        chain: list[Point] = []
        for p in seq:
            while len(chain) >= 2:
                (ox, oy), (ax, ay) = chain[-2], chain[-1]
                if (ax - ox) * (p[1] - oy) - (ay - oy) * (p[0] - ox) <= 0:
                    chain.pop()
                else:
                    break
            chain.append(p)
        return chain[:-1]

    return half(pts) + half(pts[::-1])


def _polygon_area(poly: Sequence[Point]) -> float:
    if len(poly) < 3:
        return 0.0
    acc = 0.0
    for i, (x1, y1) in enumerate(poly):
        x2, y2 = poly[(i + 1) % len(poly)]
        acc += x1 * y2 - y1 * x2
    return abs(acc) / 2.0


def _hull_area(points: Sequence[Point]) -> float:
    return _polygon_area(_convex_hull(points))


def _solve_linear(a: list[list[float]], b: list[float]) -> list[float] | None:
    """Gaussian elimination with partial pivoting. Returns None if singular."""
    n = len(b)
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(a[r][col]))
        if abs(a[pivot][col]) < 1e-12:
            return None
        a[col], a[pivot] = a[pivot], a[col]
        b[col], b[pivot] = b[pivot], b[col]
        d = a[col][col]
        for row in range(col + 1, n):
            f = a[row][col] / d
            if f == 0.0:
                continue
            for k in range(col, n):
                a[row][k] -= f * a[col][k]
            b[row] -= f * b[col]
    x = [0.0] * n
    for row in range(n - 1, -1, -1):
        acc = b[row] - sum(a[row][k] * x[k] for k in range(row + 1, n))
        x[row] = acc / a[row][row]
    return x


def _normalise(pts: Sequence[Point]) -> tuple[list[Point], tuple[float, ...]]:
    """Hartley normalisation; returns the points and the 3x3 transform."""
    n = len(pts)
    cx = sum(p[0] for p in pts) / n
    cy = sum(p[1] for p in pts) / n
    mean_d = sum(math.hypot(p[0] - cx, p[1] - cy) for p in pts) / n
    s = math.sqrt(2.0) / mean_d if mean_d > 1e-12 else 1.0
    t = (s, 0.0, -s * cx, 0.0, s, -s * cy, 0.0, 0.0, 1.0)
    return [(s * (p[0] - cx), s * (p[1] - cy)) for p in pts], t


def _mat_mul3(a: Sequence[float], b: Sequence[float]) -> tuple[float, ...]:
    return tuple(
        sum(a[r * 3 + k] * b[k * 3 + c] for k in range(3)) for r in range(3) for c in range(3)
    )


def _mat_inv3(m: Sequence[float]) -> tuple[float, ...] | None:
    c = (
        m[4] * m[8] - m[5] * m[7],
        m[2] * m[7] - m[1] * m[8],
        m[1] * m[5] - m[2] * m[4],
        m[5] * m[6] - m[3] * m[8],
        m[0] * m[8] - m[2] * m[6],
        m[2] * m[3] - m[0] * m[5],
        m[3] * m[7] - m[4] * m[6],
        m[1] * m[6] - m[0] * m[7],
        m[0] * m[4] - m[1] * m[3],
    )
    det = m[0] * c[0] + m[1] * c[3] + m[2] * c[6]
    if abs(det) < 1e-12:
        return None
    return tuple(v / det for v in c)


def _fit_dlt(src: Sequence[Point], dst: Sequence[Point]) -> tuple[float, ...] | None:
    if len(src) < 4 or len(src) != len(dst):
        return None
    ns, ts = _normalise(src)
    nd, td = _normalise(dst)

    ata = [[0.0] * 8 for _ in range(8)]
    atb = [0.0] * 8
    for (x, y), (u, v) in zip(ns, nd, strict=True):
        for row, rhs in (
            ([x, y, 1.0, 0.0, 0.0, 0.0, -u * x, -u * y], u),
            ([0.0, 0.0, 0.0, x, y, 1.0, -v * x, -v * y], v),
        ):
            for i in range(8):
                if row[i] == 0.0:
                    continue
                for j in range(8):
                    ata[i][j] += row[i] * row[j]
                atb[i] += row[i] * rhs

    h = _solve_linear(ata, atb)
    if h is None:
        return None
    hn = (*h, 1.0)
    td_inv = _mat_inv3(td)
    if td_inv is None:
        return None
    return _mat_mul3(td_inv, _mat_mul3(hn, ts))


def _apply(h: Sequence[float], p: Point) -> Point | None:
    w = h[6] * p[0] + h[7] * p[1] + h[8]
    if abs(w) < 1e-12:
        return None
    return ((h[0] * p[0] + h[1] * p[1] + h[2]) / w, (h[3] * p[0] + h[4] * p[1] + h[5]) / w)


# --------------------------------------------------------------------------
# Public kernels
# --------------------------------------------------------------------------


def fit_homography(
    src: Sequence[Point],
    dst: Sequence[Point],
    threshold: float = 1.5,
    iterations: int = 1000,
    seed: int = 0,
) -> tuple[tuple[float, ...], list[int]]:
    """Fit a ``src -> dst`` homography with deterministic RANSAC.

    Returns ``(h, inlier_indices)`` where ``h`` is row-major 3x3, flattened.

    Raises:
        ValueError: fewer than 4 correspondences, mismatched lengths, or no
            consensus set could be found (typically a degenerate, collinear
            keypoint set).
    """
    if len(src) != len(dst):
        raise ValueError("src and dst must be the same length")
    if len(src) < 4:
        raise ValueError("need at least 4 correspondences to fit a homography")

    if _rs is not None:
        h, inliers = _rs.fit_homography(
            [tuple(p) for p in src], [tuple(p) for p in dst], threshold, iterations, seed
        )
        return tuple(h), list(inliers)

    n = len(src)
    if n == 4:
        h = _fit_dlt(src, dst)
        if h is None:
            raise ValueError("homography fit failed: correspondences are degenerate")
        return h, [0, 1, 2, 3]

    # xorshift64, mirroring the Rust sampler so both backends explore the same
    # subsets for a given seed.
    state = seed | 1

    def rand() -> int:
        nonlocal state
        state ^= (state << 13) & 0xFFFFFFFFFFFFFFFF
        state ^= state >> 7
        state ^= (state << 17) & 0xFFFFFFFFFFFFFFFF
        return state

    best: list[int] = []
    for _ in range(iterations):
        idx: list[int] = []
        guard = 0
        while len(idx) < 4 and guard < 64:
            guard += 1
            c = rand() % n
            if c not in idx:
                idx.append(c)
        if len(idx) < 4:
            continue
        h = _fit_dlt([src[i] for i in idx], [dst[i] for i in idx])
        if h is None:
            continue
        inliers = []
        for i in range(n):
            p = _apply(h, src[i])
            if p is not None and math.hypot(p[0] - dst[i][0], p[1] - dst[i][1]) <= threshold:
                inliers.append(i)
        if len(inliers) > len(best):
            best = inliers

    if len(best) < 4:
        raise ValueError("homography fit failed: no consensus set of 4+ inliers")
    refined = _fit_dlt([src[i] for i in best], [dst[i] for i in best])
    if refined is None:
        raise ValueError("homography refit on consensus set failed")
    return refined, best


def project(h: Sequence[float], pts: Sequence[Point]) -> list[Point | None]:
    """Map points through a flat 9-element homography.

    Points that land on the horizon come back as ``None`` — they correspond to
    no finite pitch location and must not be treated as position (0, 0).
    """
    if len(h) != 9:
        raise ValueError("homography must have 9 elements")
    if _rs is not None:
        projected = _rs.project(list(h), [tuple(p) for p in pts])
        return [None if p is None else (p[0], p[1]) for p in projected]
    return [_apply(h, p) for p in pts]


def pitch_control(
    positions: Sequence[Point],
    velocities: Sequence[Point],
    is_attacking: Sequence[bool],
    pitch_length: float = 105.0,
    pitch_width: float = 68.0,
    nx: int = 105,
    ny: int = 68,
    params: ControlParams | None = None,
) -> ControlGrid:
    """Probability the attacking team controls each grid cell.

    An empty player list yields a uniform 0.5 field; a one-sided list yields a
    saturated one.
    """
    params = params or ControlParams()
    if nx <= 0 or ny <= 0:
        raise ValueError("grid dimensions must be positive")
    if not (len(positions) == len(velocities) == len(is_attacking)):
        raise ValueError("positions, velocities and is_attacking must be the same length")

    if _rs is not None:
        values, gx, gy, cell_area = _rs.pitch_control(
            [tuple(p) for p in positions],
            [tuple(v) for v in velocities],
            list(is_attacking),
            pitch_length,
            pitch_width,
            nx,
            ny,
            params.reaction_time,
            params.max_speed,
            params.tau,
        )
        return ControlGrid(tuple(values), gx, gy, cell_area)

    cell_area = (pitch_length * pitch_width) / (nx * ny)
    values = [0.5] * (nx * ny)
    for iy in range(ny):
        for ix in range(nx):
            q = _cell_centre(ix, iy, nx, ny, pitch_length, pitch_width)
            t_att = math.inf
            t_def = math.inf
            for pos, vel, att in zip(positions, velocities, is_attacking, strict=True):
                t = params.time_to_intercept(pos, vel, q)
                if att:
                    t_att = min(t_att, t)
                else:
                    t_def = min(t_def, t)
            if math.isinf(t_att) and math.isinf(t_def):
                v = 0.5
            elif math.isinf(t_def):
                v = 1.0
            elif math.isinf(t_att):
                v = 0.0
            else:
                v = 1.0 / (1.0 + math.exp((t_att - t_def) / params.tau))
            values[iy * nx + ix] = v
    return ControlGrid(tuple(values), nx, ny, cell_area)


def space_ownership(
    positions: Sequence[Point],
    velocities: Sequence[Point],
    is_attacking: Sequence[bool],
    pitch_length: float = 105.0,
    pitch_width: float = 68.0,
    nx: int = 105,
    ny: int = 68,
    params: ControlParams | None = None,
) -> list[float]:
    """Area in m^2 that each player is fastest to reach.

    A velocity-aware Voronoi partition: the returned areas sum to the pitch
    area, and index ``i`` corresponds to ``positions[i]``.
    """
    params = params or ControlParams()
    if nx <= 0 or ny <= 0:
        raise ValueError("grid dimensions must be positive")
    if not positions:
        return []

    if _rs is not None:
        return list(
            _rs.space_ownership(
                [tuple(p) for p in positions],
                [tuple(v) for v in velocities],
                list(is_attacking),
                pitch_length,
                pitch_width,
                nx,
                ny,
                params.reaction_time,
                params.max_speed,
                params.tau,
            )
        )

    cell_area = (pitch_length * pitch_width) / (nx * ny)
    owned = [0.0] * len(positions)
    for iy in range(ny):
        for ix in range(nx):
            q = _cell_centre(ix, iy, nx, ny, pitch_length, pitch_width)
            best_i, best_t = 0, math.inf
            for i, (pos, vel) in enumerate(zip(positions, velocities, strict=True)):
                t = params.time_to_intercept(pos, vel, q)
                if t < best_t:
                    best_t, best_i = t, i
            owned[best_i] += cell_area
    return owned


def pitch_value(p: Point, pitch_length: float = 105.0, pitch_width: float = 68.0) -> float:
    """Analytic threat surface in [0, 1] for a team attacking +x.

    Combines proximity to goal with the angle subtended by the goal mouth.
    This is the crudest component of the model and the one most worth replacing
    with an expected-threat surface fitted to real shot data — see
    ``docs/06-roadmap.md``.
    """
    goal = (pitch_length, pitch_width / 2.0)
    half_goal = 7.32 / 2.0
    ax, ay = goal[0] - p[0], (goal[1] - half_goal) - p[1]
    bx, by = goal[0] - p[0], (goal[1] + half_goal) - p[1]

    d = math.hypot(goal[0] - p[0], goal[1] - p[1])
    dist_term = math.exp(-d / 25.0)
    angle = abs(math.atan2(abs(ax * by - ay * bx), ax * bx + ay * by))
    angle_term = angle / math.pi
    return min(1.0, max(0.0, dist_term * 0.65 + angle_term * 0.35))


def dangerous_space(
    grid: ControlGrid, pitch_length: float = 105.0, pitch_width: float = 68.0
) -> float:
    """Controlled area weighted by how much that area is worth.

    The headline off-the-ball figure: space in your own corner scores close to
    nothing, space between the centre-backs scores heavily.
    """
    if _rs is not None:
        return float(
            _rs.dangerous_space(
                list(grid.values), grid.nx, grid.ny, grid.cell_area, pitch_length, pitch_width
            )
        )
    acc = 0.0
    for iy in range(grid.ny):
        for ix in range(grid.nx):
            q = _cell_centre(ix, iy, grid.nx, grid.ny, pitch_length, pitch_width)
            acc += grid.at(ix, iy) * pitch_value(q, pitch_length, pitch_width)
    return acc * grid.cell_area


def offside_line(defenders_x: Sequence[float], ball_x: float) -> float | None:
    """Offside line for a team attacking +x, per Law 11.

    The greater of the second-last defender's ``x`` and the ball's ``x``.
    ``defenders_x`` must include the goalkeeper. Returns ``None`` with fewer
    than two defenders tracked — a vision failure, which callers must treat as
    "cannot score this frame" rather than substituting a default.
    """
    if _rs is not None:
        return _rs.offside_line(list(defenders_x), ball_x)
    if len(defenders_x) < 2:
        return None
    return max(sorted(defenders_x, reverse=True)[1], ball_x)


def offside_margin(attacker_x: float, line_x: float) -> float:
    """Signed metres beyond the offside line; positive means offside."""
    return attacker_x - line_x


def passing_lane(
    ball: Point, receiver: Point, defenders: Sequence[Point], corridor: float = 1.2
) -> LaneVerdict:
    """Whether the straight ball -> receiver lane is free of defenders.

    ``corridor`` is the interceptable reach in metres. Defenders behind the
    passer do not block, which is why this uses segment rather than line
    distance.
    """
    if _rs is not None:
        is_open, clearance, blockers = _rs.passing_lane(
            tuple(ball), tuple(receiver), [tuple(d) for d in defenders], corridor
        )
        return LaneVerdict(bool(is_open), float(clearance), int(blockers))

    clearance = math.inf
    blockers = 0
    for d in defenders:
        dist = _point_segment_distance(d, ball, receiver)
        clearance = min(clearance, dist)
        if dist <= corridor:
            blockers += 1
    return LaneVerdict(blockers == 0, clearance, blockers)


def team_shape(players: Sequence[Point]) -> TeamShape | None:
    """Convex-hull area, depth, width and centroid of a set of players.

    Exclude the goalkeeper before calling: including them inflates depth by
    20-30m and makes the number meaningless.
    """
    if not players:
        return None
    if _rs is not None:
        res = _rs.team_shape([tuple(p) for p in players])
        if res is None:
            return None
        area, depth, width, cx, cy = res
        return TeamShape(area, depth, width, (cx, cy))

    xs = [p[0] for p in players]
    ys = [p[1] for p in players]
    return TeamShape(
        hull_area=_hull_area(players),
        depth=max(xs) - min(xs),
        width=max(ys) - min(ys),
        centroid=(sum(xs) / len(xs), sum(ys) / len(ys)),
    )


def defensive_lines(defenders_x: Sequence[float], k: int = 3, iterations: int = 50) -> list[float]:
    """Group defenders into banks by 1-D k-means on ``x``, sorted ascending.

    Quantile-initialised rather than random, so match reports are reproducible.
    """
    if _rs is not None:
        return list(_rs.defensive_lines(list(defenders_x), k, iterations))
    if not defenders_x or k <= 0:
        return []
    k = min(k, len(defenders_x))
    xs = sorted(defenders_x)
    centres = [xs[int((i + 0.5) / k * len(xs)) % len(xs)] for i in range(k)]

    for _ in range(iterations):
        sums = [0.0] * k
        counts = [0] * k
        for x in xs:
            best = min(range(k), key=lambda i: abs(x - centres[i]))
            sums[best] += x
            counts[best] += 1
        moved = False
        for i in range(k):
            if counts[i]:
                nc = sums[i] / counts[i]
                if abs(nc - centres[i]) > 1e-9:
                    moved = True
                centres[i] = nc
        if not moved:
            break
    return sorted(centres)


def lines_broken(attacker_x: float, line_centres: Sequence[float]) -> int:
    """How many opposition banks the attacker has positioned themselves beyond."""
    return sum(1 for c in line_centres if attacker_x > c)


def marking_pressure(player: Point, opponents: Sequence[Point], scale: float = 5.0) -> float:
    """Pressure in [0, 1]; 1 means an opponent is on top of the player.

    ``scale`` is the distance at which pressure decays to 1/e. About 5m matches
    the range over which a defender can contest a first touch.
    """
    if _rs is not None:
        return float(_rs.marking_pressure(tuple(player), [tuple(o) for o in opponents], scale))
    if not opponents:
        return 0.0
    nearest = min(math.hypot(o[0] - player[0], o[1] - player[1]) for o in opponents)
    return min(1.0, max(0.0, math.exp(-nearest / max(scale, 1e-6))))


def nearest_opponent_distance(player: Point, opponents: Sequence[Point]) -> float | None:
    """Distance to the closest opponent, or ``None`` if none are tracked."""
    if not opponents:
        return None
    return min(math.hypot(o[0] - player[0], o[1] - player[1]) for o in opponents)
