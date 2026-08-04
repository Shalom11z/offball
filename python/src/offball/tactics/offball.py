"""Off-the-ball positioning metrics for a single frame.

This is the analytical core of the platform. Everything upstream exists to
produce a :class:`~offball.types.FrameState`; everything downstream aggregates
what this module emits.

Design rules that the rest of the codebase relies on:

* **Direction is normalised before scoring.** Every metric assumes the team in
  possession attacks toward +x. :func:`score_frame` flips coordinates when it
  does not, so no metric needs to know which way the teams kicked off.
* **Missing data abstains.** A frame without a calibration, without a ball, or
  without two tracked defenders produces no score rather than a default one.
  Averaging a fabricated zero into a match report is worse than a smaller
  sample.
* **Nothing here is a judgement.** These are geometric measurements. Turning
  "held a 0.4m onside margin for 12 minutes" into "times runs well" is the
  reporting layer's job, and a coach's.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

from .. import kernels
from ..kernels import ControlParams
from ..types import FrameState, PlayerObservation, Point, Team

__all__ = [
    "OffBallScore",
    "FrameScore",
    "ScoringConfig",
    "score_frame",
]


@dataclass(frozen=True, slots=True)
class ScoringConfig:
    pitch_length: float = 105.0
    pitch_width: float = 68.0
    #: Control-grid resolution. 1m cells over a standard pitch. Halving this to
    #: 2m cells is ~4x faster and moves reported areas by under 2%.
    grid_nx: int = 105
    grid_ny: int = 68
    control: ControlParams = ControlParams()
    #: Interceptable reach either side of a passing lane, in metres.
    lane_corridor: float = 1.2
    #: Distance at which marking pressure decays to 1/e.
    pressure_scale: float = 5.0
    #: Number of banks to fit to the defensive block.
    defensive_banks: int = 3


@dataclass(frozen=True, slots=True)
class OffBallScore:
    """One attacking player's positioning, in one frame."""

    track_id: int
    position: Point

    #: Pitch area (m^2) this player is the fastest to reach.
    space_owned: float
    #: Metres beyond the offside line. Negative is onside. ``None`` when the
    #: line could not be established.
    offside_margin: float | None
    #: Whether the ball carrier has a clear lane to this player.
    lane_open: bool
    #: Distance from that lane to the nearest defender, in metres.
    lane_clearance: float
    #: How many opposition banks this player is positioned beyond.
    lines_broken: int
    #: Nearest-opponent pressure in [0, 1]; 1 is tightly marked.
    marking_pressure: float
    #: Distance to the nearest opponent, in metres. ``None`` if none tracked.
    nearest_opponent: float | None
    #: Threat value in [0, 1] of the ground this player is standing on.
    position_value: float

    @property
    def is_offside(self) -> bool:
        return self.offside_margin is not None and self.offside_margin > 0.0

    @property
    def is_available(self) -> bool:
        """A realistic passing option right now: open lane, onside, not smothered."""
        return self.lane_open and not self.is_offside and self.marking_pressure < 0.6


@dataclass(frozen=True, slots=True)
class FrameScore:
    """All off-the-ball measurements for one frame."""

    frame_index: int
    timestamp: float
    attacking_team: Team
    players: tuple[OffBallScore, ...]

    #: Attacking team's controlled area, m^2.
    team_space: float
    #: Controlled area weighted by threat value.
    team_dangerous_space: float
    #: Convex-hull area of the attacking block, m^2.
    attacking_hull: float
    #: Convex-hull area of the defending block, m^2.
    defending_hull: float
    #: Fitted x positions of the opposition banks.
    defensive_lines: tuple[float, ...]
    offside_line: float | None

    @property
    def available_options(self) -> int:
        """How many teammates the ball carrier could actually pass to."""
        return sum(1 for p in self.players if p.is_available)


def _flip(p: Point, length: float, width: float) -> Point:
    return (length - p[0], width - p[1])


def _flip_vel(v: Point) -> Point:
    return (-v[0], -v[1])


def _ball_carrier(
    ball: Point, players: Sequence[tuple[int, Point]]
) -> tuple[int, Point] | None:
    """Nearest attacker to the ball, treated as the carrier."""
    if not players:
        return None
    return min(players, key=lambda tp: (tp[1][0] - ball[0]) ** 2 + (tp[1][1] - ball[1]) ** 2)


def score_frame(
    state: FrameState, config: ScoringConfig | None = None
) -> FrameScore | None:
    """Score one frame's off-the-ball positioning.

    Returns ``None`` — deliberately, rather than a partially-filled result —
    when the frame cannot be scored honestly:

    * no team in possession,
    * no ball position,
    * fewer than two players on either side with pitch coordinates.

    Args:
        state: A calibrated frame. ``state.attacking_team`` sets the direction
            of play; see the module docstring.
        config: Scoring parameters.
    """
    config = config or ScoringConfig()

    if state.attacking_team is None or not state.attacking_team.is_player:
        return None
    if state.ball_pitch_xy is None:
        return None

    attackers = state.team(state.attacking_team)
    defenders = state.team(state.attacking_team.opponent())
    if len(attackers) < 2 or len(defenders) < 2:
        return None

    length, width = config.pitch_length, config.pitch_width

    # Normalise so the attacking team always plays toward +x. `attacks_positive`
    # would come from the match's period metadata in production; absent that,
    # infer it from which goal the defence is protecting.
    def_x = sum(d.pitch_xy[0] for d in defenders) / len(defenders)
    att_x = sum(a.pitch_xy[0] for a in attackers) / len(attackers)
    flip = def_x < att_x

    def pos(p: PlayerObservation) -> Point:
        return _flip(p.pitch_xy, length, width) if flip else p.pitch_xy

    def vel(p: PlayerObservation) -> Point:
        v = p.velocity or (0.0, 0.0)
        return _flip_vel(v) if flip else v

    ball = _flip(state.ball_pitch_xy, length, width) if flip else state.ball_pitch_xy

    att_pos = [pos(p) for p in attackers]
    att_vel = [vel(p) for p in attackers]
    def_pos = [pos(p) for p in defenders]
    def_vel = [vel(p) for p in defenders]

    # --- team-level fields -------------------------------------------------
    all_pos = att_pos + def_pos
    all_vel = att_vel + def_vel
    flags = [True] * len(att_pos) + [False] * len(def_pos)

    grid = kernels.pitch_control(
        all_pos, all_vel, flags, length, width, config.grid_nx, config.grid_ny, config.control
    )
    team_space = grid.total()
    team_dangerous = kernels.dangerous_space(grid, length, width)

    ownership = kernels.space_ownership(
        all_pos, all_vel, flags, length, width, config.grid_nx, config.grid_ny, config.control
    )

    att_shape = kernels.team_shape(att_pos)
    def_shape = kernels.team_shape(def_pos)
    lines = kernels.defensive_lines(
        [p[0] for p in def_pos], config.defensive_banks
    )
    line_x = kernels.offside_line([p[0] for p in def_pos], ball[0])

    # --- per-player --------------------------------------------------------
    carrier = _ball_carrier(ball, [(a.track_id, p) for a, p in zip(attackers, att_pos)])
    carrier_id = carrier[0] if carrier else None

    scores: list[OffBallScore] = []
    for i, (player, p) in enumerate(zip(attackers, att_pos)):
        # The carrier is on the ball by definition; they are not an off-ball
        # actor and including them skews every team aggregate.
        if player.track_id == carrier_id:
            continue

        lane = kernels.passing_lane(ball, p, def_pos, config.lane_corridor)
        scores.append(
            OffBallScore(
                track_id=player.track_id,
                position=p,
                space_owned=ownership[i],
                offside_margin=None if line_x is None else kernels.offside_margin(p[0], line_x),
                lane_open=lane.open,
                lane_clearance=lane.clearance,
                lines_broken=kernels.lines_broken(p[0], lines),
                marking_pressure=kernels.marking_pressure(p, def_pos, config.pressure_scale),
                nearest_opponent=kernels.nearest_opponent_distance(p, def_pos),
                position_value=kernels.pitch_value(p, length, width),
            )
        )

    return FrameScore(
        frame_index=state.frame_index,
        timestamp=state.timestamp,
        attacking_team=state.attacking_team,
        players=tuple(scores),
        team_space=team_space,
        team_dangerous_space=team_dangerous,
        attacking_hull=att_shape.hull_area if att_shape else 0.0,
        defending_hull=def_shape.hull_area if def_shape else 0.0,
        defensive_lines=tuple(lines),
        offside_line=line_x,
    )
