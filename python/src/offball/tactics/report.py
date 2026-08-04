"""Aggregate per-frame scores into a match report.

Aggregation choices that matter more than they look:

* **Medians, not means, for distance-like quantities.** A handful of frames
  where tracking put a player on the wrong side of the pitch will drag a mean
  arbitrarily far. The median shrugs them off.
* **Rates, not counts.** Players are on the pitch for different lengths of
  time and their team has the ball for different shares of it. A raw count of
  line-breaking positions mostly measures minutes played.
* **Sample size travels with every number.** ``frames`` is on every summary so
  a consumer can discard thin samples instead of over-reading them.

The natural-language layer that turns these into coaching points is out of
scope here; see ``docs/06-roadmap.md``.
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Iterable, Sequence

from ..types import Team
from .offball import FrameScore

__all__ = ["PlayerSummary", "TeamSummary", "MatchReport", "build_report"]


def _median(values: Sequence[float]) -> float:
    return statistics.median(values) if values else 0.0


def _mean(values: Sequence[float]) -> float:
    return statistics.fmean(values) if values else 0.0


@dataclass(frozen=True, slots=True)
class PlayerSummary:
    """One player's off-the-ball profile over the frames they were scored in."""

    track_id: int
    #: Frames this player was scored in — the sample size for every figure
    #: below. Treat anything under a few hundred as indicative only.
    frames: int
    #: Seconds of scored off-ball time.
    duration: float

    #: Median pitch area owned, m^2.
    median_space_owned: float
    #: Median threat value of the ground occupied, in [0, 1].
    median_position_value: float
    #: Share of frames offering the carrier a clear passing lane.
    availability_rate: float
    #: Share of frames in an offside position.
    offside_rate: float
    #: Median distance inside the offside line, in metres. Small negative
    #: values mean a player who consistently plays on the shoulder.
    median_offside_margin: float | None
    #: Mean number of opposition banks played beyond.
    mean_lines_broken: float
    #: Median nearest-opponent distance, in metres.
    median_separation: float | None
    #: Mean marking pressure in [0, 1].
    mean_pressure: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class TeamSummary:
    """Team-level shape and space over the frames it was in possession."""

    team: Team
    frames: int
    duration: float
    median_controlled_space: float
    median_dangerous_space: float
    median_attacking_hull: float
    median_defending_hull: float
    #: Mean count of teammates offering a viable pass at any moment. This is
    #: the single most useful team-level number the platform produces.
    mean_passing_options: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class MatchReport:
    frames_scored: int
    frames_total: int
    teams: tuple[TeamSummary, ...]
    players: tuple[PlayerSummary, ...]

    @property
    def coverage(self) -> float:
        """Share of frames that could be scored.

        The headline health metric for a run. Below roughly 0.6 the vision
        stage struggled and the tactical numbers should be treated as
        provisional — check calibration first.
        """
        return self.frames_scored / self.frames_total if self.frames_total else 0.0

    def player(self, track_id: int) -> PlayerSummary | None:
        return next((p for p in self.players if p.track_id == track_id), None)

    def to_dict(self) -> dict:
        return {
            "frames_scored": self.frames_scored,
            "frames_total": self.frames_total,
            "coverage": self.coverage,
            "teams": [t.to_dict() for t in self.teams],
            "players": [p.to_dict() for p in self.players],
        }


def build_report(
    scores: Iterable[FrameScore], frames_total: int, fps: float = 25.0
) -> MatchReport:
    """Aggregate frame scores into a match report.

    Args:
        scores: Per-frame results from :func:`offball.tactics.offball.score_frame`.
            Frames that could not be scored should simply be absent.
        frames_total: Frames in the source video, used for :attr:`coverage`.
        fps: Source frame rate, used to convert frame counts to seconds.
    """
    scores = list(scores)
    dt = 1.0 / fps if fps > 0 else 0.0

    # --- per player --------------------------------------------------------
    space: dict[int, list[float]] = defaultdict(list)
    values: dict[int, list[float]] = defaultdict(list)
    margins: dict[int, list[float]] = defaultdict(list)
    separations: dict[int, list[float]] = defaultdict(list)
    pressures: dict[int, list[float]] = defaultdict(list)
    lines: dict[int, list[int]] = defaultdict(list)
    available: dict[int, int] = defaultdict(int)
    offside: dict[int, int] = defaultdict(int)
    counts: dict[int, int] = defaultdict(int)

    for frame in scores:
        for p in frame.players:
            counts[p.track_id] += 1
            space[p.track_id].append(p.space_owned)
            values[p.track_id].append(p.position_value)
            pressures[p.track_id].append(p.marking_pressure)
            lines[p.track_id].append(p.lines_broken)
            if p.offside_margin is not None:
                margins[p.track_id].append(p.offside_margin)
            if p.nearest_opponent is not None:
                separations[p.track_id].append(p.nearest_opponent)
            if p.is_available:
                available[p.track_id] += 1
            if p.is_offside:
                offside[p.track_id] += 1

    players = tuple(
        PlayerSummary(
            track_id=tid,
            frames=n,
            duration=n * dt,
            median_space_owned=_median(space[tid]),
            median_position_value=_median(values[tid]),
            availability_rate=available[tid] / n,
            offside_rate=offside[tid] / n,
            median_offside_margin=_median(margins[tid]) if margins[tid] else None,
            mean_lines_broken=_mean(lines[tid]),
            median_separation=_median(separations[tid]) if separations[tid] else None,
            mean_pressure=_mean(pressures[tid]),
        )
        for tid, n in sorted(counts.items())
        if n > 0
    )

    # --- per team ----------------------------------------------------------
    by_team: dict[Team, list[FrameScore]] = defaultdict(list)
    for frame in scores:
        by_team[frame.attacking_team].append(frame)

    teams = tuple(
        TeamSummary(
            team=team,
            frames=len(frames),
            duration=len(frames) * dt,
            median_controlled_space=_median([f.team_space for f in frames]),
            median_dangerous_space=_median([f.team_dangerous_space for f in frames]),
            median_attacking_hull=_median([f.attacking_hull for f in frames]),
            median_defending_hull=_median([f.defending_hull for f in frames]),
            mean_passing_options=_mean([f.available_options for f in frames]),
        )
        for team, frames in sorted(by_team.items(), key=lambda kv: kv[0].value)
    )

    return MatchReport(
        frames_scored=len(scores), frames_total=frames_total, teams=teams, players=players
    )
