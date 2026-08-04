"""A synthetic match sequence, so the pipeline can be run end to end today.

There is no footage and no model here. It builds a scripted attacking move —
a back four holding a line, a midfield bank, and three forwards making runs
against it — and pushes it through the real tracker, the real calibration
stage, and the real metrics.

Its purpose is to make the whole system executable and its output inspectable
(``python -m offball.cli demo``) while the vision models are still being
trained. It is a scaffold, not a validation: numbers from this scene say
nothing about real-world accuracy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .types import BBox, Detection, FrameState, PlayerObservation, Point, Team

__all__ = ["synthetic_frames", "SyntheticMatch"]


def _bbox_at(x: float, y: float) -> BBox:
    """A plausible player box for a pitch position, under a fixed fake camera.

    The mapping is a crude linear one; the point is only that the tracker gets
    boxes that move coherently between frames.
    """
    px = 100.0 + x * 16.0
    py = 900.0 - y * 10.0
    return BBox(px - 12.0, py - 45.0, px + 12.0, py)


@dataclass(frozen=True, slots=True)
class SyntheticMatch:
    """Ground-truth pitch positions for a scripted attacking sequence."""

    frames: int = 120
    fps: float = 25.0

    def attackers(self, t: float) -> list[Point]:
        """Three forwards and two midfielders pushing up as the move develops."""
        return [
            (55.0 + 18.0 * t, 34.0),           # striker, running the line
            (48.0 + 22.0 * t, 12.0 + 4.0 * t), # right winger, hugging the touchline
            (48.0 + 20.0 * t, 56.0 - 4.0 * t), # left winger, cutting inside
            (40.0 + 12.0 * t, 26.0),           # attacking midfielder
            (34.0 + 8.0 * t, 42.0),            # deeper midfielder
        ]

    def defenders(self, t: float) -> list[Point]:
        """A back four dropping and a midfield two, holding shape."""
        return [
            (2.0, 34.0),                        # goalkeeper
            (74.0 - 14.0 * t, 20.0),            # right back
            (72.0 - 14.0 * t, 30.0),            # centre back
            (72.0 - 14.0 * t, 40.0),            # centre back
            (74.0 - 14.0 * t, 50.0),            # left back
            (84.0 - 12.0 * t, 28.0),            # holding midfielder
            (84.0 - 12.0 * t, 44.0),            # holding midfielder
        ]

    def ball(self, t: float) -> Point:
        """Carried by the deeper midfielder."""
        return (34.0 + 8.0 * t, 42.0)


def synthetic_frames(match: SyntheticMatch | None = None) -> list[FrameState]:
    """Build calibrated :class:`FrameState` objects for the scripted move.

    These bypass detection and tracking — they are already in pitch
    coordinates — so they exercise the tactics layer directly. For an
    end-to-end run through the tracker, see ``tests/test_pipeline.py``.
    """
    match = match or SyntheticMatch()
    states: list[FrameState] = []
    dt = 1.0 / match.fps

    prev_att: list[Point] | None = None
    prev_def: list[Point] | None = None

    for i in range(match.frames):
        t = i / max(match.frames - 1, 1)
        att = match.attackers(t)
        dfn = match.defenders(t)

        def velocity(cur: list[Point], prev: list[Point] | None, j: int) -> Point:
            if prev is None:
                return (0.0, 0.0)
            return ((cur[j][0] - prev[j][0]) / dt, (cur[j][1] - prev[j][1]) / dt)

        players: list[PlayerObservation] = []
        for j, p in enumerate(att):
            players.append(
                PlayerObservation(
                    track_id=100 + j,
                    bbox=_bbox_at(*p),
                    team=Team.HOME,
                    pitch_xy=p,
                    velocity=velocity(att, prev_att, j),
                )
            )
        for j, p in enumerate(dfn):
            players.append(
                PlayerObservation(
                    track_id=200 + j,
                    bbox=_bbox_at(*p),
                    team=Team.AWAY,
                    pitch_xy=p,
                    velocity=velocity(dfn, prev_def, j),
                )
            )

        states.append(
            FrameState(
                frame_index=i,
                timestamp=i * dt,
                players=tuple(players),
                ball_pitch_xy=match.ball(t),
                homography=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
                attacking_team=Team.HOME,
            )
        )
        prev_att, prev_def = att, dfn

    return states
