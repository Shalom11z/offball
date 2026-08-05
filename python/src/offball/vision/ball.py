"""Ball trajectory reconstruction from intermittent detections.

The ball is the hardest object in the frame to detect — often under 10 pixels
across, motion-blurred when struck, and hidden behind players for long stretches
— yet possession depends on it, and possession sets the direction every
off-the-ball metric is computed in. A detector that finds it in 60% of frames
would, taken naively, cost 40% of the analysis.

This module turns a sparse, noisy sequence of ball observations into a
continuous trajectory, and is the difference between a run that scores most
frames and one that scores almost none.

Three jobs, in order:

1. **Reject impossible observations.** A detection implying the ball moved at
   80 m/s is a white boot or a pitch marking, not the ball.
2. **Bridge gaps.** Interpolate across short occlusions. A ball behind a player
   for half a second has a well-determined position; a ball missing for ten
   seconds does not, and is left absent.
3. **Smooth.** Detection jitter of a few pixels becomes metres of apparent ball
   movement after projection, which upsets the possession assignment.

This runs as a pass over the whole sequence rather than per frame, because
interpolating a gap needs the observation *after* it as well as before.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

from ..types import Point

__all__ = [
    "BallConfig",
    "BallTracker",
    "select_ball_trajectory",
    "smooth_ball_track",
]


@dataclass(frozen=True, slots=True)
class BallConfig:
    #: Metres per second above which a step is treated as a mis-detection.
    #: A struck ball reaches ~35 m/s; a shot can exceed it briefly, so this is
    #: set well above typical play rather than at the physical limit.
    max_speed: float = 45.0
    #: Longest gap to interpolate across, in frames. At 25fps the default is
    #: two seconds — comfortably longer than a ball behind a player, shorter
    #: than a stoppage.
    max_gap: int = 50
    #: Window for the moving average, in frames. Must be odd.
    smoothing_window: int = 5
    #: Frames at the start or end of the sequence that may be filled by holding
    #: the nearest known position. Kept small: extrapolation is a guess.
    max_extrapolate: int = 5


def _distance(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _reject_outliers(
    track: list[Point | None], fps: float, config: BallConfig
) -> list[Point | None]:
    """Drop observations that imply an impossible ball speed.

    Anchored to the last *accepted* position rather than the previous
    observation, so a single wild detection cannot drag the anchor off and
    reject the good observations that follow it.
    """
    out: list[Point | None] = list(track)
    dt = 1.0 / fps if fps > 0 else 0.04

    anchor_i: int | None = None
    for i, p in enumerate(out):
        if p is None:
            continue
        if anchor_i is None:
            anchor_i = i
            continue
        elapsed = (i - anchor_i) * dt
        if elapsed <= 0:
            continue
        if _distance(p, out[anchor_i]) / elapsed > config.max_speed:
            out[i] = None  # implausible; treat as a miss
        else:
            anchor_i = i
    return out


def _interpolate(track: list[Point | None], config: BallConfig) -> list[Point | None]:
    """Fill short gaps by linear interpolation between known positions.

    Linear rather than ballistic: over the sub-second gaps worth filling, the
    difference is well under the noise in the detections themselves, and a
    ballistic fit needs a height estimate this pipeline does not have.
    """
    out: list[Point | None] = list(track)
    known = [i for i, p in enumerate(out) if p is not None]
    if not known:
        return out

    for start, end in itertools.pairwise(known):
        gap = end - start - 1
        if gap <= 0 or gap > config.max_gap:
            continue
        a, b = out[start], out[end]
        for step in range(1, gap + 1):
            t = step / (gap + 1)
            out[start + step] = (
                a[0] + (b[0] - a[0]) * t,
                a[1] + (b[1] - a[1]) * t,
            )

    # Leading and trailing gaps have only one neighbour, so hold the nearest
    # known position for a few frames rather than extrapolating a velocity.
    first, last = known[0], known[-1]
    for i in range(max(0, first - config.max_extrapolate), first):
        out[i] = out[first]
    for i in range(last + 1, min(len(out), last + 1 + config.max_extrapolate)):
        out[i] = out[last]
    return out


def _smooth(track: list[Point | None], config: BallConfig) -> list[Point | None]:
    """Centred moving average over contiguous runs of known positions.

    Averaging across a gap would pull positions toward wherever the ball was
    before an occlusion, so each run is smoothed independently.
    """
    window = max(1, config.smoothing_window | 1)
    if window == 1:
        return list(track)
    half = window // 2

    out: list[Point | None] = list(track)
    run_start: int | None = None
    for i in range(len(track) + 1):
        present = i < len(track) and track[i] is not None
        if present and run_start is None:
            run_start = i
        elif not present and run_start is not None:
            run = track[run_start:i]
            for j in range(len(run)):
                lo = max(0, j - half)
                hi = min(len(run), j + half + 1)
                window_pts = run[lo:hi]
                out[run_start + j] = (
                    sum(p[0] for p in window_pts) / len(window_pts),
                    sum(p[1] for p in window_pts) / len(window_pts),
                )
            run_start = None
    return out


def smooth_ball_track(
    track: list[Point | None], fps: float = 25.0, config: BallConfig | None = None
) -> list[Point | None]:
    """Reconstruct a continuous ball trajectory from sparse observations.

    Args:
        track: One entry per frame; ``None`` where the ball was not detected.
        fps: Frame rate, used to convert steps into speeds.
        config: Tunables.

    Returns:
        A list the same length as ``track``. Entries stay ``None`` where the
        gap was too long to bridge honestly — those frames go unscored rather
        than being filled with a guess.
    """
    config = config or BallConfig()
    if not track:
        return []
    stage = _reject_outliers(track, fps, config)
    stage = _interpolate(stage, config)
    return _smooth(stage, config)


class BallTracker:
    """Stateful wrapper for streaming use.

    :func:`smooth_ball_track` is the primary interface and is what
    :class:`~offball.pipeline.Pipeline` uses, because interpolation needs to see
    past the gap. This class exists for callers processing a live feed, where
    the future is not available: it can reject outliers and hold the last known
    position, but it cannot interpolate.
    """

    def __init__(self, fps: float = 25.0, config: BallConfig | None = None) -> None:
        self.config = config or BallConfig()
        self.fps = fps
        self._last: Point | None = None
        self._last_index: int | None = None
        self.rejected = 0

    def update(self, frame_index: int, observation: Point | None) -> Point | None:
        """Feed one frame; returns the best current estimate, or ``None``."""
        dt = 1.0 / self.fps if self.fps > 0 else 0.04

        if observation is not None and self._last is not None:
            elapsed = (frame_index - self._last_index) * dt
            if elapsed > 0 and _distance(observation, self._last) / elapsed > self.config.max_speed:
                self.rejected += 1
                observation = None

        if observation is not None:
            self._last = observation
            self._last_index = frame_index
            return observation

        # Hold the last known position through a short gap.
        if (
            self._last is not None
            and self._last_index is not None
            and frame_index - self._last_index <= self.config.max_gap
        ):
            return self._last
        return None

    def reset(self) -> None:
        self._last = None
        self._last_index = None
        self.rejected = 0


def select_ball_trajectory(
    candidates: list[list[tuple[Point, float]]],
    fps: float = 25.0,
    config: BallConfig | None = None,
) -> list[Point | None]:
    """Choose the most physically plausible ball path through per-frame candidates.

    Detecting the ball at a low confidence threshold roughly doubles recall, but
    the extra detections include white boots, the penalty spot and pitch
    markings. Choosing the highest-confidence candidate *per frame* makes that a
    net loss: a single false position is accepted, becomes the anchor for the
    speed check, and then rejects the true detections that follow it. Measured
    end to end, per-frame selection at a lower threshold reduced usable ball
    positions from 58% to 55% of frames despite raising raw recall from 32% to
    55%.

    Selecting a whole *trajectory* fixes that. This is a Viterbi pass over the
    candidate sets with an explicit "missing" state, so the path may skip frames
    where nothing plausible was detected rather than being forced onto a false
    positive.

    Args:
        candidates: Per frame, a list of ``(pitch_position, confidence)``.
        fps: Frame rate, for converting steps to speeds.
        config: Tunables; ``max_speed`` bounds a legal transition.

    Returns:
        One position per frame, or ``None`` where the path is genuinely absent.
    """
    config = config or BallConfig()
    n = len(candidates)
    if n == 0:
        return []
    dt = 1.0 / fps if fps > 0 else 0.04

    # Cost of emitting nothing. Set above the cost of a confident detection so
    # the path prefers real observations, but low enough that it will skip a
    # frame rather than accept an implausible jump.
    missing_cost = 1.2
    # Cost per metre of motion, which mildly prefers a smooth path among
    # several plausible candidates.
    smoothness = 0.05

    # cost[i][j]: best total cost of a path ending at frame i in candidate j,
    # where j == len(candidates[i]) denotes the missing state.
    costs: list[list[float]] = []
    back: list[list[int]] = []

    for i in range(n):
        options = candidates[i]
        width = len(options) + 1
        row = [math.inf] * width
        prev = [-1] * width

        for j in range(width):
            emit = missing_cost if j == len(options) else (1.0 - options[j][1])
            if i == 0:
                row[j] = emit
                continue

            best_cost, best_k = math.inf, -1
            for k in range(len(costs[i - 1])):
                if math.isinf(costs[i - 1][k]):
                    continue
                transition = 0.0
                # Both real: the step must be physically possible.
                if j < len(options) and k < len(candidates[i - 1]):
                    step = _distance(options[j][0], candidates[i - 1][k][0])
                    if step / dt > config.max_speed:
                        continue
                    transition = smoothness * step
                total = costs[i - 1][k] + transition
                if total < best_cost:
                    best_cost, best_k = total, k

            if best_k >= 0:
                row[j] = best_cost + emit
                prev[j] = best_k

        costs.append(row)
        back.append(prev)

    # Walk the cheapest path back.
    last = min(range(len(costs[-1])), key=lambda j: costs[-1][j])
    path = [last]
    for i in range(n - 1, 0, -1):
        last = back[i][last]
        if last < 0:
            last = len(candidates[i - 1])  # fall back to missing
        path.append(last)
    path.reverse()

    return [
        candidates[i][j][0] if j < len(candidates[i]) else None
        for i, j in enumerate(path)
    ]
