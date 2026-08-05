"""End-to-end orchestration: frames in, match report out.

Stage order, and why:

Per frame (:meth:`Pipeline.process_frame`):

1. **Detect** — players and ball.
2. **Track** — stable identities. Must precede team assignment, because the
   per-track colour vote is what makes assignment stable.
3. **Calibrate** — fit and temporally gate the frame homography.
4. **Project** — image ground points to pitch metres.
5. **Estimate velocity** — finite differences in *pitch* space, smoothed.
   Doing this in pitch space rather than pixel space matters: a pixel
   displacement means a different real distance at each end of the pitch.

Then over the whole sequence (:meth:`Pipeline.run`):

6. **Rejoin track fragments** — link tracklets that are the same player,
   using pitch-space motion. Without this each fragment carries a sample too
   small to aggregate, and per-player figures describe glimpses.
7. **Reconstruct the ball** — reject impossible jumps, interpolate across
   occlusions. This has to look forward as well as back, which is why it is a
   pass rather than a per-frame step, and it is what stops a detector's missed
   ball frames from silently discarding most of the match.
8. **Assign possession** — nearest player to the repaired ball, with
   hysteresis.
9. **Score** — off-the-ball metrics per frame.
10. **Aggregate** — the match report.

Parallelising is a scaling concern handled at the job level (one worker per
match), not inside the per-frame loop, where the ordering dependencies above
make it awkward and the Rust kernels already carry the numeric load.
"""

from __future__ import annotations

import math
from collections import defaultdict, deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field, replace

from .tactics.offball import FrameScore, ScoringConfig, score_frame
from .tactics.report import MatchReport, build_report
from .types import Detection, FrameState, PlayerObservation, Point, Team
from .vision.ball import BallConfig, smooth_ball_track
from .vision.calibration import (
    CalibrationConfig,
    HomographySmoother,
    calibrate_frame,
)
from .vision.stitch import StitchConfig, apply_stitching, stitch_tracks
from .vision.tracking import Tracker, TrackerConfig

__all__ = ["Pipeline", "PipelineConfig", "PipelineResult", "PossessionTracker"]


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    fps: float = 25.0
    pitch_length: float = 105.0
    pitch_width: float = 68.0
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    #: Ball trajectory reconstruction: outlier rejection and gap interpolation.
    ball: BallConfig = field(default_factory=BallConfig)
    #: Rejoining track fragments of the same player. Set to None to disable.
    stitch: StitchConfig | None = field(default_factory=StitchConfig)
    #: Frames of position history used for the velocity estimate. Five frames
    #: at 25fps is 0.2s: long enough to suppress tracking jitter, short enough
    #: to still register a sharp change of direction.
    velocity_window: int = 5
    #: A player must be this much closer to the ball than the current holder
    #: before possession switches, in metres. Without this margin, possession
    #: flickers between two players in a duel many times a second.
    possession_hysteresis: float = 1.5
    #: Beyond this distance from every player, the ball is loose and no team is
    #: credited with possession.
    possession_max_distance: float = 4.0


class PossessionTracker:
    """Assigns possession to a team, with hysteresis to stop it flickering.

    The naive "nearest player owns the ball" rule switches possession dozens of
    times during a single 50-50, and each switch flips the direction every
    off-ball metric is computed in. The margin and the loose-ball cutoff
    together make possession stable enough to score against.
    """

    def __init__(self, hysteresis: float = 1.5, max_distance: float = 4.0) -> None:
        self.hysteresis = hysteresis
        self.max_distance = max_distance
        self._current: Team | None = None

    @property
    def current(self) -> Team | None:
        return self._current

    def update(
        self, ball: Point | None, players: Sequence[PlayerObservation]
    ) -> Team | None:
        if ball is None:
            return self._current

        best: dict[Team, float] = {}
        for p in players:
            if p.pitch_xy is None or not p.team.is_player:
                continue
            d = math.hypot(p.pitch_xy[0] - ball[0], p.pitch_xy[1] - ball[1])
            if d < best.get(p.team, math.inf):
                best[p.team] = d
        if not best:
            return self._current

        nearest_team = min(best, key=best.__getitem__)
        nearest_d = best[nearest_team]

        if nearest_d > self.max_distance:
            # Ball is loose: hold the previous state rather than clearing it,
            # so a long pass between teammates does not blank out possession
            # for its entire flight.
            return self._current

        if self._current is None or nearest_team is self._current:
            self._current = nearest_team
            return self._current

        # Switching sides requires beating the incumbent by the margin.
        incumbent_d = best.get(self._current, math.inf)
        if nearest_d + self.hysteresis < incumbent_d:
            self._current = nearest_team
        return self._current

    def reset(self) -> None:
        self._current = None


class _VelocityEstimator:
    """Finite-difference velocity in pitch metres per second, per track."""

    def __init__(self, window: int, fps: float) -> None:
        self._window = max(2, window)
        self._dt = 1.0 / fps if fps > 0 else 0.04
        self._history: dict[int, deque[tuple[int, Point]]] = defaultdict(
            lambda: deque(maxlen=self._window)
        )

    def update(self, track_id: int, frame_index: int, xy: Point) -> Point:
        hist = self._history[track_id]
        hist.append((frame_index, xy))
        if len(hist) < 2:
            return (0.0, 0.0)
        (f0, p0), (f1, p1) = hist[0], hist[-1]
        span = (f1 - f0) * self._dt
        if span <= 0:
            return (0.0, 0.0)
        return ((p1[0] - p0[0]) / span, (p1[1] - p0[1]) / span)

    def reset(self) -> None:
        self._history.clear()


@dataclass(frozen=True, slots=True)
class PipelineResult:
    report: MatchReport
    frames: tuple[FrameState, ...]
    scores: tuple[FrameScore, ...]
    #: Frames for which a usable homography was available.
    calibrated_frames: int
    #: Calibrations rejected by the temporal gate.
    rejected_calibrations: int
    #: Frames in which the ball was actually detected.
    ball_detected_frames: int = 0
    #: Frames with a ball position after gap interpolation. The difference
    #: against `ball_detected_frames` is what the reconstruction recovered.
    ball_recovered_frames: int = 0
    #: Distinct track ids before fragments were rejoined.
    raw_tracks: int = 0
    #: Distinct track ids after. Compare against the 22 players actually on the
    #: pitch to judge how fragmented the tracking was.
    stitched_tracks: int = 0


class Pipeline:
    """Wires the stages together.

    The vision components are injected rather than constructed here, which is
    what lets the whole pipeline run under test with scripted detections and
    keypoints — no weights, no video, no GPU. See ``tests/test_pipeline.py``.

    Args:
        detector: Anything satisfying :class:`~offball.vision.detection.Detector`.
        keypoints: A :class:`~offball.vision.calibration.KeypointSource`.
        team_assigner: Optional; when omitted, observations keep whatever team
            they already carry. Scripted tests set teams directly.
        config: Pipeline settings.
    """

    def __init__(
        self,
        detector,
        keypoints,
        team_assigner=None,
        config: PipelineConfig | None = None,
    ) -> None:
        self.config = config or PipelineConfig()
        self.detector = detector
        self.keypoints = keypoints
        self.team_assigner = team_assigner
        self.tracker = Tracker(self.config.tracker)
        self.smoother = HomographySmoother(self.config.calibration)
        self.possession = PossessionTracker(
            self.config.possession_hysteresis, self.config.possession_max_distance
        )
        self._velocity = _VelocityEstimator(self.config.velocity_window, self.config.fps)

    def reset(self) -> None:
        self.tracker.reset()
        self.smoother.reset()
        self.possession.reset()
        self._velocity.reset()

    def process_frame(self, frame, frame_index: int) -> FrameState:
        """Run one frame through detection, tracking, calibration and projection."""
        cfg = self.config
        timestamp = frame_index / cfg.fps if cfg.fps > 0 else 0.0

        detections: list[Detection] = list(self.detector.detect(frame))
        observations = self.tracker.step(detections)

        if self.team_assigner is not None and frame is not None:
            observations = self.team_assigner.assign(frame, observations)

        calibration = self.smoother.push(
            calibrate_frame(self.keypoints.keypoints(frame), cfg.calibration)
        )

        ball_xy: Point | None = None
        if calibration is not None:
            image_points = [o.bbox.ground_point for o in observations]
            projected = calibration.to_pitch(image_points)

            placed: list[PlayerObservation] = []
            for obs, xy in zip(observations, projected, strict=True):
                if xy is None:
                    # Projected onto the horizon: no finite pitch location.
                    placed.append(obs)
                    continue
                xy = (
                    min(max(xy[0], 0.0), cfg.pitch_length),
                    min(max(xy[1], 0.0), cfg.pitch_width),
                )
                v = self._velocity.update(obs.track_id, frame_index, xy)
                placed.append(obs.with_pitch(xy, v))
            observations = placed

            ball_det = next((d for d in detections if d.is_ball), None)
            if ball_det is not None:
                ball_proj = calibration.to_pitch([ball_det.bbox.ground_point])[0]
                if ball_proj is not None:
                    ball_xy = (
                        min(max(ball_proj[0], 0.0), cfg.pitch_length),
                        min(max(ball_proj[1], 0.0), cfg.pitch_width),
                    )

        attacking = self.possession.update(ball_xy, observations)

        return FrameState(
            frame_index=frame_index,
            timestamp=timestamp,
            players=tuple(observations),
            ball_pitch_xy=ball_xy,
            homography=calibration.matrix if calibration else None,
            attacking_team=attacking,
        )

    def run(self, frames: Iterable) -> PipelineResult:
        """Process a sequence of frames and build the match report.

        Runs in passes rather than scoring inline, because two stages need to
        see the whole sequence:

        * **Ball reconstruction** interpolates across occlusions, which needs
          the observation *after* a gap as well as before it. Detectors miss
          the ball often, and scoring only the frames where it happened to be
          visible would discard most of the match.
        * **Possession** is then re-derived from the repaired ball track, so it
          is not fragmented by those same gaps.

        ``frames`` may be any iterable, including a generator reading video
        lazily — no pixel data is held beyond the current frame, though the
        per-frame results are retained.
        """
        self.reset()
        cfg = self.config

        # Pass 1: vision. Possession assigned here is provisional.
        states: list[FrameState] = []
        calibrated = 0
        for i, frame in enumerate(frames):
            state = self.process_frame(frame, i)
            states.append(state)
            if state.is_calibrated:
                calibrated += 1

        # Pass 2: rejoin track fragments. Must run before scoring, since it
        # is what gives each player a sample worth aggregating, and after the
        # vision pass, because it works in pitch coordinates.
        raw_tracks = len({p.track_id for st in states for p in st.players})
        if cfg.stitch is not None:
            mapping = stitch_tracks(states, cfg.fps, cfg.stitch)
            states = apply_stitching(states, mapping)
        stitched_tracks = len({p.track_id for st in states for p in st.players})

        # Pass 3: repair the ball trajectory.
        raw_ball = [s.ball_pitch_xy for s in states]
        detected_ball = sum(1 for p in raw_ball if p is not None)
        ball = smooth_ball_track(raw_ball, cfg.fps, cfg.ball)
        recovered_ball = sum(1 for p in ball if p is not None)

        # Pass 4: re-derive possession from the repaired track.
        self.possession.reset()
        states = [
            replace(
                state,
                ball_pitch_xy=xy,
                attacking_team=self.possession.update(xy, state.players),
            )
            for state, xy in zip(states, ball, strict=True)
        ]

        # Pass 5: score.
        scores = [
            score
            for score in (score_frame(state, cfg.scoring) for state in states)
            if score is not None
        ]

        return PipelineResult(
            report=build_report(scores, len(states), cfg.fps),
            frames=tuple(states),
            scores=tuple(scores),
            calibrated_frames=calibrated,
            rejected_calibrations=self.smoother.rejected,
            ball_detected_frames=detected_ball,
            ball_recovered_frames=recovered_ball,
            raw_tracks=raw_tracks,
            stitched_tracks=stitched_tracks,
        )
