"""End-to-end pipeline: detection -> tracking -> calibration -> scoring.

Runs the real tracker, the real calibration stage and the real metrics against
scripted detections. No model weights, no video file, no GPU — which is the
whole point of the injected-dependency design in :class:`offball.pipeline.Pipeline`.
"""

from __future__ import annotations

from offball import kernels
from offball.pipeline import Pipeline, PipelineConfig, PossessionTracker
from offball.tactics.offball import ScoringConfig
from offball.types import BBox, Detection, PlayerObservation, Team
from offball.vision.calibration import CalibrationConfig, ScriptedKeypoints
from offball.vision.detection import ScriptedDetector
from offball.vision.tracking import TrackerConfig

# A fixed broadcast camera, as an exact pitch -> image homography: gantry at
# the halfway line, so the far touchline sits higher in frame and is pulled
# toward the centre. Correspondences are generated from it rather than written
# by hand, so they are perfectly self-consistent and any calibration error the
# tests see comes from the code under test.
PITCH_TO_IMAGE = (12.0, 5.3, 250.0, 0.2, -6.5, 980.0, 0.0, 0.004, 1.0)

_LANDMARKS = [
    (0.0, 0.0), (105.0, 0.0), (105.0, 68.0), (0.0, 68.0),
    (52.5, 34.0), (52.5, 0.0), (52.5, 68.0), (16.5, 13.84),
    (88.5, 54.16), (11.0, 34.0),
]

CALIB_PAIRS = [
    (kernels.project(PITCH_TO_IMAGE, [p])[0], p) for p in _LANDMARKS
]

FAST_SCORING = ScoringConfig(grid_nx=35, grid_ny=23)


def fake_frames(n: int) -> list[object]:
    """Placeholder frame objects.

    The scripted detector and keypoint source ignore pixel data, but the frames
    must not be ``None``: the pipeline skips team assignment on a ``None``
    frame, because a real assigner has no pixels to read a kit colour from.
    """
    return [object() for _ in range(n)]


def image_box(pitch_xy: tuple[float, float]) -> BBox:
    """A player-shaped box whose ground point projects back to ``pitch_xy``."""
    p = kernels.project(PITCH_TO_IMAGE, [pitch_xy])[0]
    assert p is not None, f"{pitch_xy} does not project into the frame"
    px, py = p
    return BBox(px - 12.0, py - 46.0, px + 12.0, py)


class FixedTeamAssigner:
    """Assigns teams by track id, standing in for colour clustering.

    Kit-colour assignment needs real pixels; this keeps the end-to-end test
    focused on the geometry and orchestration.
    """

    def __init__(self, attacker_ids: set[int]) -> None:
        self.attacker_ids = attacker_ids

    def assign(self, frame, observations):
        return [
            PlayerObservation(
                track_id=o.track_id,
                bbox=o.bbox,
                team=Team.HOME if o.track_id in self.attacker_ids else Team.AWAY,
                confidence=o.confidence,
                pitch_xy=o.pitch_xy,
                velocity=o.velocity,
            )
            for o in observations
        ]


def scripted_move(frames: int = 40, fps: float = 25.0):
    """An attacking move: forwards pushing up against a retreating back line.

    Motion is expressed in metres per second of elapsed time, not as a fraction
    of the clip. Scaling displacement by frame count instead would make short
    clips play at absurd speeds and break IoU association — which is a property
    of the test data, not of the tracker.
    """
    detections = []
    keypoints = []
    for i in range(frames):
        t = i / fps  # seconds
        attackers = [
            (34.0 + 2.0 * t, 42.0),   # carrier, driving forward at 2 m/s
            (55.0 + 4.0 * t, 34.0),   # striker, running the line at 4 m/s
            (48.0 + 4.5 * t, 14.0),   # right winger
            (48.0 + 4.0 * t, 54.0),   # left winger
            (42.0 + 2.5 * t, 26.0),   # attacking midfielder
        ]
        defenders = [
            (3.0, 34.0),              # goalkeeper
            (74.0 - 3.0 * t, 22.0),   # back four retreating at 3 m/s
            (72.0 - 3.0 * t, 31.0),
            (72.0 - 3.0 * t, 39.0),
            (74.0 - 3.0 * t, 48.0),
            (84.0 - 2.5 * t, 30.0),   # holding midfielders
            (84.0 - 2.5 * t, 42.0),
        ]
        ball = attackers[0]

        frame_dets = [Detection(image_box(p), 0.9) for p in attackers + defenders]
        bx, by = kernels.project(PITCH_TO_IMAGE, [ball])[0]
        frame_dets.append(Detection(BBox(bx - 5, by - 10, bx + 5, by), 0.8, "ball"))

        detections.append(frame_dets)
        keypoints.append(list(CALIB_PAIRS))
    return detections, keypoints


def build_pipeline(frames: int = 40) -> tuple[Pipeline, int]:
    detections, keypoints = scripted_move(frames)
    # Detections are emitted attackers-first, so the tracker assigns ids 1-5 to
    # them and 6-12 to the defence.
    pipeline = Pipeline(
        detector=ScriptedDetector(detections),
        keypoints=ScriptedKeypoints(keypoints),
        team_assigner=FixedTeamAssigner(attacker_ids={1, 2, 3, 4, 5}),
        config=PipelineConfig(
            scoring=FAST_SCORING,
            calibration=CalibrationConfig(ransac_threshold=2.0, min_inliers=6),
            tracker=TrackerConfig(iou_threshold=0.2),
        ),
    )
    return pipeline, frames


def test_pipeline_runs_end_to_end():
    pipeline, frames = build_pipeline()
    result = pipeline.run(fake_frames(frames))

    assert len(result.frames) == frames
    assert result.calibrated_frames == frames, "a static camera must calibrate every frame"
    assert result.rejected_calibrations == 0
    assert result.scores, "the move should produce scored frames"
    assert result.report.coverage > 0.5


def test_projection_recovers_pitch_positions():
    pipeline, frames = build_pipeline(10)
    result = pipeline.run(fake_frames(frames))

    placed = [p for p in result.frames[-1].players if p.pitch_xy is not None]
    assert len(placed) == 12, "all 12 players should be on the pitch"
    for p in placed:
        x, y = p.pitch_xy
        assert 0.0 <= x <= 105.0
        assert 0.0 <= y <= 68.0


def test_tracking_ids_are_stable_across_the_move():
    pipeline, frames = build_pipeline()
    result = pipeline.run(fake_frames(frames))

    early = {p.track_id for p in result.frames[5].players}
    late = {p.track_id for p in result.frames[-1].players}
    assert early == late, "no player should be lost or re-identified during clean tracking"
    assert len(early) == 12


def test_velocity_is_estimated_in_metres_per_second():
    pipeline, frames = build_pipeline()
    result = pipeline.run(fake_frames(frames))

    speeds = [
        p.speed for p in result.frames[-1].players if p.speed is not None and p.speed > 0.01
    ]
    assert speeds, "moving players should have non-zero speed"
    # The scripted players run at 2-4.5 m/s. Recovering roughly that from
    # projected pixel positions is the real check here: a metres/pixels or a
    # per-frame/per-second mix-up would put this out by an order of magnitude.
    assert max(speeds) < 8.0, f"implausible speed: {max(speeds):.1f} m/s"
    assert max(speeds) > 1.0, f"suspiciously slow: {max(speeds):.1f} m/s"


def test_possession_is_detected():
    pipeline, frames = build_pipeline()
    result = pipeline.run(fake_frames(frames))
    possessed = [f for f in result.frames if f.attacking_team is Team.HOME]
    assert len(possessed) > frames * 0.5


def test_report_names_the_off_ball_players():
    pipeline, frames = build_pipeline()
    result = pipeline.run(fake_frames(frames))

    ids = {p.track_id for p in result.report.players}
    # Track 1 carries the ball throughout, so it is never an off-ball actor.
    assert 1 not in ids
    assert ids <= {2, 3, 4, 5}
    assert ids, "the other attackers should be scored"


def test_uncalibrated_frames_are_handled():
    detections, keypoints = scripted_move(20)
    # Frames 8-12 are a close-up: no pitch markings visible.
    for i in range(8, 13):
        keypoints[i] = None

    pipeline = Pipeline(
        detector=ScriptedDetector(detections),
        keypoints=ScriptedKeypoints(keypoints),
        team_assigner=FixedTeamAssigner(attacker_ids={1, 2, 3, 4, 5}),
        config=PipelineConfig(
            scoring=FAST_SCORING,
            calibration=CalibrationConfig(ransac_threshold=2.0, min_inliers=6),
            tracker=TrackerConfig(iou_threshold=0.2),
        ),
    )
    result = pipeline.run(fake_frames(20))
    # The smoother coasts through the gap rather than dropping the frames.
    assert result.calibrated_frames == 20
    assert len(result.frames) == 20


def test_reset_allows_reuse():
    pipeline, frames = build_pipeline(10)
    first = pipeline.run(fake_frames(frames))
    pipeline.detector.reset()
    pipeline.keypoints.reset()
    second = pipeline.run(fake_frames(frames))
    assert len(first.frames) == len(second.frames)
    assert first.report.frames_scored == second.report.frames_scored


# ------------------------------------------------------------------ possession


def test_possession_needs_a_clear_margin_to_switch():
    tracker = PossessionTracker(hysteresis=1.5, max_distance=4.0)
    home = PlayerObservation(1, BBox(0, 0, 1, 1), Team.HOME, pitch_xy=(50.0, 34.0))
    away = PlayerObservation(2, BBox(0, 0, 1, 1), Team.AWAY, pitch_xy=(53.0, 34.0))

    assert tracker.update((50.0, 34.0), [home, away]) is Team.HOME

    # Away is nearer now (1.0m vs 2.0m), but only by 1.0m — less than the 1.5m
    # margin. This is the 50-50 case that would otherwise flicker every frame.
    assert tracker.update((52.0, 34.0), [home, away]) is Team.HOME

    # Beaten by a clear margin (0.1m vs 2.9m): possession genuinely switches.
    assert tracker.update((52.9, 34.0), [home, away]) is Team.AWAY


def test_a_loose_ball_holds_the_previous_possession():
    tracker = PossessionTracker(max_distance=4.0)
    home = PlayerObservation(1, BBox(0, 0, 1, 1), Team.HOME, pitch_xy=(50.0, 34.0))
    assert tracker.update((50.0, 34.0), [home]) is Team.HOME
    # Ball 30m away: in flight, not a change of possession.
    assert tracker.update((80.0, 34.0), [home]) is Team.HOME


def test_possession_ignores_untracked_and_unteamed_players():
    tracker = PossessionTracker()
    unknown = PlayerObservation(1, BBox(0, 0, 1, 1), Team.UNKNOWN, pitch_xy=(50.0, 34.0))
    assert tracker.update((50.0, 34.0), [unknown]) is None
    assert tracker.update(None, []) is None
