"""Whole-pipeline integration on rendered video frames.

Unlike ``test_pipeline.py``, which injects scripted keypoints, this drives the
**real** classical calibration path: the pipeline is handed actual images and
has to find the pitch lines, fit a homography, project players, reconstruct the
ball and score — with only detection stubbed.

That makes it the closest thing to a real run that exists without footage, and
it is what would break first if the calibration and pipeline stages disagreed
about conventions.

Rendered frames are clean by construction. Passing here says the stages compose
correctly; it says nothing about broadcast video.
"""

from __future__ import annotations

import math

import pytest

np = pytest.importorskip("numpy")
cv2 = pytest.importorskip("cv2")

from test_lines import H_TRUE, render_pitch  # noqa: E402

from offball import kernels  # noqa: E402
from offball.pipeline import Pipeline, PipelineConfig  # noqa: E402
from offball.tactics.offball import ScoringConfig  # noqa: E402
from offball.types import BBox, Detection, PlayerObservation, Team  # noqa: E402
from offball.vision.ball import BallConfig  # noqa: E402
from offball.vision.calibration import CalibrationConfig  # noqa: E402
from offball.vision.detection import ScriptedDetector  # noqa: E402
from offball.vision.lines import ClassicalKeypointSource, _invert  # noqa: E402
from offball.vision.tracking import TrackerConfig  # noqa: E402

FPS = 25.0


def to_image(p):
    q = kernels.project(H_TRUE, [p])[0]
    assert q is not None
    return q


def player_box(pitch_xy) -> BBox:
    px, py = to_image(pitch_xy)
    return BBox(px - 12.0, py - 46.0, px + 12.0, py)


class FixedTeamAssigner:
    """Stands in for kit-colour clustering, which needs real pixels."""

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


def scene(frames: int = 30, *, ball_missing: set[int] | None = None):
    """A move rendered as real images, with matching scripted detections."""
    ball_missing = ball_missing or set()
    images, detections = [], []
    truth = []

    for i in range(frames):
        t = i / FPS
        attackers = [
            (34.0 + 2.0 * t, 42.0),
            (55.0 + 4.0 * t, 34.0),
            (48.0 + 4.5 * t, 14.0),
            (48.0 + 4.0 * t, 54.0),
            (42.0 + 2.5 * t, 26.0),
        ]
        defenders = [
            (3.0, 34.0),
            (74.0 - 3.0 * t, 22.0),
            (72.0 - 3.0 * t, 31.0),
            (72.0 - 3.0 * t, 39.0),
            (74.0 - 3.0 * t, 48.0),
            (84.0 - 2.5 * t, 30.0),
            (84.0 - 2.5 * t, 42.0),
        ]
        ball = attackers[0]
        truth.append((attackers, defenders, ball))

        # The pitch is rendered; players are supplied as detections rather than
        # drawn, so their shapes cannot interfere with line detection.
        images.append(render_pitch())

        frame_dets = [Detection(player_box(p), 0.9) for p in attackers + defenders]
        if i not in ball_missing:
            bx, by = to_image(ball)
            frame_dets.append(Detection(BBox(bx - 5, by - 10, bx + 5, by), 0.8, "ball"))
        detections.append(frame_dets)

    return images, detections, truth


def build(frames: int = 30, *, ball_missing: set[int] | None = None):
    images, detections, truth = scene(frames, ball_missing=ball_missing)

    keypoints = ClassicalKeypointSource()
    # A real deployment seeds this from the period's known camera setup; here it
    # resolves the pitch's own symmetry so positions land in the true frame.
    keypoints.set_prior(_invert(H_TRUE))

    pipeline = Pipeline(
        detector=ScriptedDetector(detections),
        keypoints=keypoints,
        team_assigner=FixedTeamAssigner(attacker_ids={1, 2, 3, 4, 5}),
        config=PipelineConfig(
            fps=FPS,
            scoring=ScoringConfig(grid_nx=35, grid_ny=23),
            calibration=CalibrationConfig(ransac_threshold=2.0, min_inliers=4),
            tracker=TrackerConfig(iou_threshold=0.2),
            ball=BallConfig(max_gap=25),
        ),
    )
    return pipeline, images, truth


@pytest.mark.slow
def test_pipeline_calibrates_real_images_and_scores():
    pipeline, images, _ = build()
    result = pipeline.run(images)

    assert result.calibrated_frames == len(images), (
        f"only calibrated {result.calibrated_frames}/{len(images)} rendered frames"
    )
    assert result.scores, "no frames could be scored"
    assert result.report.coverage > 0.5


@pytest.mark.slow
def test_players_land_near_their_true_pitch_positions():
    pipeline, images, truth = build(10)
    result = pipeline.run(images)

    final = result.frames[-1]
    placed = [p for p in final.players if p.pitch_xy is not None]
    assert len(placed) == 12

    attackers, defenders, _ = truth[-1]
    expected = attackers + defenders
    for obs in placed:
        nearest = min(math.dist(obs.pitch_xy, e) for e in expected)
        assert nearest < 3.0, (
            f"track {obs.track_id} at {obs.pitch_xy} is {nearest:.1f} m from any player"
        )


@pytest.mark.slow
def test_ball_reconstruction_rescues_frames_the_detector_missed():
    """The point of the ball pass: missed detections must not cost coverage."""
    missing = set(range(8, 18))  # ball hidden for 10 frames
    pipeline, images, _ = build(30, ball_missing=missing)
    result = pipeline.run(images)

    assert result.ball_detected_frames == 30 - len(missing)
    assert result.ball_recovered_frames == 30, (
        f"only recovered {result.ball_recovered_frames}/30 ball positions"
    )
    # Every frame keeps a ball position, so possession is not fragmented.
    assert all(f.ball_pitch_xy is not None for f in result.frames)


@pytest.mark.slow
def test_coverage_without_ball_reconstruction_would_be_worse():
    """Quantifies what the ball pass actually buys."""
    missing = set(range(5, 20))
    with_repair, images, _ = build(30, ball_missing=missing)
    repaired = with_repair.run(images)

    # Disable interpolation entirely. BallConfig is frozen, so rebuild rather
    # than mutate.
    without, images2, _ = build(30, ball_missing=missing)
    without = Pipeline(
        detector=without.detector,
        keypoints=without.keypoints,
        team_assigner=without.team_assigner,
        config=PipelineConfig(
            fps=FPS,
            scoring=ScoringConfig(grid_nx=35, grid_ny=23),
            calibration=CalibrationConfig(ransac_threshold=2.0, min_inliers=4),
            tracker=TrackerConfig(iou_threshold=0.2),
            ball=BallConfig(max_gap=0, max_extrapolate=0),
        ),
    )
    without.keypoints.set_prior(_invert(H_TRUE))
    raw = without.run(images2)

    assert repaired.ball_recovered_frames > raw.ball_recovered_frames
