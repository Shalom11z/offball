"""Tracker behaviour: identity stability, occlusion, and noise rejection."""

from __future__ import annotations

import pytest

from offball.types import BBox, Detection
from offball.vision.tracking import Track, Tracker, TrackerConfig


def box(x: float, y: float, w: float = 20.0, h: float = 50.0) -> BBox:
    return BBox(x, y, x + w, y + h)


def det(x: float, y: float, conf: float = 0.9) -> Detection:
    return Detection(box(x, y), conf)


def run(tracker: Tracker, frames: list[list[Detection]]) -> list[list[int]]:
    """Returns the confirmed track ids reported for each frame."""
    return [[o.track_id for o in tracker.step(f)] for f in frames]


def test_a_steadily_moving_player_keeps_one_id():
    tracker = Tracker()
    frames = [[det(100.0 + 5 * i, 200.0)] for i in range(10)]
    ids = run(tracker, frames)
    # Confirmation needs 3 hits, so nothing is reported for the first two frames.
    assert ids[0] == [] and ids[1] == []
    reported = [i for frame in ids for i in frame]
    assert reported, "the track should be confirmed and reported"
    assert len(set(reported)) == 1, "a single player must not fragment into several ids"


def test_two_players_keep_distinct_ids():
    tracker = Tracker()
    frames = [[det(100.0 + 4 * i, 200.0), det(500.0 - 4 * i, 300.0)] for i in range(10)]
    ids = run(tracker, frames)
    final = ids[-1]
    assert len(final) == 2
    assert len(set(final)) == 2


def test_a_track_survives_a_short_occlusion():
    tracker = Tracker(TrackerConfig(max_age=30))
    # Visible, then hidden for 5 frames, then back where it would have coasted to.
    frames = [[det(100.0 + 5 * i, 200.0)] for i in range(6)]
    frames += [[] for _ in range(5)]
    frames += [[det(100.0 + 5 * i, 200.0)] for i in range(11, 16)]

    ids = run(tracker, frames)
    before = ids[5][0]
    after = ids[-1][0]
    assert before == after, "the same player must not get a new id after a brief occlusion"


def test_a_track_dies_after_max_age():
    tracker = Tracker(TrackerConfig(max_age=3))
    frames = [[det(100.0, 200.0)] for _ in range(5)]
    frames += [[] for _ in range(10)]
    frames += [[det(100.0, 200.0)] for _ in range(5)]
    ids = run(tracker, frames)
    assert ids[4][0] != ids[-1][0], "after a long gap it is a new track"


def test_low_confidence_detections_do_not_spawn_tracks():
    tracker = Tracker(TrackerConfig(init_confidence=0.5))
    frames = [[det(100.0, 200.0, conf=0.2)] for _ in range(10)]
    ids = run(tracker, frames)
    assert all(frame == [] for frame in ids), "crowd-level noise must not become a player"


def test_flicker_never_reaches_the_confirmation_threshold():
    tracker = Tracker()
    # A detection appearing in a different place every frame: three separate
    # one-hit tracks, none of which confirms.
    frames = [[det(100.0 + 300 * (i % 3), 200.0)] for i in range(6)]
    ids = run(tracker, frames)
    assert all(len(frame) == 0 for frame in ids[:2])


def test_balls_are_not_tracked_as_players():
    tracker = Tracker()
    frames = [
        [det(100.0 + 5 * i, 200.0), Detection(box(300.0, 400.0, 8, 8), 0.8, "ball")]
        for i in range(6)
    ]
    ids = run(tracker, frames)
    assert len(ids[-1]) == 1, "the ball must not become a player track"


def test_velocity_is_smoothed_not_raw():
    track = Track(track_id=1, bbox=box(0.0, 0.0))
    track.update(box(10.0, 0.0), 0.9, smoothing=0.6)
    # One update with 60% weight on the previous (zero) velocity.
    assert track.velocity[0] == pytest.approx(4.0)
    assert track.velocity[1] == pytest.approx(0.0)


def test_prediction_moves_the_box_by_the_velocity():
    track = Track(track_id=1, bbox=box(0.0, 0.0), velocity=(5.0, -2.0))
    pred = track.predict()
    assert pred.x1 == pytest.approx(5.0)
    assert pred.y1 == pytest.approx(-2.0)


def test_reset_clears_state():
    tracker = Tracker()
    run(tracker, [[det(100.0, 200.0)] for _ in range(5)])
    assert tracker.tracks
    tracker.reset()
    assert tracker.tracks == ()
    ids = run(tracker, [[det(100.0, 200.0)] for _ in range(5)])
    assert ids[-1][0] == 1, "ids restart after a reset"


def test_association_is_deterministic():
    a, b = Tracker(), Tracker()
    frames = [[det(100.0 + 3 * i, 200.0), det(140.0 + 3 * i, 205.0)] for i in range(8)]
    assert run(a, frames) == run(b, frames)


def test_bbox_ground_point_is_the_feet():
    bb = box(100.0, 200.0, 20.0, 50.0)
    assert bb.ground_point == (110.0, 250.0)
    assert bb.centre == (110.0, 225.0)


def test_degenerate_bbox_is_rejected():
    with pytest.raises(ValueError, match="degenerate"):
        BBox(10.0, 10.0, 5.0, 20.0)
