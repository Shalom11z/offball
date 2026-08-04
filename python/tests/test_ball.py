"""Ball trajectory reconstruction."""

from __future__ import annotations

import pytest

from offball.vision.ball import BallConfig, BallTracker, smooth_ball_track


def straight_track(n: int = 40, speed: float = 8.0, fps: float = 25.0):
    """A ball moving steadily across the pitch."""
    return [(10.0 + speed * i / fps, 34.0) for i in range(n)]


# ------------------------------------------------------------- basic handling


def test_empty_track():
    assert smooth_ball_track([]) == []


def test_all_missing_stays_all_missing():
    """No observations means no trajectory — not a fabricated one."""
    assert smooth_ball_track([None] * 20) == [None] * 20


def test_a_complete_track_is_preserved():
    track = straight_track()
    out = smooth_ball_track(track, config=BallConfig(smoothing_window=1))
    assert out == track


def test_output_length_always_matches_input():
    for track in ([None] * 5, straight_track(3), [None, (1.0, 2.0), None]):
        assert len(smooth_ball_track(track)) == len(track)


# ---------------------------------------------------------------- gap filling


def test_short_gaps_are_interpolated():
    track: list[tuple[float, float] | None] = list(straight_track(20))
    for i in range(8, 13):  # ball behind a player for 5 frames
        track[i] = None

    out = smooth_ball_track(track, config=BallConfig(smoothing_window=1))
    assert all(p is not None for p in out), "a 5-frame occlusion should be bridged"
    # Interpolation should land close to the true straight-line motion.
    truth = straight_track(20)
    for i in range(8, 13):
        assert out[i][0] == pytest.approx(truth[i][0], abs=0.1)
        assert out[i][1] == pytest.approx(truth[i][1], abs=0.1)


def test_long_gaps_are_left_unfilled():
    """Beyond max_gap the position is not determined, so it stays absent."""
    track: list[tuple[float, float] | None] = list(straight_track(60))
    for i in range(10, 50):  # 40-frame gap, over the 20-frame limit below
        track[i] = None

    out = smooth_ball_track(track, config=BallConfig(max_gap=20, smoothing_window=1))
    assert all(out[i] is None for i in range(15, 45)), "a long gap must not be invented"


def test_leading_and_trailing_gaps_hold_the_nearest_known_position():
    track: list[tuple[float, float] | None] = [None, None, *straight_track(10)]
    track += [None, None]

    out = smooth_ball_track(track, config=BallConfig(max_extrapolate=3, smoothing_window=1))
    assert out[0] == out[2], "leading frames hold the first known position"
    assert out[-1] == out[-3], "trailing frames hold the last known position"


def test_extrapolation_is_bounded():
    track: list[tuple[float, float] | None] = [*([None] * 20), *straight_track(5)]
    out = smooth_ball_track(track, config=BallConfig(max_extrapolate=3, smoothing_window=1))
    # Only 3 frames before the first observation may be filled.
    assert out[16] is None
    assert out[17] is not None


# ------------------------------------------------------------ outlier removal


def test_an_impossible_jump_is_rejected():
    track: list[tuple[float, float] | None] = list(straight_track(20))
    track[10] = (100.0, 5.0)  # a white boot, 90m away in one frame

    out = smooth_ball_track(track, config=BallConfig(smoothing_window=1))
    truth = straight_track(20)
    # The bad frame is replaced by interpolation between its good neighbours.
    assert out[10][0] == pytest.approx(truth[10][0], abs=0.5)


def test_a_single_outlier_does_not_reject_the_frames_after_it():
    """The anchor must follow accepted points, not the last observation."""
    track: list[tuple[float, float] | None] = list(straight_track(20))
    track[5] = (95.0, 60.0)

    out = smooth_ball_track(track, config=BallConfig(smoothing_window=1))
    truth = straight_track(20)
    for i in range(6, 20):
        assert out[i][0] == pytest.approx(truth[i][0], abs=0.5), f"frame {i} was lost"


def test_a_fast_shot_is_not_rejected():
    """30 m/s is a hard shot, not a mis-detection."""
    track = [(10.0 + 30.0 * i / 25.0, 34.0) for i in range(15)]
    out = smooth_ball_track(track, config=BallConfig(max_speed=45.0, smoothing_window=1))
    assert all(p is not None for p in out)
    assert out[-1][0] == pytest.approx(track[-1][0], abs=0.01)


# ------------------------------------------------------------------ smoothing


def test_smoothing_reduces_jitter():
    import random

    rng = random.Random(0)
    truth = straight_track(40)
    noisy = [(p[0] + rng.uniform(-0.5, 0.5), p[1] + rng.uniform(-0.5, 0.5)) for p in truth]

    smoothed = smooth_ball_track(noisy, config=BallConfig(smoothing_window=5))

    def error(track):
        return sum(abs(a[0] - b[0]) + abs(a[1] - b[1]) for a, b in zip(track, truth)) / len(truth)

    assert error(smoothed) < error(noisy)


def test_smoothing_does_not_average_across_a_gap():
    """Positions either side of an unbridged gap must stay independent."""
    left = [(10.0, 34.0)] * 5
    right = [(90.0, 34.0)] * 5
    track = left + [None] * 60 + right

    out = smooth_ball_track(track, config=BallConfig(max_gap=10, smoothing_window=5))
    assert out[4][0] == pytest.approx(10.0, abs=0.01)
    assert out[-1][0] == pytest.approx(90.0, abs=0.01)


def test_smoothing_window_of_one_is_a_no_op():
    track = straight_track(10)
    assert smooth_ball_track(track, config=BallConfig(smoothing_window=1)) == track


# ------------------------------------------------------------ what it recovers


def test_reconstruction_recovers_most_of_an_intermittent_track():
    """The headline claim: a detector that misses the ball often is usable.

    Simulates a detector finding the ball in roughly 55% of frames, in bursts
    rather than uniformly — which is how real misses arrive.
    """
    import random

    rng = random.Random(7)
    truth = straight_track(200, speed=4.0)
    observed: list[tuple[float, float] | None] = list(truth)

    i = 0
    while i < len(observed):
        if rng.random() < 0.35:
            for j in range(i, min(len(observed), i + rng.randint(3, 12))):
                observed[j] = None
            i += 12
        i += 1

    detected = sum(1 for p in observed if p is not None)
    out = smooth_ball_track(observed, config=BallConfig(max_gap=50))
    recovered = sum(1 for p in out if p is not None)

    assert detected < len(truth) * 0.8, "the simulated detector should miss a lot"
    assert recovered > len(truth) * 0.95, (
        f"only recovered {recovered}/{len(truth)} from {detected} detections"
    )
    # And the recovered positions should be close to the truth.
    errors = [abs(o[0] - t[0]) for o, t in zip(out, truth) if o is not None]
    assert max(errors) < 1.0, f"worst interpolation error {max(errors):.2f} m"


# -------------------------------------------------------------- streaming API


def test_streaming_tracker_holds_through_a_gap():
    tracker = BallTracker(fps=25.0, config=BallConfig(max_gap=10))
    assert tracker.update(0, (10.0, 34.0)) == (10.0, 34.0)
    assert tracker.update(1, None) == (10.0, 34.0), "holds the last known position"
    assert tracker.update(20, None) is None, "gives up beyond max_gap"


def test_streaming_tracker_rejects_impossible_jumps():
    tracker = BallTracker(fps=25.0)
    tracker.update(0, (10.0, 34.0))
    result = tracker.update(1, (100.0, 5.0))
    assert tracker.rejected == 1
    assert result == (10.0, 34.0), "falls back to the last good position"


def test_streaming_tracker_reset():
    tracker = BallTracker()
    tracker.update(0, (10.0, 34.0))
    tracker.reset()
    assert tracker.update(1, None) is None
    assert tracker.rejected == 0
