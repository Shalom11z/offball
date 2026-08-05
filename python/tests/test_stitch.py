"""Tracklet stitching."""

from __future__ import annotations

from offball.types import BBox, FrameState, PlayerObservation, Team
from offball.vision.stitch import (
    StitchConfig,
    apply_stitching,
    stitch_tracks,
)

BOX = BBox(0.0, 0.0, 10.0, 20.0)
FPS = 25.0


def obs(tid, xy, vel=(0.0, 0.0), team=Team.HOME):
    return PlayerObservation(tid, BOX, team, pitch_xy=xy, velocity=vel)


def sequence(spec):
    """spec: list of (frame_index, [observations])."""
    return [FrameState(i, i / FPS, tuple(players)) for i, players in spec]


def test_a_fragmented_track_is_rejoined():
    """The core case: a player vanishes for a few frames and comes back."""
    states = []
    for i in range(10):  # track 1, walking at 4 m/s
        states.append((i, [obs(1, (20.0 + 4.0 * i / FPS, 30.0), (4.0, 0.0))]))
    for i in range(10, 14):  # occluded
        states.append((i, []))
    for i in range(14, 24):  # reappears as track 2, where 1 would have been
        states.append((i, [obs(2, (20.0 + 4.0 * i / FPS, 30.0), (4.0, 0.0))]))

    mapping = stitch_tracks(sequence(states), FPS)
    assert mapping[2] == mapping[1], "the two fragments should share an identity"


def test_overlapping_tracks_are_never_merged():
    """Two tracks visible at once are two players, however close."""
    states = [(i, [obs(1, (50.0, 30.0)), obs(2, (50.5, 30.0))]) for i in range(20)]
    mapping = stitch_tracks(sequence(states), FPS)
    assert mapping[1] != mapping[2]


def test_a_long_gap_is_not_bridged():
    states = [(i, [obs(1, (20.0, 30.0))]) for i in range(5)]
    states += [(i, []) for i in range(5, 120)]         # ~4.6s gap
    states += [(i, [obs(2, (20.0, 30.0))]) for i in range(120, 130)]
    mapping = stitch_tracks(sequence(states), FPS, StitchConfig(max_gap_seconds=2.0))
    assert mapping[2] != mapping[1], "beyond max_gap the link would be a guess"


def test_an_impossible_leap_is_rejected():
    """A short gap must not be used to cross the pitch."""
    states = [(i, [obs(1, (10.0, 30.0))]) for i in range(5)]
    states += [(i, []) for i in range(5, 8)]
    states += [(i, [obs(2, (95.0, 30.0))]) for i in range(8, 15)]
    mapping = stitch_tracks(sequence(states), FPS)
    assert mapping[2] != mapping[1]


def test_teams_are_not_crossed():
    states = [(i, [obs(1, (20.0, 30.0), team=Team.HOME)]) for i in range(5)]
    states += [(i, []) for i in range(5, 8)]
    states += [(i, [obs(2, (20.0, 30.0), team=Team.AWAY)]) for i in range(8, 15)]
    mapping = stitch_tracks(sequence(states), FPS)
    assert mapping[2] != mapping[1], "a home player cannot become an away player"


def test_unknown_team_may_still_be_joined():
    """Abstention is not disagreement."""
    states = [(i, [obs(1, (20.0, 30.0), team=Team.HOME)]) for i in range(5)]
    states += [(i, []) for i in range(5, 8)]
    states += [(i, [obs(2, (20.0, 30.0), team=Team.UNKNOWN)]) for i in range(8, 15)]
    mapping = stitch_tracks(sequence(states), FPS)
    assert mapping[2] == mapping[1]


def test_one_predecessor_serves_one_successor():
    """Two fragments appearing near one ended track must not both claim it."""
    states = [(i, [obs(1, (50.0, 30.0))]) for i in range(5)]
    states += [(i, []) for i in range(5, 8)]
    states += [(i, [obs(2, (50.5, 30.0)), obs(3, (51.0, 30.5))]) for i in range(8, 15)]
    mapping = stitch_tracks(sequence(states), FPS)
    assert mapping[2] != mapping[3], "different players must stay distinct"
    assert len({mapping[1], mapping[2], mapping[3]}) == 2


def test_mapping_always_covers_every_track():
    states = [(i, [obs(1, (20.0, 30.0)), obs(2, (60.0, 40.0))]) for i in range(6)]
    mapping = stitch_tracks(sequence(states), FPS)
    assert set(mapping) == {1, 2}
    for canonical in mapping.values():
        assert canonical in mapping, "canonical ids must themselves be known"


def test_empty_and_single_track_inputs():
    assert stitch_tracks([], FPS) == {}
    single = sequence([(0, [obs(1, (20.0, 30.0))])])
    assert stitch_tracks(single, FPS) == {1: 1}


def test_uncalibrated_observations_are_ignored():
    """A track with no pitch position cannot be stitched on motion."""
    states = sequence([(0, [PlayerObservation(1, BOX, Team.HOME)])])
    assert stitch_tracks(states, FPS) == {}


def test_apply_relabels_observations():
    states = sequence([(0, [obs(1, (20.0, 30.0))]), (1, [obs(2, (21.0, 30.0))])])
    out = apply_stitching(states, {1: 1, 2: 1})
    assert [p.track_id for s in out for p in s.players] == [1, 1]
    # Other fields survive.
    assert out[0].players[0].pitch_xy == (20.0, 30.0)


def test_apply_with_empty_mapping_is_a_no_op():
    states = sequence([(0, [obs(1, (20.0, 30.0))])])
    assert apply_stitching(states, {}) is states
