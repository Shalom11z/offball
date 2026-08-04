"""Off-the-ball scoring and match-report aggregation."""

from __future__ import annotations

from dataclasses import replace

import pytest

from offball.demo import SyntheticMatch, synthetic_frames
from offball.tactics.offball import ScoringConfig, score_frame
from offball.tactics.report import build_report
from offball.types import BBox, FrameState, PlayerObservation, Team

BOX = BBox(0.0, 0.0, 10.0, 20.0)

# A coarse grid keeps these tests fast; the metrics under test do not depend on
# resolution beyond a percent or two.
FAST = ScoringConfig(grid_nx=35, grid_ny=23)


def player(track_id: int, xy, team: Team, vel=(0.0, 0.0)) -> PlayerObservation:
    return PlayerObservation(
        track_id=track_id, bbox=BOX, team=team, pitch_xy=xy, velocity=vel
    )


def frame(attackers, defenders, ball, attacking=Team.HOME, index=0) -> FrameState:
    players = [player(100 + i, p, Team.HOME) for i, p in enumerate(attackers)]
    players += [player(200 + i, p, Team.AWAY) for i, p in enumerate(defenders)]
    return FrameState(
        frame_index=index,
        timestamp=index * 0.04,
        players=tuple(players),
        ball_pitch_xy=ball,
        homography=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
        attacking_team=attacking,
    )


# ------------------------------------------------------------------ abstention


def test_no_possession_is_not_scored():
    f = frame([(50.0, 30.0), (60.0, 40.0)], [(70.0, 30.0), (75.0, 40.0)], (50.0, 30.0))
    assert score_frame(replace(f, attacking_team=None), FAST) is None
    # A frame credited to the referee is not a possession either.
    assert score_frame(replace(f, attacking_team=Team.REFEREE), FAST) is None


def test_no_ball_is_not_scored():
    players = frame([(50.0, 30.0), (60.0, 40.0)], [(70.0, 30.0), (75.0, 40.0)], None).players
    f = FrameState(0, 0.0, players, None, (1.0,) * 9, Team.HOME)
    assert score_frame(f, FAST) is None


def test_too_few_tracked_players_is_not_scored():
    f = frame([(50.0, 30.0)], [(70.0, 30.0), (75.0, 40.0)], (50.0, 30.0))
    assert score_frame(f, FAST) is None


def test_uncalibrated_players_do_not_count_toward_the_minimum():
    players = (
        player(1, (50.0, 30.0), Team.HOME),
        PlayerObservation(track_id=2, bbox=BOX, team=Team.HOME),  # no pitch_xy
        player(3, (70.0, 30.0), Team.AWAY),
        player(4, (75.0, 40.0), Team.AWAY),
    )
    f = FrameState(0, 0.0, players, (50.0, 30.0), (1.0,) * 9, Team.HOME)
    assert score_frame(f, FAST) is None


# --------------------------------------------------------------------- scoring


def test_ball_carrier_is_excluded_from_off_ball_scores():
    f = frame(
        [(40.0, 34.0), (60.0, 20.0), (65.0, 50.0)],
        [(70.0, 30.0), (72.0, 40.0), (75.0, 34.0)],
        (40.0, 34.0),  # exactly on the first attacker
    )
    score = score_frame(f, FAST)
    assert score is not None
    ids = {p.track_id for p in score.players}
    assert 100 not in ids, "the carrier is on the ball, not off it"
    assert ids == {101, 102}


def test_offside_detection():
    # Defence at 70/72/75, so the offside line is 72 (second-last).
    f = frame(
        [(40.0, 34.0), (80.0, 20.0), (60.0, 50.0)],
        [(70.0, 30.0), (72.0, 40.0), (75.0, 34.0)],
        (40.0, 34.0),
    )
    score = score_frame(f, FAST)
    assert score.offside_line == pytest.approx(72.0)

    beyond = next(p for p in score.players if p.track_id == 101)
    behind = next(p for p in score.players if p.track_id == 102)
    assert beyond.is_offside
    assert beyond.offside_margin == pytest.approx(8.0)
    assert not behind.is_offside
    assert behind.offside_margin == pytest.approx(-12.0)


def test_a_marked_player_in_a_blocked_lane_is_not_available():
    f = frame(
        [(40.0, 34.0), (60.0, 34.0), (58.0, 10.0)],
        [(50.0, 34.0), (60.5, 34.0), (75.0, 50.0)],  # one in the lane, one marking
        (40.0, 34.0),
    )
    score = score_frame(f, FAST)
    marked = next(p for p in score.players if p.track_id == 101)
    assert not marked.lane_open
    assert marked.marking_pressure > 0.8
    assert not marked.is_available


def test_an_open_player_is_available():
    f = frame(
        [(40.0, 34.0), (55.0, 8.0), (50.0, 34.0)],
        [(70.0, 50.0), (75.0, 55.0), (80.0, 60.0)],
        (40.0, 34.0),
    )
    score = score_frame(f, FAST)
    open_player = next(p for p in score.players if p.track_id == 101)
    assert open_player.lane_open
    assert not open_player.is_offside
    assert open_player.is_available
    assert score.available_options >= 1


def test_direction_is_normalised_so_results_do_not_depend_on_which_way_you_attack():
    attackers = [(40.0, 34.0), (60.0, 20.0), (65.0, 50.0)]
    defenders = [(70.0, 30.0), (72.0, 40.0), (75.0, 34.0)]
    ball = (40.0, 34.0)
    forward = score_frame(frame(attackers, defenders, ball), FAST)

    # The identical scene, mirrored: the attack now runs toward -x.
    flip = lambda p: (105.0 - p[0], 68.0 - p[1])  # noqa: E731
    mirrored = score_frame(
        frame([flip(p) for p in attackers], [flip(p) for p in defenders], flip(ball)), FAST
    )

    assert forward.offside_line == pytest.approx(mirrored.offside_line)
    assert forward.team_dangerous_space == pytest.approx(mirrored.team_dangerous_space, rel=1e-9)
    for a, b in zip(forward.players, mirrored.players):
        assert a.offside_margin == pytest.approx(b.offside_margin)
        assert a.lines_broken == b.lines_broken


def test_space_and_hulls_are_populated():
    f = frame(
        [(40.0, 34.0), (60.0, 20.0), (65.0, 50.0)],
        [(70.0, 30.0), (72.0, 40.0), (75.0, 34.0)],
        (40.0, 34.0),
    )
    score = score_frame(f, FAST)
    assert 0.0 < score.team_space < 105.0 * 68.0
    assert score.team_dangerous_space > 0.0
    assert score.attacking_hull > 0.0
    assert score.defending_hull > 0.0
    assert len(score.defensive_lines) == 3
    assert all(p.space_owned > 0.0 for p in score.players)


# ----------------------------------------------------------------- aggregation


def test_report_aggregates_the_demo_sequence():
    match = SyntheticMatch(frames=40)
    states = synthetic_frames(match)
    scores = [s for s in (score_frame(st, FAST) for st in states) if s is not None]
    assert scores, "the synthetic scene must be scoreable"

    report = build_report(scores, len(states), match.fps)
    assert report.frames_scored == len(scores)
    assert report.frames_total == 40
    assert 0.0 < report.coverage <= 1.0
    assert len(report.teams) == 1
    assert report.teams[0].team is Team.HOME

    # Four attackers, minus whichever is carrying the ball.
    assert len(report.players) >= 3
    for p in report.players:
        assert p.frames > 0
        assert p.duration == pytest.approx(p.frames / match.fps)
        assert 0.0 <= p.availability_rate <= 1.0
        assert 0.0 <= p.offside_rate <= 1.0
        assert p.median_space_owned > 0.0


def test_report_of_nothing_is_empty_not_a_crash():
    report = build_report([], frames_total=100)
    assert report.frames_scored == 0
    assert report.coverage == 0.0
    assert report.players == ()
    assert report.teams == ()


def test_coverage_of_zero_frames_does_not_divide_by_zero():
    assert build_report([], frames_total=0).coverage == 0.0


def test_report_lookup_and_serialisation():
    states = synthetic_frames(SyntheticMatch(frames=20))
    scores = [s for s in (score_frame(st, FAST) for st in states) if s is not None]
    report = build_report(scores, len(states))

    first = report.players[0]
    assert report.player(first.track_id) is first
    assert report.player(999999) is None

    data = report.to_dict()
    assert data["frames_scored"] == report.frames_scored
    assert "coverage" in data
    assert len(data["players"]) == len(report.players)
    # Must be JSON-serialisable: this is what the API returns.
    import json

    json.dumps(data, default=str)
