"""Tracklet stitching: rejoin fragments of the same player.

Measured on real footage, the tracker produced 37 identities for 22 players over
12 seconds. Every fragment carries its own partial sample, so a per-player report
built on raw track ids describes glimpses rather than people — and the sample
sizes never reach the threshold at which any figure is worth reading.

Fragments happen because IoU association is local: a missed detection, a player
briefly occluded, or a camera pan that carries someone out of frame and back all
end one track and start another.

## Why motion and not appearance

The obvious fix is an appearance descriptor, and it does not work here.
Teammates wear identical kits, so colour separates *teams* — which the pipeline
already does — but tells you nothing about which of eleven identically dressed
players you are looking at. Re-identification within a team needs jersey numbers
or a learned per-player embedding.

Motion does work, and it is already available. Once observations are projected
to pitch metres, a track that ends at a known position with a known velocity
predicts where that player will be a second later. A new track appearing there is
almost certainly the same person, because football players cannot teleport and
two different players cannot occupy the same square metre.

## What this deliberately will not do

* **Bridge long gaps.** Past a couple of seconds the extrapolation is worthless
  and the link would be a guess. Fragments stay separate.
* **Merge overlapping tracks.** Two tracks visible in the same frame are two
  different players, whatever their positions suggest.
* **Cross teams.** A link that would join a home player to an away player is
  rejected outright.

The result is fewer, longer identities — not correct ones. This raises sample
sizes and makes per-player figures readable; it is not a substitute for real
identification. See ``docs/06-roadmap.md``.

## Measured headroom: modest, and that is the honest ceiling

On 300 frames of real broadcast footage this joined 53 tracks into 47. Breaking
down all 2352 candidate pairs shows why:

===========================  =====  ====================================
Outcome                      Count  Meaning
===========================  =====  ====================================
Overlapping in time           1741  Genuinely different players
Gap longer than 2s             508  Camera panned them out of view
Too far from the prediction     86  Motion did not explain the reappearance
Impossible implied speed         2  Correctly refused
Team mismatch                    8  Correctly refused
**Valid links**                  **7**
===========================  =====  ====================================

The fragments that remain are overwhelmingly *long-gap*: a player leaves frame
during a pan and returns seconds later. Constant-velocity extrapolation carries
no information across that, and widening ``max_gap_seconds`` would not recover
them — it would invent links. Closing that gap needs appearance or jersey-number
re-identification, which is why this module is a stopgap and not the answer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

from ..types import FrameState, Point, Team

__all__ = ["StitchConfig", "TrackSpan", "apply_stitching", "stitch_tracks"]


@dataclass(frozen=True, slots=True)
class StitchConfig:
    #: Longest gap that may be bridged, in seconds. Beyond roughly two seconds
    #: a constant-velocity extrapolation carries no information.
    max_gap_seconds: float = 2.0
    #: How far, in metres, a new track may be from the predicted position.
    max_distance: float = 7.0
    #: Reject a link implying a speed no player reaches. This is what stops a
    #: short gap being used to leap across the pitch.
    max_speed: float = 9.5
    #: Only join tracks the team assigner agrees about. An UNKNOWN track may
    #: join either, since abstention is not disagreement.
    respect_teams: bool = True


@dataclass(frozen=True, slots=True)
class TrackSpan:
    """Where a track began and ended, in time and on the pitch."""

    track_id: int
    first_frame: int
    last_frame: int
    first_xy: Point
    last_xy: Point
    #: Velocity at the track's final observation, m/s.
    last_velocity: Point
    team: Team
    observations: int


def _spans(states: list[FrameState]) -> dict[int, TrackSpan]:
    """Summarise every track's extent from a sequence of frames."""
    first: dict[int, tuple[int, Point]] = {}
    last: dict[int, tuple[int, Point, Point]] = {}
    teams: dict[int, dict[Team, int]] = {}
    counts: dict[int, int] = {}

    for state in states:
        for player in state.players:
            if player.pitch_xy is None:
                continue
            tid = player.track_id
            if tid not in first:
                first[tid] = (state.frame_index, player.pitch_xy)
            last[tid] = (
                state.frame_index,
                player.pitch_xy,
                player.velocity or (0.0, 0.0),
            )
            counts[tid] = counts.get(tid, 0) + 1
            teams.setdefault(tid, {})
            teams[tid][player.team] = teams[tid].get(player.team, 0) + 1

    spans: dict[int, TrackSpan] = {}
    for tid, (f0, p0) in first.items():
        f1, p1, v1 = last[tid]
        # A track's team is whatever it was labelled most often; the assigner
        # itself votes per frame, so this simply resolves the residual noise.
        vote = teams.get(tid, {})
        team = max(vote, key=vote.__getitem__) if vote else Team.UNKNOWN
        spans[tid] = TrackSpan(
            track_id=tid,
            first_frame=f0,
            last_frame=f1,
            first_xy=p0,
            last_xy=p1,
            last_velocity=v1,
            team=team,
            observations=counts.get(tid, 0),
        )
    return spans


def _compatible_team(a: Team, b: Team, respect: bool) -> bool:
    if not respect:
        return True
    if a is Team.UNKNOWN or b is Team.UNKNOWN:
        return True
    return a is b


def stitch_tracks(
    states: list[FrameState], fps: float = 25.0, config: StitchConfig | None = None
) -> dict[int, int]:
    """Work out which track ids are fragments of the same player.

    Returns a mapping from every track id to its canonical id. Tracks with no
    link map to themselves, so the result is always safe to apply.

    Each fragment is joined to at most one predecessor and each predecessor
    accepts at most one successor: a player has exactly one past and one future,
    and allowing many-to-one would merge genuinely different players who happen
    to pass through the same area.
    """
    config = config or StitchConfig()
    dt = 1.0 / fps if fps > 0 else 0.04
    spans = _spans(states)
    if len(spans) < 2:
        return {tid: tid for tid in spans}

    # Candidate links, best (closest to prediction) first.
    candidates: list[tuple[float, int, int]] = []
    ordered = sorted(spans.values(), key=lambda s: s.first_frame)

    for successor in ordered:
        for predecessor in spans.values():
            if predecessor.track_id == successor.track_id:
                continue
            # Must not overlap: simultaneous tracks are different players.
            if predecessor.last_frame >= successor.first_frame:
                continue
            gap_frames = successor.first_frame - predecessor.last_frame
            gap = gap_frames * dt
            if gap <= 0 or gap > config.max_gap_seconds:
                continue
            if not _compatible_team(
                predecessor.team, successor.team, config.respect_teams
            ):
                continue

            # Where the predecessor should be by now, at its last velocity.
            vx, vy = predecessor.last_velocity
            predicted = (
                predecessor.last_xy[0] + vx * gap,
                predecessor.last_xy[1] + vy * gap,
            )
            distance = math.dist(predicted, successor.first_xy)
            if distance > config.max_distance:
                continue
            # Reject links only bridgeable at an impossible speed.
            straight = math.dist(predecessor.last_xy, successor.first_xy)
            if straight / gap > config.max_speed:
                continue
            candidates.append((distance, predecessor.track_id, successor.track_id))

    candidates.sort()

    parent: dict[int, int] = {tid: tid for tid in spans}

    def find(t: int) -> int:
        while parent[t] != t:
            parent[t] = parent[parent[t]]
            t = parent[t]
        return t

    linked_forward: set[int] = set()
    linked_back: set[int] = set()
    for _, pred, succ in candidates:
        if pred in linked_forward or succ in linked_back:
            continue
        if find(pred) == find(succ):
            continue
        linked_forward.add(pred)
        linked_back.add(succ)
        parent[find(succ)] = find(pred)

    return {tid: find(tid) for tid in spans}


def apply_stitching(
    states: list[FrameState], mapping: dict[int, int]
) -> list[FrameState]:
    """Relabel every observation with its canonical track id."""
    if not mapping:
        return states
    out: list[FrameState] = []
    for state in states:
        players = tuple(
            replace(p, track_id=mapping.get(p.track_id, p.track_id))
            for p in state.players
        )
        out.append(replace(state, players=players))
    return out
