"""Multi-object tracking: turn per-frame detections into persistent identities.

A SORT-style tracker with constant-velocity prediction and greedy IoU
association. No Kalman filter and no appearance embedding — deliberately, for
the reasons in ``docs/02-vision-pipeline.md``:

* A Kalman filter's benefit over constant velocity is small at 25 fps where
  players move a few pixels per frame, and it adds a covariance-tuning problem.
* Appearance embeddings are the *right* answer for the hard cases (players
  overlapping in a corner-kick scrum), but they need a re-ID model. The
  interface here leaves room for one: see ``Tracker.associate``.

What this tracker gives you is stable IDs through ordinary play and clean
handling of occlusion up to ``max_age`` frames. What it does not give you is
identity preservation through a tight ruck; those tracks will swap, and the
downstream metrics treat a fresh track ID as a new player.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from ..types import BBox, Detection, PlayerObservation, Team

__all__ = ["Track", "Tracker", "TrackerConfig"]


@dataclass(slots=True)
class Track:
    """One tracked object's state across frames."""

    track_id: int
    bbox: BBox
    #: Per-frame pixel displacement, used to predict the next position.
    velocity: tuple[float, float] = (0.0, 0.0)
    #: Frames since this track was last matched to a detection.
    age: int = 0
    #: Total detections matched, ever.
    hits: int = 1
    #: Frames since the track was created.
    frames: int = 1
    team: Team = Team.UNKNOWN
    confidence: float = 1.0

    @property
    def is_confirmed(self) -> bool:
        """Whether the track has enough support to be reported.

        Suppressing one- and two-frame tracks removes most detector flicker
        (linesmen, ball boys, crowd) before it reaches the tactics layer.
        """
        return self.hits >= 3

    def predict(self) -> BBox:
        """Where this track should be next frame, at constant velocity."""
        vx, vy = self.velocity
        return BBox(
            self.bbox.x1 + vx, self.bbox.y1 + vy, self.bbox.x2 + vx, self.bbox.y2 + vy
        )

    def update(self, bbox: BBox, confidence: float, smoothing: float) -> None:
        """Fold in a matched detection, smoothing the velocity estimate."""
        prev_cx, prev_cy = self.bbox.centre
        new_cx, new_cy = bbox.centre
        vx, vy = new_cx - prev_cx, new_cy - prev_cy
        # Exponential moving average: raw frame-to-frame deltas are far too
        # noisy to predict with, especially for a jittering bbox.
        self.velocity = (
            smoothing * self.velocity[0] + (1.0 - smoothing) * vx,
            smoothing * self.velocity[1] + (1.0 - smoothing) * vy,
        )
        self.bbox = bbox
        self.confidence = confidence
        self.age = 0
        self.hits += 1
        self.frames += 1

    def mark_missed(self) -> None:
        """Coast the track forward on its own velocity through an occlusion."""
        self.bbox = self.predict()
        self.age += 1
        self.frames += 1


@dataclass(frozen=True, slots=True)
class TrackerConfig:
    #: Minimum IoU between a prediction and a detection to associate them.
    iou_threshold: float = 0.25
    #: Frames a track survives unmatched before deletion. At 25 fps, 30 frames
    #: is 1.2s — long enough to ride out a player passing behind another.
    max_age: int = 30
    #: Weight on the previous velocity estimate, in [0, 1).
    velocity_smoothing: float = 0.6
    #: Detections below this confidence never start a new track (but may still
    #: sustain an existing one, which is the core ByteTrack insight).
    init_confidence: float = 0.5


class Tracker:
    """Frame-by-frame multi-object tracker.

    Usage::

        tracker = Tracker()
        for detections in frames:
            observations = tracker.step(detections)

    The tracker is stateful and assumes frames arrive in order. Construct a new
    one per video, or call :meth:`reset`.
    """

    def __init__(self, config: TrackerConfig | None = None) -> None:
        self.config = config or TrackerConfig()
        self._tracks: list[Track] = []
        self._next_id = 1

    def reset(self) -> None:
        self._tracks.clear()
        self._next_id = 1

    @property
    def tracks(self) -> tuple[Track, ...]:
        return tuple(self._tracks)

    def associate(
        self, predictions: Sequence[BBox], detections: Sequence[Detection]
    ) -> list[tuple[int, int]]:
        """Match tracks to detections, returning ``(track_idx, det_idx)`` pairs.

        Greedy highest-IoU-first. Greedy is within a hair of optimal for this
        problem — the assignment matrix is near-diagonal because players rarely
        swap positions between adjacent frames — and avoids a Hungarian
        implementation or a scipy dependency.

        Override this method to plug in appearance-based association.
        """
        candidates: list[tuple[float, int, int]] = []
        for ti, pred in enumerate(predictions):
            for di, det in enumerate(detections):
                iou = pred.iou(det.bbox)
                if iou >= self.config.iou_threshold:
                    candidates.append((iou, ti, di))
        # Sort by IoU descending; ties broken by index so results are stable.
        candidates.sort(key=lambda c: (-c[0], c[1], c[2]))

        used_tracks: set[int] = set()
        used_dets: set[int] = set()
        pairs: list[tuple[int, int]] = []
        for _, ti, di in candidates:
            if ti in used_tracks or di in used_dets:
                continue
            used_tracks.add(ti)
            used_dets.add(di)
            pairs.append((ti, di))
        return pairs

    def step(self, detections: Sequence[Detection]) -> list[PlayerObservation]:
        """Advance one frame; returns observations for confirmed tracks."""
        players = [d for d in detections if not d.is_ball]

        predictions = [t.predict() for t in self._tracks]
        pairs = self.associate(predictions, players)

        matched_tracks = {ti for ti, _ in pairs}
        matched_dets = {di for _, di in pairs}

        for ti, di in pairs:
            det = players[di]
            self._tracks[ti].update(det.bbox, det.confidence, self.config.velocity_smoothing)

        for ti, track in enumerate(self._tracks):
            if ti not in matched_tracks:
                track.mark_missed()

        for di, det in enumerate(players):
            if di in matched_dets:
                continue
            # Low-confidence unmatched detections are usually crowd or noise;
            # letting them spawn tracks is the main source of ID inflation.
            if det.confidence < self.config.init_confidence:
                continue
            self._tracks.append(
                Track(track_id=self._next_id, bbox=det.bbox, confidence=det.confidence)
            )
            self._next_id += 1

        self._tracks = [t for t in self._tracks if t.age <= self.config.max_age]

        return [
            PlayerObservation(
                track_id=t.track_id, bbox=t.bbox, team=t.team, confidence=t.confidence
            )
            for t in self._tracks
            if t.is_confirmed and t.age == 0
        ]
