"""Team assignment from kit colour.

The approach: crop each player's torso, reduce it to a dominant colour after
throwing away pitch-green pixels, cluster the frame's players into two groups,
and hold the assignment steady over time with a per-track vote.

Why colour and not a learned classifier: kits change every match and there are
only two of them, so any supervised model has to be retrained per fixture.
Clustering is unsupervised, needs no labels, and its failure modes (similar
kits, heavy shadow) are the ones a supervised model would share anyway.

Known limits, also in ``docs/02-vision-pipeline.md``:

* Goalkeepers wear a third kit and are routinely misassigned. The pipeline
  handles this positionally rather than by colour — see
  :func:`identify_goalkeepers`.
* Referees wear a fourth. They are separated as the cluster that is smallest
  and furthest from both team centroids.
* Two teams in similar colours (a red-vs-orange fixture) will fail. The
  ``separation`` field on the result is there so callers can detect this and
  fall back rather than silently reporting nonsense.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass

from ..types import BBox, PlayerObservation, Team

__all__ = ["KitProfile", "TeamAssigner", "identify_goalkeepers"]


@dataclass(frozen=True, slots=True)
class KitProfile:
    """The two kit colours found in a match, as RGB triples in 0-255."""

    home: tuple[float, float, float]
    away: tuple[float, float, float]
    #: Euclidean RGB distance between the two centroids. Below ~40 the teams
    #: are too close in colour for this method to be trusted.
    separation: float

    @property
    def is_reliable(self) -> bool:
        return self.separation >= 40.0


def _torso_crop(frame, bbox: BBox):
    """The upper-middle of the bbox: shirt, not shorts, socks, or turf."""
    h, w = frame.shape[:2]
    x1 = max(0, int(bbox.x1 + bbox.width * 0.25))
    x2 = min(w, int(bbox.x2 - bbox.width * 0.25))
    y1 = max(0, int(bbox.y1 + bbox.height * 0.15))
    y2 = min(h, int(bbox.y1 + bbox.height * 0.50))
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2]


def dominant_colour(frame, bbox: BBox) -> tuple[float, float, float] | None:
    """Mean RGB of a player's torso, with pitch-green pixels removed.

    Returns ``None`` when the crop is empty or is essentially all grass, which
    happens for badly-placed boxes and for players lying on the ground.
    """
    import numpy as np

    crop = _torso_crop(frame, bbox)
    if crop is None or crop.size == 0:
        return None

    # Frames are BGR (OpenCV); convert once, here.
    pixels = crop.reshape(-1, 3)[:, ::-1].astype(np.float32)

    # Drop grass: green dominant over both red and blue by a clear margin.
    r, g, b = pixels[:, 0], pixels[:, 1], pixels[:, 2]
    keep = ~((g > r * 1.15) & (g > b * 1.15))
    kept = pixels[keep]
    if kept.shape[0] < max(8, 0.1 * pixels.shape[0]):
        return None
    return tuple(float(v) for v in kept.mean(axis=0))


def _kmeans2(points: list[tuple[float, float, float]], iterations: int = 25):
    """2-means on RGB, seeded with the two furthest-apart points.

    Deterministic seeding matters: a random init makes the home/away labelling
    flip between runs on the same footage.
    """
    if len(points) < 2:
        return None

    best_pair = (0, 1)
    best_d = -1.0
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            d = sum((points[i][k] - points[j][k]) ** 2 for k in range(3))
            if d > best_d:
                best_d, best_pair = d, (i, j)
    centres = [points[best_pair[0]], points[best_pair[1]]]

    labels = [0] * len(points)
    for _ in range(iterations):
        changed = False
        for idx, p in enumerate(points):
            d0 = sum((p[k] - centres[0][k]) ** 2 for k in range(3))
            d1 = sum((p[k] - centres[1][k]) ** 2 for k in range(3))
            lab = 0 if d0 <= d1 else 1
            if lab != labels[idx]:
                labels[idx] = lab
                changed = True
        for c in (0, 1):
            members = [p for p, lab in zip(points, labels, strict=True) if lab == c]
            if members:
                centres[c] = tuple(
                    sum(m[k] for m in members) / len(members) for k in range(3)
                )
        if not changed:
            break

    # Stable ordering: the darker centroid is always cluster 0, so "home" does
    # not swap between runs.
    if sum(centres[0]) > sum(centres[1]):
        centres.reverse()
        labels = [1 - lab for lab in labels]
    return centres, labels


class TeamAssigner:
    """Assigns tracks to teams and keeps the assignment stable over time.

    Two-phase by design:

    1. :meth:`fit` learns the two kit colours from a sample of frames. Doing
       this once up front, rather than per frame, is what stops assignments
       flickering when a player turns or runs through shadow.
    2. :meth:`assign` labels each observation, then a running per-track vote
       smooths out the remaining single-frame errors.
    """

    def __init__(self, min_votes: int = 5) -> None:
        self.profile: KitProfile | None = None
        self._votes: dict[int, Counter] = defaultdict(Counter)
        self._min_votes = min_votes

    def fit(self, samples: list[tuple[object, list[BBox]]]) -> KitProfile | None:
        """Learn kit colours from ``(frame, player_boxes)`` pairs.

        Sample frames from across the whole match, not just the opening
        minutes: lighting shifts substantially over 90 minutes, and a profile
        fitted only on a sunlit first half degrades badly under floodlights.

        Returns ``None`` if too few usable torso crops were found.
        """
        colours: list[tuple[float, float, float]] = []
        for frame, boxes in samples:
            for box in boxes:
                c = dominant_colour(frame, box)
                if c is not None:
                    colours.append(c)

        if len(colours) < 6:
            return None
        result = _kmeans2(colours)
        if result is None:
            return None
        centres, _ = result
        separation = sum((centres[0][k] - centres[1][k]) ** 2 for k in range(3)) ** 0.5
        self.profile = KitProfile(home=centres[0], away=centres[1], separation=separation)
        return self.profile

    def classify_colour(self, colour: tuple[float, float, float]) -> Team:
        """Nearest-kit lookup for a single torso colour."""
        if self.profile is None:
            return Team.UNKNOWN
        dh = sum((colour[k] - self.profile.home[k]) ** 2 for k in range(3))
        da = sum((colour[k] - self.profile.away[k]) ** 2 for k in range(3))
        return Team.HOME if dh <= da else Team.AWAY

    def assign(
        self, frame, observations: list[PlayerObservation]
    ) -> list[PlayerObservation]:
        """Label observations with a team, using the running per-track vote.

        A track keeps ``Team.UNKNOWN`` until it has accumulated ``min_votes``
        observations. Abstaining is correct here: a wrongly-teamed player
        corrupts the offside line and the control field for every frame they
        appear in.
        """
        if self.profile is None:
            return observations

        out: list[PlayerObservation] = []
        for obs in observations:
            colour = dominant_colour(frame, obs.bbox)
            if colour is not None:
                self._votes[obs.track_id][self.classify_colour(colour)] += 1

            votes = self._votes[obs.track_id]
            total = sum(votes.values())
            team = votes.most_common(1)[0][0] if total >= self._min_votes else Team.UNKNOWN
            out.append(
                PlayerObservation(
                    track_id=obs.track_id,
                    bbox=obs.bbox,
                    team=team,
                    confidence=obs.confidence,
                    pitch_xy=obs.pitch_xy,
                    velocity=obs.velocity,
                )
            )
        return out

    def reset(self) -> None:
        self._votes.clear()


def identify_goalkeepers(
    players: list[PlayerObservation], pitch_length: float = 105.0
) -> dict[int, Team]:
    """Find each team's goalkeeper positionally, returning ``track_id -> team``.

    Colour clustering cannot do this — keepers wear a third kit — but geometry
    can: over a full match the two players with the most extreme mean ``x`` are
    the keepers, and each belongs to the team defending that end.

    Call this with observations aggregated over many frames, not a single one;
    on one frame an overlapping full-back will beat the keeper.
    """
    positioned = [p for p in players if p.pitch_xy is not None]
    if len(positioned) < 2:
        return {}

    by_x = sorted(positioned, key=lambda p: p.pitch_xy[0])
    deepest, highest = by_x[0], by_x[-1]

    out: dict[int, Team] = {}
    # A keeper is only credible near their own goal line.
    if deepest.pitch_xy[0] < pitch_length * 0.12:
        out[deepest.track_id] = deepest.team
    if highest.pitch_xy[0] > pitch_length * 0.88:
        out[highest.track_id] = highest.team
    return out
