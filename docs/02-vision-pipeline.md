# Vision pipeline

How raw frames become pitch coordinates, and the ways each stage fails.

The failure modes matter more than the happy paths. Every stage here works on
clean footage; what separates a usable system from a demo is what happens on
a corner-kick scrum, a floodlit second half, or a director cutting to a
close-up.

## 1. Detection

**Module**: [`offball/vision/detection.py`](../python/src/offball/vision/detection.py)

The pipeline depends on the `Detector` protocol, never a concrete model.

```python
class Detector(Protocol):
    def detect(self, frame) -> list[Detection]: ...
```

Two implementations ship: `YoloDetector` (Ultralytics, lazily imported) and
`ScriptedDetector` (replays a fixed list, for deterministic tests).

**No weights ship with this repository.**

### What a COCO-pretrained model gets you

Players: acceptable. `person` covers players, officials and, in wide shots,
some of the crowd behind the advertising boards.

The ball: badly. It is often under 10 pixels across in a broadcast wide shot,
motion-blurs into an ellipse when struck, and disappears entirely behind
players. COCO's `sports ball` class was not trained on it at that scale.

**Spend the annotation budget on the ball.** It is the single highest-leverage
label in the system: possession assignment depends on it, and possession sets
the direction every off-ball metric is computed in.

### Settings that matter

`image_size` defaults to **1280**, not the usual 640. Halving input resolution
loses the ball entirely. This roughly doubles inference cost and is worth it.

`confidence` is deliberately low (0.25) at the detector, with stricter
filtering downstream: the tracker's `init_confidence` (0.5) governs whether a
detection may *start* a track, while low-confidence detections may still
*sustain* an existing one. That asymmetry is the core ByteTrack insight and it
buys most of the benefit without a second association pass.

## 2. Tracking

**Module**: [`offball/vision/tracking.py`](../python/src/offball/vision/tracking.py)

SORT-style: constant-velocity prediction, greedy IoU association, exponential
smoothing of the velocity estimate.

### Why not a Kalman filter

At 25fps a player moves a few pixels per frame. The gain over constant velocity
is small, and a Kalman filter brings a covariance-tuning problem that is
genuinely fiddly to get right on this data. Constant velocity plus EMA
smoothing is within noise of it here.

### Why greedy rather than Hungarian

The assignment matrix is near-diagonal — players rarely swap positions between
adjacent frames — so greedy highest-IoU-first is within a hair of optimal, and
avoids a scipy dependency. Ties break by index, so association is
deterministic.

### What it does well

- Stable identities through ordinary play.
- Occlusion up to `max_age` frames (default 30 ≈ 1.2s at 25fps), coasting the
  box forward on its own velocity.
- Suppressing detector flicker: tracks are only reported after 3 hits, which
  removes most crowd and linesman noise before it reaches the tactics layer.

### What it does badly

**Identity through a tight ruck.** A corner-kick scrum will swap IDs. There is
no appearance model, so once two boxes overlap heavily the IoU signal cannot
distinguish them.

The consequence is contained by design: a swapped ID looks like two players
each ending one track and starting another. Downstream, a fresh `track_id` is
simply a new player with its own (smaller) sample. It corrupts *attribution*,
not the team-level metrics.

The seam for fixing this is `Tracker.associate`, which can be overridden to
add appearance-embedding association. See [06 Roadmap](06-roadmap.md).

**High `stride` values.** IoU association assumes small frame-to-frame motion.
At stride 5 a sprinting player moves further than their own box width and
association breaks down. Raise stride for throughput only if you accept
degraded identity; the tactical geometry itself changes slowly enough that 5Hz
sampling costs almost nothing.

## 3. Team assignment

**Module**: [`offball/vision/teams.py`](../python/src/offball/vision/teams.py)

Crop the torso, discard pitch-green pixels, take the mean colour, cluster the
match's players into two groups, then hold assignment steady with a per-track
vote.

### Why clustering rather than a classifier

Kits change every match and there are only ever two of them. Any supervised
model needs retraining per fixture. Clustering needs no labels, and its failure
modes (similar kits, heavy shadow) are ones a supervised model would share.

### Design details that matter

**Torso crop, not the full box.** The middle 50% horizontally and 15-50%
vertically: shirt, not shorts, socks or turf.

**Green suppression.** Pixels where green clearly dominates both red and blue
are dropped before averaging. Without this, every player trends toward the
pitch colour and the two clusters collapse.

**Deterministic seeding.** 2-means is initialised from the two furthest-apart
samples, and the darker centroid is always cluster 0. A random init makes the
home/away labelling flip between runs on identical footage.

**Fit once, apply many.** Kit colours are learned from frames sampled across
the *whole* match, not the opening minutes. Lighting shifts substantially over
90 minutes and a profile fitted under sunlight degrades badly under
floodlights.

**Per-track voting with abstention.** A track stays `Team.UNKNOWN` until it has
accumulated `min_votes` observations. Abstaining is correct: a wrongly-teamed
player corrupts the offside line and the control field for every frame they
appear in.

### Known failures

| Failure | Handling |
| --- | --- |
| Goalkeepers wear a third kit | Handled positionally by `identify_goalkeepers`, not by colour |
| Referees wear a fourth | Separated as the small cluster far from both team centroids |
| Two teams in similar colours | **Not handled.** `KitProfile.separation` exposes the cluster distance; below ~40 the caller must fall back rather than trust the output |

## 4. Calibration

**Module**: [`offball/vision/calibration.py`](../python/src/offball/vision/calibration.py)

A pitch is flat, so one 3x3 homography per frame maps pixels to metres. No
camera intrinsics, no lens model.

The algebra is straightforward — normalised DLT, solved in
[`homography.rs`](../rust/offball-core/src/homography.rs). Robustness is the
hard part.

### Failure 1: pitch symmetry

A pitch has rotational and reflective symmetry. The two penalty boxes are
identical. A keypoint model will confidently label the wrong one some
percentage of the time, and **one such frame throws every player ~50m across
the pitch**.

Two defences:

- **RANSAC** catches the case where a minority of keypoints are wrong. Seeded
  deterministically, so the same footage always yields the same solution.
- **`HomographySmoother`** catches the case where the *whole frame* is
  mis-solved. It projects a fixed image point through each new homography and
  rejects solutions that move it more than `max_centre_jump` (default 15m)
  from the previous frame. A real camera pans smoothly; a 30m jump in 40ms is
  a mis-solve, not a camera move.

This second gate is the one that matters. A symmetry mis-solve is *internally
consistent* — every keypoint agrees, RANSAC finds no outliers, reprojection
error is near zero. Only temporal continuity reveals it.

### Failure 2: no pitch in view

Close-ups, replays and crowd shots contain no usable markings. Those frames
must be reported as uncalibrated, never guessed at. The smoother coasts on the
last good homography for up to `max_coast` frames — enough to cross a brief
loss of markings — then returns `None` and the frame goes unscored.

### Quality gates

A fit is rejected unless it clears all of: at least 4 correspondences,
`min_inliers` (default 5) in the consensus set, and mean inlier reprojection
error under `max_error` (default 3m).

### The missing piece

`KeypointSource` is a protocol. **No keypoint model ships here.** This is one
of the two gaps between the current state and end-to-end video analysis. See
[06 Roadmap](06-roadmap.md).

## 5. Projection and velocity

**Module**: [`offball/pipeline.py`](../python/src/offball/pipeline.py)

**Project the bottom-centre of the box, not the centre.** `BBox.ground_point`
is where the player meets the turf. Projecting the box centre puts every player
roughly a metre further from the camera than they are, and the error grows with
distance.

**Difference in pitch space, not image space.** A pixel means a different
number of metres at each end of the pitch; differencing pixels first produces
speeds that depend on where the player is standing.

The velocity window (default 5 frames ≈ 0.2s) is long enough to suppress
tracking jitter and short enough to register a sharp change of direction.

Points that project onto the horizon return `None` and are left unplaced,
rather than being clamped to a pitch corner.

## 6. Possession

Nearest player to the ball owns it — but naively applied, that rule switches
possession dozens of times during a single 50-50, and each switch flips the
direction every off-ball metric is computed in.

Two corrections:

- **Hysteresis**: a challenger must be nearer than the incumbent by
  `possession_hysteresis` (default 1.5m) before possession switches.
- **Loose-ball cutoff**: beyond `possession_max_distance` (default 4m) from
  every player, possession *holds its previous value* rather than clearing.
  A long pass between teammates should not blank out possession for its whole
  flight.

## Testing without weights

Because detection and keypoints are injected protocols, the entire pipeline is
tested end to end with scripted inputs:

- `ScriptedDetector` replays fixed detections.
- `ScriptedKeypoints` replays fixed correspondences, including `None` entries
  for frames with no pitch view.

[`test_pipeline.py`](../python/tests/test_pipeline.py) builds a scripted
attacking move, projects known pitch positions through a known camera
homography into image boxes, and asserts the pipeline recovers those positions,
keeps stable identities, estimates plausible speeds, and detects possession.

One caveat learned the hard way while writing those tests: scripted motion must
be expressed in metres **per second**, not as a fraction of the clip. Scaling
displacement by frame count makes short clips play at absurd speeds and breaks
IoU association — a property of the test data, not of the tracker.
