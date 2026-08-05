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

### The classical detector

`KeypointSource` is a protocol, and
[`ClassicalKeypointSource`](../python/src/offball/vision/lines.py) implements it
with no trained model at all.

**Why it matches lines rather than corners.** A homography maps lines to lines,
so the image of the intersection of two pitch lines is exactly the intersection
of their images — whether or not paint exists at that point. That matters
twice over: a line is supported by thousands of pixels where a corner has a
handful, and most useful intersections (the halfway line crossed with a
penalty-area edge) are never painted at all.

The stages:

1. Mask the pitch — green in HSV *and* green-dominant in BGR, opened before
   closing. The opening is load-bearing: without it, green-ish speckle from a
   crowd survives and the close welds it into a solid false "pitch".
2. Isolate paint with a morphological top-hat, after a Gaussian blur. The blur
   is also load-bearing: raw sensor noise survives the morphology as speckle,
   and Hough will happily find strong lines through it.
3. Hough for infinite lines, merging duplicates. Duplicates are compared by
   **distance sampled across the image**, not by rho — rho is measured from the
   origin, so a fraction of a degree of theta moves it by tens of pixels and
   genuine duplicates fail to merge.
4. Split into two families by angle, then order each spatially. Ordering needs
   the normals aligned first: a family straddling the 0/pi seam contains lines
   that are nearly parallel but whose normals point opposite ways, which makes
   their signed offsets meaningless.
5. Search order-preserving assignments to the template, fit each by
   least-squares DLT, and score by how much of the projected template lands on
   real line pixels.

The ordering in step 4 is what makes step 5 tractable: a projective view of a
plane preserves the order of a pencil of parallel lines, so only monotonic
assignments need considering.

Scoring against the dense line mask rather than the fitted points is what makes
it robust — a wrong assignment fits its own four points perfectly and still puts
the rest of the pitch nowhere near any paint.

### The symmetry that cannot be resolved from one frame

The pitch markings are **exactly invariant** under `x -> length - x` and
`y -> width - y`. `TEMPLATE_X` maps 0↔105, 5.5↔99.5, 16.5↔88.5 with 52.5 fixed;
`TEMPLATE_Y` mirrors the same way. A homography absorbs the resulting rotation
perfectly, so every member of that symmetry group explains the image equally
well.

**No amount of image processing removes this.** It is a property of the pitch.
It is resolved with outside information: the previous frame's homography (which
the source keeps automatically, and `set_prior` sets explicitly), or the known
camera side and the period's attacking direction.

`last_ambiguous` reports when several candidates tied, which is worth logging
when calibration goes wrong.

### What has actually been verified

The detector is tested by **rendering a pitch through a known homography and
checking it is recovered** — clean, with noise, and with lines missing. That is
a real end-to-end check of the geometry, and it is only possible because the
answer is known ground truth.

Synthetic frames are clean by construction. Passing those tests means the
geometry is right; it does **not** mean the detector works on broadcast video.

### Measured on real broadcast footage: it does not work

**Man Utd v Arsenal, full first half (SoccerNet, 720p25). Calibration rate:
0 of 120 frames sampled across the half. Only 3% of frames produced any
correspondences at all.**

This is structural, not a threshold in need of tuning. Every stage before
matching behaves: the pitch mask finds 74-98% green, and Hough returns 6-16
lines per frame. The problem is *what those lines are*.

A standard broadcast centre view contains exactly **one** straight pitch line —
the halfway line — plus the centre circle. The near touchline is out of frame,
the far one is lost against the advertising hoardings, and neither penalty area
is visible. The matcher needs at least two lines in each of the two pitch
directions (see `_candidates`), and that information is not in the picture.

Worse, most lines Hough returns are artefacts: **chords of the centre circle**,
which a straight-line transform fragments into a dozen strong votes, plus edges
from hoardings and mown stripes. On the frame inspected, 15 of 16 detected lines
were spurious; only the halfway line was real.

Two ways forward, neither of them more tuning:

1. **Fit the centre circle as an ellipse.** A circle maps to an ellipse under a
   homography; its known 9.15m radius, plus the fact that its centre lies on the
   halfway line, adds enough constraint to solve exactly the shot that dominates
   broadcast footage.
2. **Train the learned keypoint model** ([06 Roadmap](06-roadmap.md)). This is
   what the field does, and the measurement above is the evidence for why.

The classical detector is kept: it is correct where it applies, costs nothing,
and is an honest baseline to measure a learned model against. It should not be
relied on for broadcast footage.

### Measured against ground truth: the camera fit is NOT accurate

The SoccerNet-Calibration split (3141 annotated broadcast frames) provides a
ground-truth homography per image. It is **openly downloadable via HuggingFace**
and needs no NDA — the video NDA is a separate thing, and the older ownCloud
route is what rejects the video password.

Running the camera calibrator against it, restricted to the centre views it is
designed for:

===================  ==========  ============  =========
Configuration        Calibrated  Median error  Under 5 m
===================  ==========  ============  =========
Camera fit                  83%       51.3 m         0%
+ support guard              6%       34.4 m         0%
===================  ==========  ============  =========

**51 metres is about half a pitch.** The calibrator is not measuring the pitch;
it is producing a confident, self-consistent, wrong answer on most frames.

This matters beyond the number. Every earlier quality claim about this
calibrator rested on two things — a low fit residual and overlays that put the
template neatly on the paint — and **neither detects this failure**. The
residual only says the camera explains the three cues it was handed. An overlay
can look perfect for a homography that is badly wrong, because the projected
template lands on paint either way. Ground truth caught in one run what visual
inspection structurally cannot.

### Causes found, using ground truth as an oracle

Guessing at whole hypotheses failed three times. What worked was projecting each
*evidence source* through the known-correct ground-truth homography and asking
what it really is. That located two concrete bugs immediately:

**1. `cv2.fitEllipse` does not sort its axes.** It returns `(width, height)` of
the rotated box, in that order. The code treated the first as the major axis, so
size and eccentricity filters compared the wrong ones and rejected legitimate
centre circles — real frames returned axes like `(90, 489)`. Fixed; on a sample
frame the detected circle now measures **9.10m radius against a true 9.15m**.

**2. The "halfway line" was frequently a penalty-area line.** The heuristic took
the strongest steeply-angled Hough line. Projected through ground truth, that
line landed at **x = 16.3 and x = 88.6** — the penalty-area lines — while the
solver was told it was `x = 52.5`. A 36m lie fed straight into the fit, which is
the right order of magnitude for the observed error. The halfway line is now
only accepted if it bisects a detected centre circle, and the frame abstains
when no circle corroborates it.

A third convention issue was confirmed but is *not* the dominant term: the
pitch-mask boundary corresponds to `y = 0` in some frames and `y = width` in
others, depending which side the camera is on. Both hypotheses are now fitted
and the better kept; this did not move the median.

### Root cause: the evidence is insufficient, not just noisy

Applying the oracle to the frames that still produced a fit settled it. Three of
them — 00060, 00153, 00301 — had **evidence verified correct against ground
truth**: detected circle radius 9.1-9.4m against a true 9.15m, touchline landing
on a real touchline. Their fits were still 43-60m wrong.

Correct inputs, wrong output. That moves the fault out of the evidence and into
the solve.

Multi-start optimisation confirmed it from the other direction: trying 28
starting points *lowered the residual* and raised the median error from 50m to
80m. A better-fitting camera was not a more correct one.

The reason is degrees of freedom. A circle supplies 5 independent constraints
and a touchline 2 — exactly the camera's 7. Exactly determined means **no
redundancy**, so noisy evidence admits many cameras that fit equally well and
are geometrically wrong, and the residual cannot tell them apart.

**The halfway line is what makes the system over-determined, and it is the piece
that cannot yet be identified reliably** — the strongest steep line in a frame
is frequently a penalty-area edge. That is the thing to fix, and until it is
fixed no amount of solver work will help.

### Halfway-line identification: improved, still not sufficient

The line is now accepted only when it bisects a detected centre circle, which is
the property that actually defines it. The frame abstains when no circle
corroborates a candidate, because with nothing to check against there is no way
to tell the halfway line from a penalty-area edge.

Two approaches were tried and one was reverted:

* **Joint identification** — requiring a valid (ellipse, line) pair before
  accepting either. Correct in principle and it detected *nothing*: the line
  erasure that precedes ellipse fitting fragments the circle into arcs, so the
  surviving contour's centre rarely sits within the threshold of the true line.
  Reverted.
* **Independent circle, then bisection check** — kept. Accept rate 39%,
  median error still ~49m.

The remaining barrier is the contour-based circle finder. Straight lines must be
erased before fitting an ellipse (or the circle merges with the halfway line
into one unfittable blob), but that same erasure splits the circle into arcs
whose closed contours are thin slivers and fail the eccentricity test.
Morphological closing to rejoin them merges the whole pitch outline instead.

**Conic fitting is implemented and verified** (`fit_conic`,
`conic_ellipse_params`): five points on an ellipse recover its centre and both
semi-axes exactly, at any coordinate scale. Two bugs had to be fixed to get
there, and both returned "not an ellipse" for a perfect one:

* the design matrix was unnormalised — the `x²` column is ~1e5 against a
  constant column of 1, so the SVD null vector followed scale rather than
  geometry. It needs the same Hartley normalisation the homography solver
  already uses;
* the semi-axis closed form divides the whole root by the discriminant and
  *multiplies* by the bracket; dividing by the bracket yields nothing.

Driving it with RANSAC over the line mask is **not** yet usable. Maximising
inlier count selects the wrong ellipse: large 31-46m curves threaded through
scattered pixels beat the real 9.15m circle. A size prior, or scoring against
the expected pitch geometry rather than raw inlier count, is what that needs.
The contour-based path is retained in the meantime.

The original reasoning still holds — it needs no erasure and tolerates the halfway line
crossing the circle. That is the next concrete step and the last known blocker
on this calibrator.

### Still unresolved

After both fixes the accept rate falls from 83% to 37% — fewer confidently wrong
answers — but the median error on those remains **~50m**. Something further is
wrong and is not yet identified. The fixes above are verified individually
against the oracle and are worth keeping regardless; the calibrator is still not
safe to treat as a measurement.

The earlier hypotheses, all tested and rejected before the oracle approach:

1. **A penalty arc mistaken for the centre circle.** Plausible — the two are
   similar ellipses and the shift is about the right size. Requiring the
   ellipse centre to lie on the halfway line, which is true of the centre circle
   and not of an arc, did not change the median.
2. **The pitch's mirror symmetry.** 51m is suspiciously close to half of 105m.
   Scoring all four symmetry variants against the paint and taking the best did
   not change the median.
3. **Unverified fits.** Adding a support check against the whole projected
   template cut the accept rate from 83% to 6% without improving accuracy — it
   rejects good and bad alike.

The cause is still unknown. The finding stands on its own and is more useful
than the fix would have been on its own: **do not treat these coordinates as
measurements until this is resolved.**

### The fix: fit a camera, not a homography

`offball.vision.broadcast` solves the same footage by taking option 1 above.
**Measured on the same Premier League half: 45-50% of frames calibrate, median
residual 0.27m**, against 0% for line matching.

Three sources of evidence, each supplying what the others cannot:

| Source | Where it comes from | Why it matters |
| --- | --- | --- |
| Far touchline | The **pitch mask boundary**, where grass meets the hoardings | Nothing is painted there, so Hough can never see it — yet it is present in ~80% of frames |
| Halfway line | Paint | The one straight line the centre view reliably shows |
| Centre circle | Fitted as an **ellipse** | Its 9.15m radius is a strong constraint; straight-line Hough actively destroys it |

Two decisions carry the result:

**Fit a camera, not a free homography.** The evidence supplies around nine
constraints, but they are badly conditioned for an unconstrained 8-DOF fit,
which wanders into projectively valid nonsense. A physical camera has 7 DOF and,
crucially, cannot be underground or have negative focal length. Bounds on those
parameters are what keep the solution honest.

**Demand that a contour genuinely *is* an ellipse.** Requiring 65% of contour
points to lie within 3px of the fitted ellipse is what separates a real centre
circle from an arbitrary blob. Adding this single check moved the calibration
rate from 10% to ~50%; without it the fit is fed hundreds of junk points.

Residuals are expressed in **metres** by un-projecting detected image points
onto the pitch — circle points should sit 9.15m from the centre spot, touchline
points on `y = width` — so no arbitrary weighting between the three terms is
needed.

Verified visually as well as numerically: overlays of fitted frames put the
projected circle and halfway line on the paint. Note the residual is a measure
of self-consistency, not accuracy; there is still visible error of around a
metre at the edges, and the remaining ~50% of frames (replays, close-ups, goal-
mouth shots) are not solved by this at all.

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
