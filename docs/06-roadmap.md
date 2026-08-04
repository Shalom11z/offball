# Roadmap

What is missing, in the order it should be built.

## Where things actually stand

| Component | State |
| --- | --- |
| Rust kernels | Complete — 39 tests, clippy clean |
| Tactical metrics | Complete — usable on tracking data from any provider |
| Tracking, calibration, projection, possession | Complete — tested end to end with scripted input |
| Team assignment | Implemented; unvalidated against real footage |
| Detection | Interface + Ultralytics wrapper. **No weights** |
| Pitch keypoints | Interface only. **No model** |
| HTTP API | Endpoints and lifecycle complete; worker not wired |
| TypeScript SDK | Complete — 29 tests |
| Postgres schema | Complete; not wired to the API |

**You cannot point this at an MP4 today and get a report.** Two missing models
stand between here and that, and both need annotated footage.

## Phase 1 — Close the loop

The goal is one real match, end to end. Nothing else matters until this works.

### 1.1 Pitch-keypoint model

**The critical path.** Without a homography there are no pitch coordinates, and
without pitch coordinates there are no metrics.

Needs a model that locates the named landmarks in
`Pitch::template_keypoints` — corners, halfway line, penalty areas, centre
circle, penalty spots — and plugs into the existing `KeypointSource` protocol.

A heatmap-regression architecture (HRNet-style) over ~2,000 annotated frames
spanning several stadiums and lighting conditions is the standard approach.
Segment the pitch lines and fit intersections as an alternative.

Everything downstream — RANSAC, the temporal continuity gate, the quality
thresholds — is already built and tested against synthetic correspondences.

### 1.2 Fine-tuned detector

COCO-pretrained YOLO finds players acceptably and the ball badly.

**Spend the annotation budget on the ball.** It is under 10 pixels across in a
wide shot, motion-blurs when struck, and possession assignment depends on it —
which in turn sets the direction every off-ball metric is computed in.

Also worth a distinct class for officials, which would remove the referee from
the kit-colour clustering problem entirely.

### 1.3 Validate on real footage

Everything above is tested against scripted inputs, which validates the *code*
and says nothing about *accuracy*. Needed:

- Tracking: ID switches per match against hand-labelled ground truth.
- Calibration: reprojection error distribution; how often the symmetry gate
  fires, and how often it should have.
- Team assignment: accuracy across kit combinations, including a same-colour
  fixture.
- Metrics: correlation with an existing tracking provider on a shared match.

Until this is done, no accuracy claim should be made about this system.

### 1.4 Wire the worker

Connect `run_analysis` to the pipeline and the Postgres `JobStore`, replacing
the in-memory store and the deliberate `NotImplementedError`. Populate
`progress` during the run.

## Phase 2 — Make the numbers trustworthy

### 2.1 A real expected-threat surface

`pitch_value` is currently an analytic function of distance and goal angle. It
gets the ordering right and the magnitudes mean nothing — and `dangerous_space`
inherits that.

Replace it with a surface fitted to real shot and goal outcomes. This is the
highest-value modelling work outstanding, because it converts "controlled
space" from a geometric curiosity into something connected to scoring.

### 2.2 Fit the control model

`reaction_time`, `max_speed` and `tau` are league-average literature values.
Fit them from tracking data, ideally per player. A 34-year-old centre-back and
a winger currently share a 7.8 m/s top speed.

Also worth adding: acceleration limits, and ball flight time weighting — the
two omissions that make the current model optimistic about distant space.

### 2.3 Appearance-based re-identification

The tracker loses identity through a corner-kick scrum. `Tracker.associate` is
the seam: add an appearance embedding and combine it with IoU.

This matters most for per-player attribution over a full match, which is the
thing a coach actually wants.

### 2.4 Period and direction metadata

`match_period.home_attacks_positive_x` exists in the schema but the Python
layer still infers attacking direction from player centroids. That inference
fails against a deep block, where both teams sit in the same half. Feed the
column in instead.

## Phase 3 — Product

### 3.1 Clip export

A number is an argument; a clip is evidence. Given a `player_frame_score` row,
cut the corresponding seconds of footage with an overlay. The per-frame tier
exists precisely to make this possible.

### 3.2 Visualisation

Pitch-control heatmaps, space-ownership overlays, off-ball run trails. The
control grid is already computed and thrown away per frame; persisting a
downsampled version would make this cheap.

### 3.3 Natural-language feedback

The stated goal is *automated tactical feedback*. Turning "held a −0.4m median
offside margin across 12 minutes of possession" into a coaching point is a
distinct layer, and deliberately not built yet — it should sit on top of
measurements that have been validated, not before.

### 3.4 Cross-match aggregation

Player profiles across a season, with the caveat that comparisons are only
valid within a `pipeline_version`. The schema already pins this per job.

## Explicitly out of scope

Worth stating so the boundaries are clear:

- **Event detection** (passes, shots, tackles). Well-served by existing
  providers, and orthogonal to off-ball positioning.
- **Real-time analysis.** Everything here assumes batch. Live would require
  rethinking the calibration smoother and the possession hysteresis, both of
  which look backwards.
- **Multi-camera / tracking-system input.** The whole vision stage assumes a
  single broadcast feed. If you have Second Spectrum data, skip to
  `offball.tactics` — it works on any tracking source.
- **Player identification from footage.** `track_identity` assumes an external
  (often manual) mapping. Jersey-number OCR is a genuinely separate project.

## Contributing

The most useful contributions right now, in order:

1. Annotated pitch-keypoint data.
2. Ball-detection annotations.
3. Any match with both footage and third-party tracking data, for validation.

Code contributions should keep the two invariants the codebase is built on:
**abstain rather than guess**, and **carry uncertainty alongside every number**.
