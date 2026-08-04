# Architecture

## The shape of the problem

One 90-minute broadcast at 1080p25 is ~135,000 frames. Detecting and tracking
22 players in each produces ~3 million player-frame observations. The tactical
layer then evaluates a pitch-control grid per frame, which is where the compute
actually goes.

Three consequences drive every decision below:

1. **Analysis is a batch job, not a request.** Nothing user-facing can wait for
   it, so the API is job-based and the worker is separable.
2. **The numeric inner loop needs a compiled language.** Everything else does
   not.
3. **Per-frame data is too large to serve and too valuable to discard.** Hence
   the two-tier storage split in [04 Data model](04-data-model.md).

## Data flow

```
                    ┌──────────────┐
   match footage ──►│  Detection   │  players + ball, per frame
                    └──────┬───────┘
                           ▼
                    ┌──────────────┐
                    │   Tracking   │  stable identities across frames
                    └──────┬───────┘
                           ▼
                    ┌──────────────┐
                    │ Team assign  │  kit colour, per-track vote
                    └──────┬───────┘
                           ▼
                    ┌──────────────┐
   pitch keypoints ►│ Calibration  │  homography + temporal gate
                    └──────┬───────┘
                           ▼
                    ┌──────────────┐
                    │  Projection  │  image pixels ──► pitch metres
                    └──────┬───────┘
                           ▼
                    ┌──────────────┐
                    │  Velocity    │  finite differences in pitch space
                    └──────┬───────┘
                           ▼
                    ┌──────────────┐
                    │  Possession  │  nearest player, with hysteresis
                    └──────┬───────┘
                           ▼
                    ┌──────────────┐      ┌────────────────────┐
                    │   Scoring    │◄─────┤ Rust kernels       │
                    └──────┬───────┘      │ (or Python fallback)│
                           ▼              └────────────────────┘
                    ┌──────────────┐
                    │ Aggregation  │  match report
                    └──────────────┘
```

Stage order is not arbitrary:

- **Tracking before team assignment.** The per-track colour vote is what makes
  assignment stable; assigning per detection flickers.
- **Calibration before velocity.** Velocity must be computed in pitch metres.
  A pixel displacement means a different real distance at each end of the
  pitch, so differencing in image space produces speeds that vary with where
  the player is standing.
- **Possession before scoring.** Possession sets which direction "forward" is,
  and every off-ball metric is directional.

## Component boundaries

### Rust — `rust/offball-core`

Owns anything evaluated per player per grid cell: homography fitting and
projection, the pitch-control field, space ownership, and the geometric
primitives underneath them.

Exposed to Python through PyO3 behind an optional `python` feature, so
`cargo test` runs without a Python interpreter. The crate is also a normal Rust
library.

**The extension is optional.** `python/src/offball/kernels.py` dispatches to it
when present and falls back to a pure-Python implementation otherwise. The
Python version is the specification; if the two disagree, Rust is wrong.
[`test_parity.py`](../python/tests/test_parity.py) enforces this, and CI
asserts the compiled backend is actually loaded before running it — otherwise
the suite would skip itself green.

### Python — `python/`

Three layers, deliberately separable:

**`offball.kernels`, `offball.types`, `offball.tactics`** depend only on the
standard library. This means the metrics run against tracking data from any
provider, and `pip install offball` pulls nothing heavy. If you already have
Second Spectrum or StatsBomb tracking, this layer is usable on its own.

**`offball.vision`** holds detection, tracking, team assignment and
calibration. Heavy dependencies (OpenCV, ultralytics, torch) are imported
lazily *inside* the classes that need them, so importing the package is cheap
and works without the `vision` extra.

**`offball.api`** is a FastAPI job service. `offball.pipeline` orchestrates.

Vision components are **injected**, not constructed internally. That is what
lets the full pipeline run under test with scripted detections and keypoints —
no weights, no footage, no GPU — which is how
[`test_pipeline.py`](../python/tests/test_pipeline.py) exercises tracking,
calibration, projection, velocity and possession end to end.

### TypeScript — `ts/sdk`

A dependency-free client (platform `fetch` only). Beyond transport it holds
`analysis.ts`: sample-size gates, squad ranking, and pattern detectors like
"finds space but is never reachable".

These live client-side because they are *interpretation*, and interpretation
changes faster than measurement. What counts as a thin sample is a judgement a
UI should be able to revise without a server deploy. The API ships
measurements; the SDK interprets them. Nothing in it invents data.

### SQL — `sql/`

Postgres schema. Constraints encode the domain rather than living only in
application code: IFAB pitch bounds, `frames_scored <= frames_total`, failed
jobs must record an error, and `coverage` as a generated column so it cannot
drift from its inputs.

## Scaling

The current worker runs in a FastAPI background task. That is correct for one
node and wrong for anything larger. The intended shape:

```
  API (stateless, N replicas)
        │  enqueue
        ▼
  Job queue (Postgres SKIP LOCKED, or SQS)
        │
        ▼
  Workers (GPU, 1 match each)
        │
        ├──► Object storage: raw per-frame tracking as Parquet
        └──► Postgres: summaries only
```

**Parallelise per match, not per frame.** The per-frame loop has hard ordering
dependencies (tracking is sequential by nature; the calibration smoother is
stateful), and the Rust kernels already carry the numeric load. One worker per
match scales linearly with no coordination.

Where the time actually goes, per match, at stride 1:

| Stage | Share | Bound by |
| --- | --- | --- |
| Detection | ~70% | GPU |
| Pitch control | ~20% | CPU (Rust) |
| Tracking, calibration, aggregation | ~10% | CPU |

Detection dominates, so throughput work should start with batching frames
through the detector and raising `stride`, not with micro-optimising metrics.

## Failure philosophy

Two rules, applied consistently:

**Abstain rather than guess.** A frame that cannot be scored honestly produces
nothing. This appears as `None` returns from `score_frame`, `Team.UNKNOWN` as a
first-class value, `offside_line` returning `None` with fewer than two
defenders, and projection returning `None` for points on the horizon.

**Surface uncertainty with the result.** `coverage` on every report, `frames`
on every summary, `is_reliable` in the view and the SDK. The failure mode this
guards against is not a crash — it is someone confidently reading a number
derived from 40 frames of a badly-calibrated clip.

## Related documents

- [02 Vision pipeline](02-vision-pipeline.md) — stage internals and failure modes
- [03 Tactical metrics](03-tactical-metrics.md) — metric definitions and model limits
- [04 Data model](04-data-model.md) — storage tiers and retention
- [05 API](05-api.md) — endpoints and job lifecycle
- [06 Roadmap](06-roadmap.md) — what is missing
