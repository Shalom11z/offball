# offball

Automated off-the-ball positioning analysis from raw soccer match footage.

Most football analytics measures what happens to the ball. This measures what
the other twenty-one players are doing — the runs that pull a centre-back out
of position, the striker who holds the shoulder of the last defender, the
winger who finds space nobody can pass to.

> **Status: early alpha.** The tactical layer is implemented, tested and
> usable today against tracking data from any source. The vision stage is a
> working skeleton with pluggable model seams — no detector weights or
> pitch-keypoint model ship with this repository. See
> [What works today](#what-works-today).

## Why four languages

Each does a job the others do badly. This is not polyglot for its own sake.

| Language | Responsibility | Why |
| --- | --- | --- |
| **Rust** (`rust/offball-core`) | Geometry, homography, pitch control | The control grid is ~200M cell-player evaluations for one match. This is the only part where the constant factor decides whether analysis takes minutes or hours. |
| **Python** (`python/`) | CV pipeline, tactics, HTTP API | Where the CV and ML ecosystem lives. Orchestration and metrics are I/O- and logic-bound, not compute-bound. |
| **TypeScript** (`ts/sdk`) | Client SDK and report interpretation | Sample-size gates and ranking are presentation decisions a UI should tune without a server deploy. |
| **SQL** (`sql/`) | Postgres schema | Constraints belong where the data lives. IFAB pitch bounds, "failed jobs must record an error", and coverage-as-a-generated-column cannot drift if the database enforces them. |

The Rust kernels are **optional**. Every one has a pure-Python reference
implementation, and [`test_parity.py`](python/tests/test_parity.py) asserts the
two agree to 1e-9. Without the compiled extension the package is slower and
otherwise identical.

## Quick start

```bash
git clone https://github.com/Shalom11z/offball.git
cd offball/python
pip install -e '.[api,dev]'
offball demo
```

`offball demo` runs a scripted attacking move through the real metrics and
prints a report. No weights, no footage, no GPU.

Build the compiled kernels (optional, ~100x faster on the control grid):

```bash
cd rust/offball-core && maturin develop --release
```

Check which backend is active:

```bash
offball info
```

## What it measures

All metrics are computed in pitch metres after the frame homography, with the
attacking side normalised to play toward +x.

**Per player, per frame**

| Metric | Meaning |
| --- | --- |
| Space owned | Pitch area (m²) this player reaches before anyone else — a velocity-aware Voronoi cell, not a static one |
| Offside margin | Signed metres from the offside line. `-0.3` is a player timing runs finely; `-6` is one dropping off |
| Passing lane | Whether the carrier has a clear line, and the nearest defender's distance from it |
| Lines broken | How many opposition banks the player is positioned beyond |
| Marking pressure | Nearest-opponent proximity, 0-1 |
| Position value | Threat value of the ground occupied |

**Per team, per frame**: controlled space, threat-weighted "dangerous" space,
block shape (convex-hull area, depth, width), fitted defensive banks, and the
count of teammates offering a viable pass.

Full definitions, model assumptions and known limitations:
[`docs/03-tactical-metrics.md`](docs/03-tactical-metrics.md).

## Design decisions that shape everything else

**Missing data abstains.** A frame with no calibration, no ball, or fewer than
two tracked defenders produces *no score*, not a default one. Averaging
fabricated zeros into a match report is worse than a smaller sample. Every
summary carries its own `frames` count so consumers can discard thin samples.

**Sample size and coverage travel with the numbers.** `coverage` — the share of
frames that could be scored — is on every report, and `is_reliable` is exposed
in the SQL view and the TS SDK. A consumer should not be able to read a
player's figures without also being handed the reason to distrust them.

**Determinism.** RANSAC is driven by a seeded xorshift and k-means is
quantile-initialised, so re-running the same footage produces the same report.
Without this you cannot tell a model change from a reroll.

**Pipeline version is pinned per job.** Metric definitions will change. Without
recording which code produced a number, you cannot tell whether a year-on-year
difference is the player or the model.

## What works today

| Component | State |
| --- | --- |
| Rust kernels | Complete. 39 tests, clippy clean |
| Tactical metrics | Complete. Usable on tracking data from any provider |
| Tracking, calibration, possession | Complete. Tested end to end with scripted input |
| Team assignment | Implemented; needs real footage to validate |
| Detection | Interface + Ultralytics wrapper. **No weights ship here** |
| Pitch keypoints | Interface only. **No model ships here** |
| HTTP API | Endpoints and job lifecycle complete; the worker fails loudly rather than returning a fabricated report until a detector is configured |
| TypeScript SDK | Complete. 29 tests |
| Postgres schema | Complete, not yet wired to the API |

The honest summary: **you cannot point this at an MP4 today and get a report.**
The two missing pieces are a fine-tuned detector and a pitch-keypoint model,
both of which need annotated footage. Everything downstream of them is built
and tested. See [`docs/06-roadmap.md`](docs/06-roadmap.md).

## Repository layout

```
rust/offball-core/   Compiled kernels (optional, with PyO3 bindings)
python/              CV pipeline, tactics, FastAPI service, CLI
ts/sdk/              TypeScript client and report interpretation
sql/                 Postgres migrations
docs/                Architecture, pipeline, metrics, data model, API, roadmap
```

## Documentation

| Document | Covers |
| --- | --- |
| [01 Architecture](docs/01-architecture.md) | Component boundaries, data flow, scaling |
| [02 Vision pipeline](docs/02-vision-pipeline.md) | Detection, tracking, teams, calibration, and their failure modes |
| [03 Tactical metrics](docs/03-tactical-metrics.md) | Every metric, its model, and what it cannot tell you |
| [04 Data model](docs/04-data-model.md) | Storage tiers, partitioning, retention |
| [05 API](docs/05-api.md) | Endpoints, job lifecycle, error semantics |
| [06 Roadmap](docs/06-roadmap.md) | What is missing and in what order |

## Development

```bash
# Rust
cd rust/offball-core && cargo test && cargo clippy --all-targets

# Python
cd python && pytest -q && ruff check src tests

# TypeScript
cd ts/sdk && npm install && npm test && npm run typecheck
```

CI runs all of the above plus a Rust/Python parity job and a Postgres
migration-idempotency check.

## Licence

MIT. See [LICENSE](LICENSE).
