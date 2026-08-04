# HTTP API

**Implementation**: [`offball/api/`](../python/src/offball/api/) ·
**Client**: [`ts/sdk`](../ts/sdk)

## Why it is job-based

A full match takes minutes to hours depending on stride and hardware. No
request can wait for that, so the API accepts work, returns immediately, and
the caller polls.

```bash
cd python && offball serve
# http://127.0.0.1:8000/docs
```

## Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/healthz` | Liveness, plus which kernel backend is compiled in |
| `POST` | `/v1/analyses` | Submit footage → `202` with a job id |
| `GET` | `/v1/analyses` | Recent jobs, newest first |
| `GET` | `/v1/analyses/{id}` | Job status |
| `GET` | `/v1/analyses/{id}/report` | The finished report |

### Submit

```bash
curl -X POST http://localhost:8000/v1/analyses \
  -H 'Content-Type: application/json' \
  -d '{
        "video_uri": "s3://bucket/match.mp4",
        "match_id": "2026-08-04-ARS-CHE",
        "fps": 25,
        "stride": 1,
        "pitch_length": 105,
        "pitch_width": 68
      }'
```

Returns `202 Accepted`:

```json
{
  "job_id": "8f14e45fceea167a5a36dedd4bea2543",
  "status": "queued",
  "match_id": "2026-08-04-ARS-CHE",
  "created_at": "2026-08-04T10:00:00Z",
  "updated_at": "2026-08-04T10:00:00Z",
  "progress": 0.0,
  "error": null
}
```

Validation is enforced at the schema layer: `fps` must be positive, `stride`
at least 1, and pitch dimensions must fall inside IFAB limits (90-120m by
45-90m). Invalid requests get `422` with the offending field named.

### Fetch the report

```bash
curl http://localhost:8000/v1/analyses/{job_id}/report
```

```json
{
  "job_id": "8f14e45f...",
  "match_id": "2026-08-04-ARS-CHE",
  "frames_scored": 98234,
  "frames_total": 135000,
  "coverage": 0.7277,
  "teams": [
    {
      "team": "home",
      "frames": 52104,
      "duration": 2084.16,
      "median_controlled_space": 3612.4,
      "median_dangerous_space": 541.9,
      "median_attacking_hull": 1804.2,
      "median_defending_hull": 1502.7,
      "mean_passing_options": 3.42
    }
  ],
  "players": [
    {
      "track_id": 7,
      "frames": 41230,
      "duration": 1649.2,
      "median_space_owned": 421.5,
      "median_position_value": 0.31,
      "availability_rate": 0.62,
      "offside_rate": 0.04,
      "median_offside_margin": -0.8,
      "mean_lines_broken": 1.9,
      "median_separation": 6.2,
      "mean_pressure": 0.22
    }
  ]
}
```

**Read `coverage` before anything else.** Below ~0.6 the vision stage
struggled and every figure below it is provisional.

## Status codes

| Code | Meaning |
| --- | --- |
| `202` | Job accepted |
| `200` | Success |
| `404` | No such job |
| `409` | Job exists but has no report yet (queued, running, or failed) |
| `422` | Request failed validation |

The `404`/`409` split is deliberate and load-bearing: a caller polling for a
report must be able to distinguish **"not ready yet"** from **"this job never
existed"**. Collapsing both into `404` makes a polling client unable to tell a
typo from a slow job.

## Job lifecycle

```
queued ──► running ──┬──► succeeded    report available
                     └──► failed       error is populated
```

The database enforces that a failed job explains itself:

```sql
CONSTRAINT failed_jobs_explain_themselves
    CHECK (status <> 'failed' OR error IS NOT NULL)
```

### Jobs currently fail

With no detector configured, `run_analysis` raises and the job transitions to
`failed` with a clear message:

> No detector is configured. Set `OFFBALL_DETECTOR_WEIGHTS` and install the
> 'vision' extra to enable video analysis.

This is deliberate. The alternative — returning an empty or synthetic report —
would be a fabricated result presented as a real one. A test asserts this
behaviour explicitly.

## Health

```bash
curl http://localhost:8000/healthz
```

```json
{ "status": "ok", "version": "0.1.0", "kernel_backend": "rust" }
```

`kernel_backend` is `"rust"` or `"python"`. Worth alerting on: a deployment
that silently fell back to the pure-Python kernels will be dramatically slower
while producing identical numbers, so it will not show up as an error anywhere
else.

## TypeScript client

```ts
import { OffballClient, assessQuality, rankBy } from "@offball/sdk";

const client = new OffballClient({ baseUrl: "https://api.example.com" });

const report = await client.analyseAndWait(
  { videoUri: "s3://bucket/match.mp4", matchId: "2026-08-04-ARS-CHE" },
  { pollIntervalMs: 10_000, onProgress: (j) => console.log(j.status, j.progress) },
);

const { confidence, warnings } = assessQuality(report);
if (confidence === "low") console.warn(warnings.join("\n"));

for (const p of rankBy(report, "medianSpaceOwned", { limit: 5 })) {
  console.log(`#${p.trackId}: ${p.medianSpaceOwned.toFixed(0)} m²`);
}
```

The SDK converts `snake_case` to `camelCase` at the boundary and revives
timestamps as `Date`. Errors are typed: `OffballApiError` carries `status` and
an `isRetryable` flag; `OffballTimeoutError` states explicitly that the job is
still running server-side and gives the id to keep polling.

## Deployment shape

`app.py` runs the worker in a FastAPI background task. That is correct for a
single node and wrong for anything larger — the process holds the job state and
the GPU work in the same place as the HTTP handlers.

`JobStore` is the seam. It is a `Protocol` with an in-memory implementation;
the production path is a Postgres-backed store against
[`sql/001_init.sql`](../sql/001_init.sql), with workers pulling via
`SELECT ... FOR UPDATE SKIP LOCKED`. Nothing in the API layer assumes storage
is in-process. See [01 Architecture](01-architecture.md#scaling).

## Not yet implemented

- **Authentication.** The SDK sends `Authorization: Bearer` when given a token;
  the server does not check it.
- **Rate limiting.**
- **Progress reporting.** `progress` exists on the schema but the worker does
  not update it during a run.
- **Webhooks.** Polling is the only completion signal.
- **Pagination.** `GET /v1/analyses` takes a `limit` but has no cursor.

## OpenAPI

Served at `/openapi.json`, with Swagger UI at `/docs`. A test asserts the
schema stays generatable, since `ts/sdk/src/types.ts` is meant to be
regenerated from it rather than hand-edited.
