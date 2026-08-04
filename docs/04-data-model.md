# Data model

**Schema**: [`sql/001_init.sql`](../sql/001_init.sql),
[`sql/002_metrics.sql`](../sql/002_metrics.sql)

## The sizing problem

One match at 25fps with 22 players is ~3 million player-frame rows. A season of
a single 20-team league is ~1.1 billion. Storing that in the same table the API
queries would be a mistake in both directions: too slow to serve, too expensive
to keep hot.

Hence three tiers.

| Tier | Contents | Store | Grows with |
| --- | --- | --- | --- |
| Raw tracking | Bounding boxes, homographies, per-frame detections | Object storage (Parquet) | Frames |
| Per-frame metrics | `frame_score`, `player_frame_score` | Postgres, partitioned | Frames |
| Summaries | `player_match_summary`, `team_match_summary` | Postgres | Matches |

**The API only ever reads the summary tier.** Per-frame tables exist for batch
analysis, model iteration and clip export.

**Raw tracking deliberately does not live in Postgres.** It is columnar,
append-only, never queried transactionally, and an order of magnitude larger
than the metrics. Parquet in object storage, keyed by `analysis_job.id`, with
the location recorded in `analysis_job.tracking_uri`.

## Reference entities

`club`, `player`, `venue`, `match`, `match_period`.

Two worth calling out:

**`venue.pitch_length` / `pitch_width`.** IFAB permits 90-120m by 45-90m. Pitch
size is per-venue and materially changes every area metric, so it is stored,
never assumed. A `space_owned` figure from a 100x64 pitch is not comparable to
one from 105x68 without knowing both.

**`match_period.home_attacks_positive_x`.** Which way each side is attacking,
per period. Without this the metrics layer must infer direction from player
positions — a guess that fails on a deep block, where both teams' centroids sit
in the same half. The Python layer currently infers it; this column is how it
should be told instead.

## Jobs

`analysis_job` is the unit of work and the unit of provenance.

```sql
coverage numeric(5,4) GENERATED ALWAYS AS (
    CASE WHEN frames_total > 0
         THEN round(frames_scored::numeric / frames_total, 4)
         ELSE 0 END
) STORED
```

Generated, not written, so it cannot drift from its inputs. Coverage is the
headline health metric for a run; below ~0.6 the vision stage struggled.

`pipeline_version` pins the code that produced the numbers. **Metric
definitions will change.** Without this you cannot tell whether a year-on-year
difference is the player or the model — which is the single most expensive
mistake an analytics platform can make.

Constraints encode invariants the application should not be trusted to
maintain alone:

```sql
CONSTRAINT scored_within_total CHECK (frames_scored <= frames_total)
CONSTRAINT failed_jobs_explain_themselves
    CHECK (status <> 'failed' OR error IS NOT NULL)
```

## Identity

`track_identity` maps a tracker's per-run integer `track_id` to a real player.

This table exists because **tracking cannot know identities**. It produces
stable-ish integers within one run; nothing more. Mapping those to squad
numbers is a separate, often manual, step. That is why every metric table keys
on `(job_id, track_id)` and joins through here rather than storing `player_id`
directly.

`confidence` on the mapping lets low-confidence links be filtered rather than
silently trusted.

## Per-frame tables

Both are `PARTITION BY HASH (match_id)`.

Hash rather than range on time, because a match's frames are always queried
together and never across matches. Partitioning by match means re-analysing a
fixture is a `DETACH`, not a mass `DELETE`.

**Eight partitions is a starting point, not a capacity plan.** The count cannot
be changed without rewriting the table, so revisit it before the first
production load. As a rough guide, target partitions of 50-100M rows: at ~3M
player-frame rows per match, 8 partitions holds a few hundred matches
comfortably and should be raised well before a full season of a major league.

Sign conventions are recorded as column comments, because they are the easiest
thing to get wrong when querying this directly:

```sql
COMMENT ON COLUMN player_frame_score.offside_margin IS
    'Metres beyond the offside line: positive is an offside position, negative is
     onside. A small negative value is a player holding the shoulder of the last
     defender.';
```

`lane_clearance` stores `NULL` rather than a sentinel when no defenders were
tracked — infinity is not a distance.

## Summaries

`player_match_summary` and `team_match_summary` mirror the dataclasses in
[`report.py`](../python/src/offball/tactics/report.py). Medians rather than
means, for the reasons in [03 Tactical metrics](03-tactical-metrics.md).

`frames` is `CHECK (frames > 0)` — a summary over zero frames is not a summary.

## The `player_report` view

What the API serves. It joins identity in and, critically, carries the run's
coverage on every row:

```sql
(s.frames >= 500 AND j.coverage >= 0.6) AS is_reliable
```

The thresholds mirror `MIN_RELIABLE_FRAMES` and `MIN_RELIABLE_COVERAGE` in the
TypeScript SDK. Duplicating them is a deliberate trade: it means a consumer
reading the database directly gets the same warning as one going through the
SDK. If you change one, change both.

The design intent is that **a consumer cannot read a player's figures without
also being handed the reason to distrust them.**

## Retention

Not yet implemented; the shape it should take:

| Tier | Suggested retention | Rationale |
| --- | --- | --- |
| Summaries | Indefinite | Small, and the product |
| Per-frame metrics | 1 season hot, then cold storage | Needed for model iteration and clip export |
| Raw tracking Parquet | Indefinite in cold storage | Expensive to regenerate — needs the original footage and GPU time |
| Source footage | Per rights agreement | Usually not ours to keep |

Re-analysing a match with a newer pipeline should write a **new**
`analysis_job` rather than updating the old one. Two jobs for the same match
with different `pipeline_version` values is the point: it is how model changes
become measurable instead of invisible.

## Migrations

Applied in filename order. Each is written to be safe to re-run — CI applies
every migration twice against a fresh Postgres 16 to prove it.

This is why `CREATE TYPE job_status` is wrapped in an exception-handling
`DO` block: `CREATE TYPE` has no `IF NOT EXISTS`, and the naive version fails
the second time.

Migrations roll **forward**, not back. There are no down-migrations; a mistake
is corrected by a new migration.
