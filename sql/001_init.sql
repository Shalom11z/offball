-- offball: core schema.
--
-- Postgres 15+. Apply in filename order; each migration is idempotent enough
-- to re-run against a fresh database but is not written to be reversible —
-- roll forward, not back.
--
-- Sizing note that drives most of the design below: one match at 25fps with 22
-- players is ~3 million player-frame rows. Storing per-frame data for a season
-- of a single league is billions of rows. So:
--
--   * `frame_score` and `player_frame_score` are partitioned by match and are
--     the only tables that grow with frame count.
--   * Aggregates live in `player_match_summary` / `team_match_summary`, which
--     is what the API actually serves. Nothing user-facing queries the
--     per-frame tables directly.
--   * Raw tracking (bounding boxes, homographies) is *not* stored here. It
--     belongs in object storage as Parquet, keyed by `analysis_job.id`.

BEGIN;

CREATE EXTENSION IF NOT EXISTS "pgcrypto";  -- gen_random_uuid()

-- ---------------------------------------------------------------- reference

CREATE TABLE IF NOT EXISTS club (
    id          uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    name        text        NOT NULL,
    short_name  text,
    country     char(3),                      -- ISO 3166-1 alpha-3
    created_at  timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT club_name_not_blank CHECK (length(btrim(name)) > 0)
);

CREATE UNIQUE INDEX IF NOT EXISTS club_name_country_key
    ON club (lower(name), coalesce(country, ''));

CREATE TABLE IF NOT EXISTS player (
    id           uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    full_name    text        NOT NULL,
    known_as     text,
    birth_date   date,
    primary_foot text CHECK (primary_foot IN ('left', 'right', 'both')),
    created_at   timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT player_name_not_blank CHECK (length(btrim(full_name)) > 0)
);

CREATE TABLE IF NOT EXISTS venue (
    id            uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    name          text        NOT NULL,
    -- IFAB permits 90-120m by 45-90m. Pitch size is per-venue and materially
    -- changes every area metric, so it is stored, never assumed.
    pitch_length  numeric(5,2) NOT NULL DEFAULT 105.00
        CHECK (pitch_length BETWEEN 90 AND 120),
    pitch_width   numeric(5,2) NOT NULL DEFAULT 68.00
        CHECK (pitch_width BETWEEN 45 AND 90),
    created_at    timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS match (
    id            uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    external_ref  text,                       -- caller's own fixture id
    venue_id      uuid REFERENCES venue (id) ON DELETE SET NULL,
    home_club_id  uuid REFERENCES club (id)  ON DELETE SET NULL,
    away_club_id  uuid REFERENCES club (id)  ON DELETE SET NULL,
    kickoff_at    timestamptz,
    competition   text,
    season        text,
    created_at    timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT match_clubs_differ CHECK (home_club_id IS DISTINCT FROM away_club_id)
);

CREATE UNIQUE INDEX IF NOT EXISTS match_external_ref_key
    ON match (external_ref) WHERE external_ref IS NOT NULL;
CREATE INDEX IF NOT EXISTS match_kickoff_idx ON match (kickoff_at DESC);

-- Which direction each side attacks in each period. Without this the metrics
-- layer has to infer attacking direction from player positions, which is a
-- guess that fails on a deep block.
CREATE TABLE IF NOT EXISTS match_period (
    match_id            uuid    NOT NULL REFERENCES match (id) ON DELETE CASCADE,
    period              smallint NOT NULL CHECK (period BETWEEN 1 AND 5),
    started_at_seconds  numeric(8,2) NOT NULL,
    ended_at_seconds    numeric(8,2),
    -- true when the home side attacks toward increasing x in pitch coordinates
    home_attacks_positive_x boolean NOT NULL,
    PRIMARY KEY (match_id, period),
    CONSTRAINT period_ends_after_start
        CHECK (ended_at_seconds IS NULL OR ended_at_seconds > started_at_seconds)
);

-- ------------------------------------------------------------------- jobs

CREATE TYPE job_status AS ENUM ('queued', 'running', 'succeeded', 'failed');

CREATE TABLE IF NOT EXISTS analysis_job (
    id             uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    match_id       uuid REFERENCES match (id) ON DELETE CASCADE,
    status         job_status  NOT NULL DEFAULT 'queued',
    video_uri      text        NOT NULL,
    -- Where the raw per-frame tracking Parquet was written, if retained.
    tracking_uri   text,
    fps            numeric(5,2) NOT NULL DEFAULT 25.00 CHECK (fps > 0),
    stride         smallint    NOT NULL DEFAULT 1 CHECK (stride >= 1),
    progress       real        NOT NULL DEFAULT 0 CHECK (progress BETWEEN 0 AND 1),
    frames_total   integer     NOT NULL DEFAULT 0 CHECK (frames_total >= 0),
    frames_scored  integer     NOT NULL DEFAULT 0 CHECK (frames_scored >= 0),
    error          text,
    -- Pins the code that produced these numbers. Metric definitions change;
    -- without this you cannot tell whether a year-on-year difference is the
    -- player or the model.
    pipeline_version text     NOT NULL DEFAULT 'unknown',
    created_at     timestamptz NOT NULL DEFAULT now(),
    updated_at     timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT scored_within_total CHECK (frames_scored <= frames_total),
    CONSTRAINT failed_jobs_explain_themselves
        CHECK (status <> 'failed' OR error IS NOT NULL)
);

CREATE INDEX IF NOT EXISTS analysis_job_status_idx  ON analysis_job (status, created_at DESC);
CREATE INDEX IF NOT EXISTS analysis_job_match_idx   ON analysis_job (match_id);

-- Share of frames that could be scored. Below ~0.6, treat the run's numbers as
-- provisional. Generated rather than written, so it can never drift.
ALTER TABLE analysis_job
    ADD COLUMN IF NOT EXISTS coverage numeric(5,4)
    GENERATED ALWAYS AS (
        CASE WHEN frames_total > 0
             THEN round(frames_scored::numeric / frames_total, 4)
             ELSE 0 END
    ) STORED;

-- Maps a tracker's per-run integer id to a real player. Tracking cannot know
-- identities; this is populated by a separate (often manual) step, which is
-- why every metric table keys on track_id and joins through here.
CREATE TABLE IF NOT EXISTS track_identity (
    job_id      uuid     NOT NULL REFERENCES analysis_job (id) ON DELETE CASCADE,
    track_id    integer  NOT NULL,
    player_id   uuid     REFERENCES player (id) ON DELETE SET NULL,
    club_id     uuid     REFERENCES club (id)   ON DELETE SET NULL,
    shirt_number smallint CHECK (shirt_number BETWEEN 1 AND 99),
    side        text     NOT NULL DEFAULT 'unknown'
                CHECK (side IN ('home', 'away', 'referee', 'unknown')),
    is_goalkeeper boolean NOT NULL DEFAULT false,
    -- How the mapping was made, so low-confidence links can be filtered.
    confidence  real     CHECK (confidence BETWEEN 0 AND 1),
    PRIMARY KEY (job_id, track_id)
);

CREATE INDEX IF NOT EXISTS track_identity_player_idx ON track_identity (player_id);

COMMIT;
