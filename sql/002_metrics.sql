-- offball: metric storage.
--
-- Two tiers, for the reason set out in 001_init.sql:
--
--   * per-frame tables, partitioned, written once by the worker and read only
--     by batch jobs and clip export;
--   * summary tables, tiny, and the only thing the API reads.
--
-- Units are metres, seconds and square metres throughout. Column comments
-- record the sign conventions, which are the easiest thing to get wrong when
-- querying this directly.

BEGIN;

-- --------------------------------------------------------------- per frame

-- Partitioned by match: a match's frames are always queried together, and
-- dropping a re-analysed match becomes a DETACH rather than a mass DELETE.
CREATE TABLE IF NOT EXISTS frame_score (
    job_id              uuid        NOT NULL REFERENCES analysis_job (id) ON DELETE CASCADE,
    match_id            uuid        NOT NULL,
    frame_index         integer     NOT NULL CHECK (frame_index >= 0),
    timestamp_seconds   numeric(9,3) NOT NULL,
    period              smallint,
    attacking_side      text        NOT NULL CHECK (attacking_side IN ('home', 'away')),

    team_space              real NOT NULL CHECK (team_space >= 0),
    team_dangerous_space    real NOT NULL CHECK (team_dangerous_space >= 0),
    attacking_hull          real NOT NULL CHECK (attacking_hull >= 0),
    defending_hull          real NOT NULL CHECK (defending_hull >= 0),
    -- Offside line in pitch x, already normalised so the attacking side plays
    -- toward +x. NULL when fewer than two defenders were tracked.
    offside_line            real,
    -- Fitted defensive bank positions, ascending, in the same normalised x.
    defensive_lines         real[],
    available_options       smallint NOT NULL CHECK (available_options >= 0),

    PRIMARY KEY (match_id, job_id, frame_index)
) PARTITION BY HASH (match_id);

COMMENT ON COLUMN frame_score.offside_line IS
    'Pitch x of the offside line, normalised so the attacking side plays toward +x. '
    'NULL means it could not be established (a tracking failure), not "no offside".';

-- Eight partitions is a starting point, not a considered capacity plan; see
-- docs/04-data-model.md before changing it, as the count cannot be altered
-- without rewriting the table.
DO $$
BEGIN
    FOR i IN 0..7 LOOP
        EXECUTE format(
            'CREATE TABLE IF NOT EXISTS frame_score_p%s PARTITION OF frame_score '
            'FOR VALUES WITH (MODULUS 8, REMAINDER %s)', i, i);
    END LOOP;
END $$;

CREATE TABLE IF NOT EXISTS player_frame_score (
    job_id            uuid    NOT NULL,
    match_id          uuid    NOT NULL,
    frame_index       integer NOT NULL,
    track_id          integer NOT NULL,

    pitch_x           real NOT NULL,
    pitch_y           real NOT NULL,
    space_owned       real NOT NULL CHECK (space_owned >= 0),
    -- Metres beyond the offside line. Positive is an offside position.
    offside_margin    real,
    lane_open         boolean NOT NULL,
    lane_clearance    real,
    lines_broken      smallint NOT NULL CHECK (lines_broken >= 0),
    marking_pressure  real NOT NULL CHECK (marking_pressure BETWEEN 0 AND 1),
    nearest_opponent  real CHECK (nearest_opponent IS NULL OR nearest_opponent >= 0),
    position_value    real NOT NULL CHECK (position_value BETWEEN 0 AND 1),

    PRIMARY KEY (match_id, job_id, frame_index, track_id)
) PARTITION BY HASH (match_id);

COMMENT ON COLUMN player_frame_score.offside_margin IS
    'Metres beyond the offside line: positive is an offside position, negative is onside. '
    'A small negative value is a player holding the shoulder of the last defender.';
COMMENT ON COLUMN player_frame_score.lane_clearance IS
    'Distance from the ball-to-player passing lane to the nearest defender. '
    'NULL when no defenders were tracked; infinity is stored as NULL, not a sentinel.';

DO $$
BEGIN
    FOR i IN 0..7 LOOP
        EXECUTE format(
            'CREATE TABLE IF NOT EXISTS player_frame_score_p%s PARTITION OF player_frame_score '
            'FOR VALUES WITH (MODULUS 8, REMAINDER %s)', i, i);
    END LOOP;
END $$;

CREATE INDEX IF NOT EXISTS player_frame_score_track_idx
    ON player_frame_score (job_id, track_id, frame_index);

-- ---------------------------------------------------------------- summaries

CREATE TABLE IF NOT EXISTS player_match_summary (
    job_id      uuid    NOT NULL REFERENCES analysis_job (id) ON DELETE CASCADE,
    track_id    integer NOT NULL,
    match_id    uuid    NOT NULL REFERENCES match (id) ON DELETE CASCADE,
    player_id   uuid    REFERENCES player (id) ON DELETE SET NULL,

    -- Sample size. Carried on every summary so consumers can discard thin
    -- samples rather than over-reading them.
    frames      integer NOT NULL CHECK (frames > 0),
    duration_seconds numeric(9,3) NOT NULL CHECK (duration_seconds >= 0),

    -- Medians, not means: a handful of frames where tracking put a player on
    -- the wrong side of the pitch would drag a mean arbitrarily far.
    median_space_owned      real NOT NULL CHECK (median_space_owned >= 0),
    median_position_value   real NOT NULL CHECK (median_position_value BETWEEN 0 AND 1),
    availability_rate       real NOT NULL CHECK (availability_rate BETWEEN 0 AND 1),
    offside_rate            real NOT NULL CHECK (offside_rate BETWEEN 0 AND 1),
    median_offside_margin   real,
    mean_lines_broken       real NOT NULL CHECK (mean_lines_broken >= 0),
    median_separation       real CHECK (median_separation IS NULL OR median_separation >= 0),
    mean_pressure           real NOT NULL CHECK (mean_pressure BETWEEN 0 AND 1),

    created_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (job_id, track_id),
    FOREIGN KEY (job_id, track_id) REFERENCES track_identity (job_id, track_id)
        ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS player_match_summary_player_idx
    ON player_match_summary (player_id, match_id);

CREATE TABLE IF NOT EXISTS team_match_summary (
    job_id   uuid NOT NULL REFERENCES analysis_job (id) ON DELETE CASCADE,
    match_id uuid NOT NULL REFERENCES match (id) ON DELETE CASCADE,
    side     text NOT NULL CHECK (side IN ('home', 'away')),
    club_id  uuid REFERENCES club (id) ON DELETE SET NULL,

    frames            integer NOT NULL CHECK (frames > 0),
    duration_seconds  numeric(9,3) NOT NULL CHECK (duration_seconds >= 0),
    median_controlled_space real NOT NULL CHECK (median_controlled_space >= 0),
    median_dangerous_space  real NOT NULL CHECK (median_dangerous_space >= 0),
    median_attacking_hull   real NOT NULL CHECK (median_attacking_hull >= 0),
    median_defending_hull   real NOT NULL CHECK (median_defending_hull >= 0),
    mean_passing_options    real NOT NULL CHECK (mean_passing_options >= 0),

    created_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (job_id, side)
);

-- --------------------------------------------------------------------- views

-- What the API serves. Joins identity in and exposes the run's coverage
-- alongside every row, so a consumer cannot read a player's figures without
-- also being handed the reason to distrust them.
CREATE OR REPLACE VIEW player_report AS
SELECT
    s.job_id,
    s.match_id,
    s.track_id,
    s.player_id,
    p.known_as                AS player_name,
    ti.side,
    ti.shirt_number,
    ti.is_goalkeeper,
    s.frames,
    s.duration_seconds,
    s.median_space_owned,
    s.median_position_value,
    s.availability_rate,
    s.offside_rate,
    s.median_offside_margin,
    s.mean_lines_broken,
    s.median_separation,
    s.mean_pressure,
    j.coverage,
    j.pipeline_version,
    -- Mirrors MIN_RELIABLE_FRAMES / MIN_RELIABLE_COVERAGE in the TS SDK.
    (s.frames >= 500 AND j.coverage >= 0.6) AS is_reliable
FROM player_match_summary s
JOIN analysis_job  j  ON j.id = s.job_id
JOIN track_identity ti ON ti.job_id = s.job_id AND ti.track_id = s.track_id
LEFT JOIN player   p  ON p.id = s.player_id;

COMMENT ON VIEW player_report IS
    'API-facing player figures. `is_reliable` combines per-player sample size with '
    'the run''s frame coverage; render a warning whenever it is false.';

COMMIT;
