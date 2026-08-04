/**
 * Wire types for the offball API.
 *
 * These mirror the Pydantic models in `python/src/offball/api/schemas.py`.
 * That file is the source of truth; when it changes, regenerate from the
 * OpenAPI document (`npm run codegen`) rather than editing here by hand.
 */

export type JobStatus = "queued" | "running" | "succeeded" | "failed";

export type TeamName = "home" | "away" | "referee" | "unknown";

/** Request to analyse one match video. */
export interface AnalysisRequest {
  /** Location of the source footage (`s3://`, `gs://`, or a local path). */
  videoUri: string;
  /** Your own identifier for the fixture. */
  matchId?: string;
  /** Source frame rate. Defaults to 25. */
  fps?: number;
  /**
   * Analyse every Nth frame. Raising this trades tracking robustness for
   * throughput.
   */
  stride?: number;
  /** Pitch length in metres (90-120). */
  pitchLength?: number;
  /** Pitch width in metres (45-90). */
  pitchWidth?: number;
}

export interface Job {
  jobId: string;
  status: JobStatus;
  matchId: string | null;
  createdAt: Date;
  updatedAt: Date;
  /** 0-1, best effort. */
  progress: number;
  error: string | null;
}

/** One player's off-the-ball profile. */
export interface PlayerSummary {
  trackId: number;
  /**
   * Frames this player was scored in — the sample size behind every figure
   * here. Treat anything under a few hundred as indicative only.
   */
  frames: number;
  /** Seconds of scored off-ball time. */
  duration: number;
  /** Median pitch area owned, m². */
  medianSpaceOwned: number;
  /** Median threat value of the ground occupied, 0-1. */
  medianPositionValue: number;
  /** Share of frames offering the ball carrier a clear passing lane. */
  availabilityRate: number;
  /** Share of frames in an offside position. */
  offsideRate: number;
  /**
   * Median metres beyond the offside line; negative is onside. Small negative
   * values indicate a player who consistently plays on the shoulder.
   * `null` when the line could not be established.
   */
  medianOffsideMargin: number | null;
  /** Mean number of opposition banks played beyond. */
  meanLinesBroken: number;
  /** Median nearest-opponent distance, metres. */
  medianSeparation: number | null;
  /** Mean marking pressure, 0-1. */
  meanPressure: number;
}

export interface TeamSummary {
  team: TeamName;
  frames: number;
  duration: number;
  medianControlledSpace: number;
  medianDangerousSpace: number;
  medianAttackingHull: number;
  medianDefendingHull: number;
  /** Mean count of teammates offering a viable pass at any moment. */
  meanPassingOptions: number;
}

export interface Report {
  jobId: string;
  matchId: string | null;
  framesScored: number;
  framesTotal: number;
  /**
   * Share of frames that could be scored. Below ~0.6 the vision stage
   * struggled and the figures are provisional.
   */
  coverage: number;
  teams: TeamSummary[];
  players: PlayerSummary[];
}
