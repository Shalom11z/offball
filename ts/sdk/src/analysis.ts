/**
 * Client-side helpers for reading a report.
 *
 * Deliberately kept on this side of the wire: these are presentation and
 * triage decisions (what counts as a thin sample, how to rank a squad, when to
 * distrust a run) that a UI wants to tune without a server deploy. The server
 * ships measurements; this file interprets them.
 *
 * Nothing here invents data. Every function is a pure transformation of a
 * {@link Report}.
 */

import type { PlayerSummary, Report, TeamSummary } from "./types.js";

/**
 * Below this many scored frames, a player's figures are too noisy to rank on.
 * At 25fps this is 20 seconds of off-ball time.
 */
export const MIN_RELIABLE_FRAMES = 500;

/** Below this coverage, the vision stage struggled and the run needs a look. */
export const MIN_RELIABLE_COVERAGE = 0.6;

export type Confidence = "high" | "low";

export interface QualityAssessment {
  confidence: Confidence;
  /** Human-readable reasons, empty when confidence is high. */
  warnings: string[];
}

/**
 * Whether a report should be trusted, and why not if not.
 *
 * Surface this in any UI that shows the numbers. A silent low-coverage report
 * is the most likely way for someone to draw a confident wrong conclusion.
 */
export function assessQuality(report: Report): QualityAssessment {
  const warnings: string[] = [];

  if (report.framesTotal === 0) {
    warnings.push("No frames were analysed.");
  } else if (report.coverage < MIN_RELIABLE_COVERAGE) {
    warnings.push(
      `Only ${(report.coverage * 100).toFixed(0)}% of frames could be scored ` +
        `(want ${(MIN_RELIABLE_COVERAGE * 100).toFixed(0)}%+). Check camera calibration.`,
    );
  }

  const thin = report.players.filter((p) => p.frames < MIN_RELIABLE_FRAMES);
  if (thin.length > 0) {
    warnings.push(
      `${thin.length} of ${report.players.length} players have fewer than ` +
        `${MIN_RELIABLE_FRAMES} scored frames; their figures are indicative only.`,
    );
  }

  if (report.players.length === 0) {
    warnings.push("No players were scored.");
  }

  return { confidence: warnings.length === 0 ? "high" : "low", warnings };
}

/** Players with enough scored frames to compare against each other. */
export function reliablePlayers(
  report: Report,
  minFrames = MIN_RELIABLE_FRAMES,
): PlayerSummary[] {
  return report.players.filter((p) => p.frames >= minFrames);
}

export type RankableMetric = {
  [K in keyof PlayerSummary]: PlayerSummary[K] extends number ? K : never;
}[keyof PlayerSummary];

/**
 * Rank players by a numeric metric, descending by default.
 *
 * Only considers players above the frame threshold — ranking a squad on a
 * player who appeared for 30 frames produces a confident lie.
 */
export function rankBy(
  report: Report,
  metric: RankableMetric,
  options: { ascending?: boolean; minFrames?: number; limit?: number } = {},
): PlayerSummary[] {
  const { ascending = false, minFrames = MIN_RELIABLE_FRAMES, limit } = options;
  const sorted = reliablePlayers(report, minFrames).sort((a, b) =>
    ascending ? a[metric] - b[metric] : b[metric] - a[metric],
  );
  return limit === undefined ? sorted : sorted.slice(0, limit);
}

/**
 * Players who consistently position themselves on the last defender's
 * shoulder: onside, but only just.
 *
 * `medianOffsideMargin` in [-`band`, 0] is the signature of a striker timing
 * runs finely. A large negative median means they habitually drop off.
 */
export function shoulderRunners(report: Report, band = 2.0): PlayerSummary[] {
  return reliablePlayers(report)
    .filter(
      (p) =>
        p.medianOffsideMargin !== null &&
        p.medianOffsideMargin <= 0 &&
        p.medianOffsideMargin >= -band,
    )
    .sort((a, b) => (b.medianOffsideMargin ?? -Infinity) - (a.medianOffsideMargin ?? -Infinity));
}

/**
 * Players who find space but are rarely reachable.
 *
 * A high space-owned figure with a low availability rate is the classic
 * "great movement, wrong moments" profile — or a sign their team is not
 * looking for them.
 */
export function unreachablePlayers(
  report: Report,
  options: { minSpace?: number; maxAvailability?: number } = {},
): PlayerSummary[] {
  const players = reliablePlayers(report);
  if (players.length === 0) return [];

  const { maxAvailability = 0.3 } = options;
  const minSpace =
    options.minSpace ?? median(players.map((p) => p.medianSpaceOwned));

  return players
    .filter((p) => p.medianSpaceOwned >= minSpace && p.availabilityRate <= maxAvailability)
    .sort((a, b) => b.medianSpaceOwned - a.medianSpaceOwned);
}

/** The team summary for a given side, if present. */
export function teamSummary(report: Report, team: TeamSummary["team"]): TeamSummary | undefined {
  return report.teams.find((t) => t.team === team);
}

/**
 * Dangerous space as a share of all controlled space, 0-1.
 *
 * Separates "we have lots of the pitch" from "we have the parts that matter".
 * Two sides can control identical area while one does it entirely in their own
 * half.
 */
export function threatEfficiency(team: TeamSummary): number {
  if (team.medianControlledSpace <= 0) return 0;
  return team.medianDangerousSpace / team.medianControlledSpace;
}

export function median(values: number[]): number {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  // `mid` and `mid - 1` are provably in range for a non-empty array, but
  // `noUncheckedIndexedAccess` cannot see that.
  const hi = sorted[mid] as number;
  if (sorted.length % 2 !== 0) return hi;
  return ((sorted[mid - 1] as number) + hi) / 2;
}

/** Format a metric for display, handling the `null` cases sensibly. */
export function formatMetres(value: number | null, digits = 1): string {
  return value === null ? "n/a" : `${value.toFixed(digits)} m`;
}

export function formatPercent(value: number, digits = 0): string {
  return `${(value * 100).toFixed(digits)}%`;
}
