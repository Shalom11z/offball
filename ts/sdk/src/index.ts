/**
 * offball — TypeScript client for the soccer off-the-ball analytics API.
 *
 * ```ts
 * import { OffballClient, rankBy, assessQuality } from "@offball/sdk";
 *
 * const client = new OffballClient({ baseUrl: "https://api.example.com" });
 * const report = await client.analyseAndWait({ videoUri: "s3://bucket/match.mp4" });
 *
 * const { confidence, warnings } = assessQuality(report);
 * if (confidence === "low") console.warn(warnings.join("\n"));
 *
 * for (const p of rankBy(report, "medianSpaceOwned", { limit: 5 })) {
 *   console.log(p.trackId, p.medianSpaceOwned.toFixed(0), "m²");
 * }
 * ```
 */

export {
  OffballClient,
  OffballApiError,
  OffballTimeoutError,
  type ClientOptions,
  type WaitOptions,
} from "./client.js";

export {
  MIN_RELIABLE_COVERAGE,
  MIN_RELIABLE_FRAMES,
  assessQuality,
  formatMetres,
  formatPercent,
  median,
  rankBy,
  reliablePlayers,
  shoulderRunners,
  teamSummary,
  threatEfficiency,
  unreachablePlayers,
  type Confidence,
  type QualityAssessment,
  type RankableMetric,
} from "./analysis.js";

export type {
  AnalysisRequest,
  Job,
  JobStatus,
  PlayerSummary,
  Report,
  TeamName,
  TeamSummary,
} from "./types.js";
