/**
 * SDK tests. Run with `npm test` (Node's built-in runner, no dependencies).
 *
 * The HTTP layer is exercised against an injected `fetch` stub rather than a
 * live server, so these run in CI without the Python API.
 */

import assert from "node:assert/strict";
import { describe, it } from "node:test";

import {
  OffballApiError,
  OffballClient,
  OffballTimeoutError,
} from "../src/client.ts";
import {
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
} from "../src/analysis.ts";
import type { PlayerSummary, Report } from "../src/types.ts";

// --------------------------------------------------------------- fetch stubs

type Handler = (url: string, init: RequestInit) => { status: number; body: unknown };

function stubFetch(handler: Handler) {
  const calls: { url: string; init: RequestInit }[] = [];
  const fn = (async (input: string | URL | Request, init: RequestInit = {}) => {
    const url = String(input);
    calls.push({ url, init });
    const { status, body } = handler(url, init);
    return new Response(JSON.stringify(body), {
      status,
      headers: { "Content-Type": "application/json" },
    });
  }) as unknown as typeof globalThis.fetch;
  return { fn, calls };
}

const jobBody = (overrides: Record<string, unknown> = {}) => ({
  job_id: "job-1",
  status: "queued",
  match_id: null,
  created_at: "2026-08-04T10:00:00Z",
  updated_at: "2026-08-04T10:00:00Z",
  progress: 0,
  error: null,
  ...overrides,
});

const reportBody = {
  job_id: "job-1",
  match_id: "m1",
  frames_scored: 900,
  frames_total: 1000,
  coverage: 0.9,
  teams: [
    {
      team: "home",
      frames: 900,
      duration: 36,
      median_controlled_space: 3600,
      median_dangerous_space: 540,
      median_attacking_hull: 1800,
      median_defending_hull: 1500,
      mean_passing_options: 3.4,
    },
  ],
  players: [
    {
      track_id: 7,
      frames: 900,
      duration: 36,
      median_space_owned: 420,
      median_position_value: 0.31,
      availability_rate: 0.62,
      offside_rate: 0.04,
      median_offside_margin: -0.8,
      mean_lines_broken: 1.9,
      median_separation: 6.2,
      mean_pressure: 0.22,
    },
    {
      track_id: 9,
      frames: 900,
      duration: 36,
      median_space_owned: 610,
      median_position_value: 0.44,
      availability_rate: 0.18,
      offside_rate: 0.11,
      median_offside_margin: -0.2,
      mean_lines_broken: 2.6,
      median_separation: 9.1,
      mean_pressure: 0.12,
    },
    {
      track_id: 11,
      frames: 40, // thin sample
      duration: 1.6,
      median_space_owned: 999,
      median_position_value: 0.9,
      availability_rate: 0.95,
      offside_rate: 0,
      median_offside_margin: null,
      mean_lines_broken: 3.0,
      median_separation: null,
      mean_pressure: 0.05,
    },
  ],
};

// ------------------------------------------------------------------- client

describe("OffballClient", () => {
  it("requires a baseUrl", () => {
    assert.throws(() => new OffballClient({ baseUrl: "" }), /baseUrl is required/);
  });

  it("strips trailing slashes from the base URL", async () => {
    const { fn, calls } = stubFetch(() => ({ status: 200, body: jobBody() }));
    const client = new OffballClient({ baseUrl: "https://api.test/", fetch: fn });
    await client.getJob("job-1");
    assert.equal(calls[0]!.url, "https://api.test/v1/analyses/job-1");
  });

  it("converts camelCase requests to snake_case on the wire", async () => {
    const { fn, calls } = stubFetch(() => ({ status: 202, body: jobBody() }));
    const client = new OffballClient({ baseUrl: "https://api.test", fetch: fn });
    await client.analyse({ videoUri: "s3://b/m.mp4", matchId: "m1", pitchLength: 105 });

    const sent = JSON.parse(calls[0]!.init.body as string);
    assert.deepEqual(sent, { video_uri: "s3://b/m.mp4", match_id: "m1", pitch_length: 105 });
  });

  it("omits undefined optional fields rather than sending null", async () => {
    const { fn, calls } = stubFetch(() => ({ status: 202, body: jobBody() }));
    const client = new OffballClient({ baseUrl: "https://api.test", fetch: fn });
    await client.analyse({ videoUri: "s3://b/m.mp4" });
    assert.deepEqual(JSON.parse(calls[0]!.init.body as string), { video_uri: "s3://b/m.mp4" });
  });

  it("converts snake_case responses to camelCase", async () => {
    const { fn } = stubFetch(() => ({ status: 200, body: reportBody }));
    const client = new OffballClient({ baseUrl: "https://api.test", fetch: fn });
    const report = await client.getReport("job-1");

    assert.equal(report.framesScored, 900);
    assert.equal(report.players[0]!.medianSpaceOwned, 420);
    assert.equal(report.teams[0]!.meanPassingOptions, 3.4);
  });

  it("revives timestamps as Date objects", async () => {
    const { fn } = stubFetch(() => ({ status: 200, body: jobBody() }));
    const client = new OffballClient({ baseUrl: "https://api.test", fetch: fn });
    const job = await client.getJob("job-1");
    assert.ok(job.createdAt instanceof Date);
    assert.equal(job.createdAt.getUTCFullYear(), 2026);
  });

  it("sends a bearer token when configured", async () => {
    const { fn, calls } = stubFetch(() => ({ status: 200, body: jobBody() }));
    const client = new OffballClient({ baseUrl: "https://api.test", token: "secret", fetch: fn });
    await client.getJob("job-1");
    const headers = calls[0]!.init.headers as Record<string, string>;
    assert.equal(headers.Authorization, "Bearer secret");
  });

  it("throws a typed error carrying the server's detail", async () => {
    const { fn } = stubFetch(() => ({ status: 404, body: { detail: "no such job: x" } }));
    const client = new OffballClient({ baseUrl: "https://api.test", fetch: fn });

    await assert.rejects(
      () => client.getJob("x"),
      (err: unknown) => {
        assert.ok(err instanceof OffballApiError);
        assert.equal(err.status, 404);
        assert.match(err.detail, /no such job/);
        assert.equal(err.isRetryable, false);
        return true;
      },
    );
  });

  it("marks 5xx and 429 as retryable", async () => {
    for (const [status, retryable] of [
      [500, true],
      [503, true],
      [429, true],
      [400, false],
      [409, false],
    ] as const) {
      const { fn } = stubFetch(() => ({ status, body: { detail: "x" } }));
      const client = new OffballClient({ baseUrl: "https://api.test", fetch: fn });
      await assert.rejects(
        () => client.getJob("x"),
        (err: unknown) => {
          assert.ok(err instanceof OffballApiError);
          assert.equal(err.isRetryable, retryable, `status ${status}`);
          return true;
        },
      );
    }
  });

  it("encodes job ids into the path", async () => {
    const { fn, calls } = stubFetch(() => ({ status: 200, body: jobBody() }));
    const client = new OffballClient({ baseUrl: "https://api.test", fetch: fn });
    await client.getJob("a/b c");
    assert.equal(calls[0]!.url, "https://api.test/v1/analyses/a%2Fb%20c");
  });
});

describe("waitForReport", () => {
  it("polls until the job succeeds, then returns the report", async () => {
    const statuses = ["running", "running", "succeeded"];
    let poll = 0;
    const { fn } = stubFetch((url) => {
      if (url.endsWith("/report")) return { status: 200, body: reportBody };
      return { status: 200, body: jobBody({ status: statuses[Math.min(poll++, 2)] }) };
    });

    const client = new OffballClient({ baseUrl: "https://api.test", fetch: fn });
    const seen: string[] = [];
    const report = await client.waitForReport("job-1", {
      pollIntervalMs: 1,
      onProgress: (j) => seen.push(j.status),
    });

    assert.equal(report.framesScored, 900);
    // onProgress fires for the terminal poll too, so a UI can render the
    // final state rather than jumping straight from "running" to a report.
    assert.deepEqual(seen, ["running", "running", "succeeded"]);
  });

  it("throws when the job fails, surfacing the server's reason", async () => {
    const { fn } = stubFetch(() => ({
      status: 200,
      body: jobBody({ status: "failed", error: "no detector configured" }),
    }));
    const client = new OffballClient({ baseUrl: "https://api.test", fetch: fn });

    await assert.rejects(
      () => client.waitForReport("job-1", { pollIntervalMs: 1 }),
      (err: unknown) => {
        assert.ok(err instanceof OffballApiError);
        assert.match(err.detail, /no detector configured/);
        return true;
      },
    );
  });

  it("times out with an actionable error rather than hanging", async () => {
    const { fn } = stubFetch(() => ({ status: 200, body: jobBody({ status: "running" }) }));
    const client = new OffballClient({ baseUrl: "https://api.test", fetch: fn });

    await assert.rejects(
      () => client.waitForReport("job-1", { pollIntervalMs: 1, timeoutMs: 20 }),
      (err: unknown) => {
        assert.ok(err instanceof OffballTimeoutError);
        assert.equal(err.jobId, "job-1");
        // The message must tell the caller the job is still alive server-side.
        assert.match(err.message, /still running/);
        return true;
      },
    );
  });

  it("can be aborted", async () => {
    const { fn } = stubFetch(() => ({ status: 200, body: jobBody({ status: "running" }) }));
    const client = new OffballClient({ baseUrl: "https://api.test", fetch: fn });
    const controller = new AbortController();
    setTimeout(() => controller.abort(new Error("cancelled")), 5);

    await assert.rejects(() =>
      client.waitForReport("job-1", { pollIntervalMs: 2, signal: controller.signal }),
    );
  });
});

// ----------------------------------------------------------------- analysis

const report = (): Report => JSON.parse(JSON.stringify({
  jobId: "job-1",
  matchId: "m1",
  framesScored: 900,
  framesTotal: 1000,
  coverage: 0.9,
  teams: [
    {
      team: "home",
      frames: 900,
      duration: 36,
      medianControlledSpace: 3600,
      medianDangerousSpace: 540,
      medianAttackingHull: 1800,
      medianDefendingHull: 1500,
      meanPassingOptions: 3.4,
    },
  ],
  players: [
    {
      trackId: 7, frames: 900, duration: 36, medianSpaceOwned: 420,
      medianPositionValue: 0.31, availabilityRate: 0.62, offsideRate: 0.04,
      medianOffsideMargin: -0.8, meanLinesBroken: 1.9, medianSeparation: 6.2,
      meanPressure: 0.22,
    },
    {
      trackId: 9, frames: 900, duration: 36, medianSpaceOwned: 610,
      medianPositionValue: 0.44, availabilityRate: 0.18, offsideRate: 0.11,
      medianOffsideMargin: -0.2, meanLinesBroken: 2.6, medianSeparation: 9.1,
      meanPressure: 0.12,
    },
    {
      trackId: 11, frames: 40, duration: 1.6, medianSpaceOwned: 999,
      medianPositionValue: 0.9, availabilityRate: 0.95, offsideRate: 0,
      medianOffsideMargin: null, meanLinesBroken: 3.0, medianSeparation: null,
      meanPressure: 0.05,
    },
  ],
}) as Report);

describe("assessQuality", () => {
  it("passes a clean report", () => {
    const r = report();
    r.players = r.players.filter((p) => p.frames >= 500);
    assert.equal(assessQuality(r).confidence, "high");
    assert.deepEqual(assessQuality(r).warnings, []);
  });

  it("flags thin per-player samples", () => {
    const q = assessQuality(report());
    assert.equal(q.confidence, "low");
    assert.match(q.warnings.join(" "), /fewer than 500 scored frames/);
  });

  it("flags low coverage", () => {
    const r = report();
    r.coverage = 0.3;
    assert.match(assessQuality(r).warnings.join(" "), /Only 30%/);
  });

  it("flags an empty run", () => {
    const r = report();
    r.framesTotal = 0;
    r.players = [];
    const warnings = assessQuality(r).warnings.join(" ");
    assert.match(warnings, /No frames were analysed/);
    assert.match(warnings, /No players were scored/);
  });
});

describe("ranking helpers", () => {
  it("excludes thin samples from rankings", () => {
    const ranked = rankBy(report(), "medianSpaceOwned");
    // Track 11 has the largest value but only 40 frames.
    assert.deepEqual(ranked.map((p) => p.trackId), [9, 7]);
  });

  it("ranks ascending when asked", () => {
    const ranked = rankBy(report(), "medianSpaceOwned", { ascending: true });
    assert.deepEqual(ranked.map((p) => p.trackId), [7, 9]);
  });

  it("respects the limit", () => {
    assert.equal(rankBy(report(), "medianSpaceOwned", { limit: 1 }).length, 1);
  });

  it("can lower the frame threshold deliberately", () => {
    assert.equal(reliablePlayers(report(), 10).length, 3);
  });

  it("finds shoulder runners", () => {
    const runners = shoulderRunners(report());
    // Both are within 2m of the line; -0.2 is finer than -0.8.
    assert.deepEqual(runners.map((p) => p.trackId), [9, 7]);
  });

  it("finds players who get open but do not get the ball", () => {
    const unreachable = unreachablePlayers(report());
    assert.deepEqual(unreachable.map((p) => p.trackId), [9]);
  });
});

describe("team helpers", () => {
  it("looks up a team", () => {
    assert.equal(teamSummary(report(), "home")?.frames, 900);
    assert.equal(teamSummary(report(), "away"), undefined);
  });

  it("computes threat efficiency", () => {
    const home = teamSummary(report(), "home")!;
    assert.equal(threatEfficiency(home), 540 / 3600);
  });

  it("does not divide by zero", () => {
    const home = { ...teamSummary(report(), "home")!, medianControlledSpace: 0 };
    assert.equal(threatEfficiency(home), 0);
  });
});

describe("formatting", () => {
  it("handles nulls", () => {
    assert.equal(formatMetres(null), "n/a");
    assert.equal(formatMetres(-0.83), "-0.8 m");
    assert.equal(formatPercent(0.625), "63%");
  });

  it("computes medians for both parities", () => {
    assert.equal(median([3, 1, 2]), 2);
    assert.equal(median([4, 1, 2, 3]), 2.5);
    assert.equal(median([]), 0);
  });
});
