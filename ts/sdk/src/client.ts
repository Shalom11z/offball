/**
 * Typed client for the offball API.
 *
 * Analysis is asynchronous — a full match takes minutes to hours — so the
 * primary flow is submit, poll, fetch. {@link OffballClient.analyseAndWait}
 * wraps that loop for you.
 *
 * Dependency-free: uses the platform `fetch` and `AbortSignal`, so it runs in
 * Node 18+, Deno, Bun, and the browser without a bundler.
 */

import type {
  AnalysisRequest,
  Job,
  JobStatus,
  PlayerSummary,
  Report,
  TeamSummary,
} from "./types.js";

/** Thrown for any non-2xx response. */
export class OffballApiError extends Error {
  // Declared explicitly rather than as constructor parameter properties, which
  // Node's type-stripping loader (`--experimental-strip-types`) cannot handle.
  readonly status: number;
  readonly detail: string;
  readonly path: string;

  constructor(status: number, detail: string, path: string) {
    super(`offball API ${status} on ${path}: ${detail}`);
    this.name = "OffballApiError";
    this.status = status;
    this.detail = detail;
    this.path = path;
  }

  /** Whether retrying the same request could plausibly succeed. */
  get isRetryable(): boolean {
    return this.status >= 500 || this.status === 429;
  }
}

/** Thrown when {@link OffballClient.analyseAndWait} exceeds its deadline. */
export class OffballTimeoutError extends Error {
  readonly jobId: string;
  readonly lastStatus: JobStatus;

  constructor(jobId: string, lastStatus: JobStatus) {
    super(
      `job ${jobId} did not finish in time (last status: ${lastStatus}). ` +
        `The job is still running server-side; poll getJob("${jobId}") to continue.`,
    );
    this.name = "OffballTimeoutError";
    this.jobId = jobId;
    this.lastStatus = lastStatus;
  }
}

export interface ClientOptions {
  /** Base URL of the API, e.g. `https://api.example.com`. */
  baseUrl: string;
  /** Sent as `Authorization: Bearer <token>` when provided. */
  token?: string;
  /** Per-request timeout in ms. Defaults to 30s. */
  timeoutMs?: number;
  /** Injectable for tests. Defaults to the global `fetch`. */
  fetch?: typeof globalThis.fetch;
}

export interface WaitOptions {
  /** How often to poll, in ms. Defaults to 5000. */
  pollIntervalMs?: number;
  /** Give up after this long, in ms. Defaults to 1 hour. */
  timeoutMs?: number;
  /** Called after each poll, for progress reporting. */
  onProgress?: (job: Job) => void;
  /** Abort the wait early. */
  signal?: AbortSignal;
}

/** snake_case on the wire, camelCase in TypeScript. */
function toCamel<T>(value: unknown): T {
  if (Array.isArray(value)) return value.map((v) => toCamel(v)) as T;
  if (value === null || typeof value !== "object") return value as T;
  const out: Record<string, unknown> = {};
  for (const [k, v] of Object.entries(value as Record<string, unknown>)) {
    out[k.replace(/_([a-z0-9])/g, (_, c: string) => c.toUpperCase())] = toCamel(v);
  }
  return out as T;
}

function toSnake(value: Record<string, unknown>): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  for (const [k, v] of Object.entries(value)) {
    if (v === undefined) continue;
    out[k.replace(/[A-Z]/g, (c) => `_${c.toLowerCase()}`)] = v;
  }
  return out;
}

const sleep = (ms: number, signal?: AbortSignal): Promise<void> =>
  new Promise((resolve, reject) => {
    if (signal?.aborted) return reject(signal.reason);
    const timer = setTimeout(resolve, ms);
    signal?.addEventListener(
      "abort",
      () => {
        clearTimeout(timer);
        reject(signal.reason);
      },
      { once: true },
    );
  });

export class OffballClient {
  private readonly baseUrl: string;
  private readonly token?: string;
  private readonly timeoutMs: number;
  private readonly doFetch: typeof globalThis.fetch;

  constructor(options: ClientOptions) {
    if (!options.baseUrl) throw new Error("baseUrl is required");
    this.baseUrl = options.baseUrl.replace(/\/+$/, "");
    this.token = options.token;
    this.timeoutMs = options.timeoutMs ?? 30_000;
    this.doFetch = options.fetch ?? globalThis.fetch.bind(globalThis);
  }

  private async request<T>(
    path: string,
    init: RequestInit = {},
    signal?: AbortSignal,
  ): Promise<T> {
    const headers: Record<string, string> = {
      Accept: "application/json",
      ...(init.body ? { "Content-Type": "application/json" } : {}),
      ...(this.token ? { Authorization: `Bearer ${this.token}` } : {}),
      ...((init.headers as Record<string, string>) ?? {}),
    };

    // Combine the caller's abort signal with our own timeout.
    const timeout = AbortSignal.timeout(this.timeoutMs);
    const combined = signal ? AbortSignal.any([signal, timeout]) : timeout;

    const response = await this.doFetch(`${this.baseUrl}${path}`, {
      ...init,
      headers,
      signal: combined,
    });

    if (!response.ok) {
      let detail = response.statusText;
      try {
        const body = (await response.json()) as { detail?: string };
        if (body?.detail) detail = body.detail;
      } catch {
        // Non-JSON error body; the status text will have to do.
      }
      throw new OffballApiError(response.status, detail, path);
    }

    if (response.status === 204) return undefined as T;
    return toCamel<T>(await response.json());
  }

  /** Liveness check. Also reports which numeric backend the server compiled in. */
  async health(): Promise<{ status: string; version: string; kernelBackend: string }> {
    return this.request("/healthz");
  }

  /** Queue a match for analysis. Returns immediately with a job. */
  async analyse(request: AnalysisRequest): Promise<Job> {
    const job = await this.request<Job>("/v1/analyses", {
      method: "POST",
      body: JSON.stringify(toSnake(request as unknown as Record<string, unknown>)),
    });
    return reviveJob(job);
  }

  async getJob(jobId: string, signal?: AbortSignal): Promise<Job> {
    return reviveJob(
      await this.request<Job>(`/v1/analyses/${encodeURIComponent(jobId)}`, {}, signal),
    );
  }

  async listJobs(limit = 50): Promise<Job[]> {
    const jobs = await this.request<Job[]>(`/v1/analyses?limit=${limit}`);
    return jobs.map(reviveJob);
  }

  /**
   * Fetch a finished report.
   *
   * Throws {@link OffballApiError} with status 409 while the job is still
   * running, which is distinguishable from a 404 for an unknown job.
   */
  async getReport(jobId: string, signal?: AbortSignal): Promise<Report> {
    return this.request<Report>(
      `/v1/analyses/${encodeURIComponent(jobId)}/report`,
      {},
      signal,
    );
  }

  /** Submit a match and poll until the report is ready. */
  async analyseAndWait(request: AnalysisRequest, options: WaitOptions = {}): Promise<Report> {
    const job = await this.analyse(request);
    return this.waitForReport(job.jobId, options);
  }

  /** Poll an existing job until it finishes, then return its report. */
  async waitForReport(jobId: string, options: WaitOptions = {}): Promise<Report> {
    const interval = options.pollIntervalMs ?? 5_000;
    const deadline = Date.now() + (options.timeoutMs ?? 3_600_000);
    let last: Job = await this.getJob(jobId, options.signal);

    while (Date.now() < deadline) {
      options.onProgress?.(last);

      if (last.status === "succeeded") {
        return this.getReport(jobId, options.signal);
      }
      if (last.status === "failed") {
        throw new OffballApiError(422, last.error ?? "analysis failed", `/v1/analyses/${jobId}`);
      }

      await sleep(interval, options.signal);
      last = await this.getJob(jobId, options.signal);
    }

    throw new OffballTimeoutError(jobId, last.status);
  }
}

/** Dates arrive as ISO strings; hand callers real `Date` objects. */
function reviveJob(job: Job): Job {
  return {
    ...job,
    createdAt: new Date(job.createdAt),
    updatedAt: new Date(job.updatedAt),
  };
}

export type { AnalysisRequest, Job, JobStatus, PlayerSummary, Report, TeamSummary };
