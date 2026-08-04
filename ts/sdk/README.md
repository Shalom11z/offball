# @offball/sdk

TypeScript client for the [offball](https://github.com/Shalom11z/offball) API.

Dependency-free — uses the platform `fetch`, so it runs in Node 18+, Deno, Bun
and the browser.

```bash
npm install @offball/sdk
```

```ts
import { OffballClient, assessQuality, rankBy } from "@offball/sdk";

const client = new OffballClient({ baseUrl: "https://api.example.com" });

const report = await client.analyseAndWait(
  { videoUri: "s3://bucket/match.mp4", matchId: "2026-08-04-ARS-CHE" },
  { onProgress: (job) => console.log(job.status, job.progress) },
);

const { confidence, warnings } = assessQuality(report);
if (confidence === "low") console.warn(warnings.join("\n"));

for (const p of rankBy(report, "medianSpaceOwned", { limit: 5 })) {
  console.log(`#${p.trackId}: ${p.medianSpaceOwned.toFixed(0)} m²`);
}
```

## Why the analysis helpers live here

`src/analysis.ts` holds the interpretation layer — what counts as a thin
sample, how to rank a squad, when to distrust a run. These are presentation
decisions a UI should be able to tune without a server deploy. The API ships
measurements; this package interprets them. Nothing in it invents data.

## Scripts

| Command | Purpose |
| --- | --- |
| `npm test` | Node's built-in runner, no dependencies |
| `npm run typecheck` | `tsc --noEmit` under `strict` |
| `npm run build` | Emit `dist/` with declarations |
