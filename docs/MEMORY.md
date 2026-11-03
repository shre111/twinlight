# Memory

Working memory for coding sessions. Read at the start of every session; update at the end of every point or session with `/wrap`. Keep it short: current truth only, newest first in each section. Move anything stable into DECISIONS.md or ARCHITECTURE.md instead of letting it pile up here.

## Now
- Milestone: M0 Foundations
- Point: P-005 Dev infrastructure (Docker healthy again after the restart). Lane B on P-015 (stage core and perf HUD merged, `just shots` next), then P-016 on a synthetic tape. Lane C (minds) not started yet.
- Branch: `feat/p005-dev-infra` (local only; Postgres 17 + PostGIS + h3 image committed and built)
- Next: P-005 compose and gateway connection, then M1 Lane A (P-010 toy city, P-011 engine core).

## Recent changes
- 2026-11-03 P-006 done: path-filtered CI with shared lockfile/workflow filters, a `ci-ok` gate, actions on their Node 24 majors (#10). Lane B merged stage core (#9) and the perf HUD (#11).
- 2026-11-03 P-007 done: `@twinlight/ui` generates CSS vars and TS constants from `design/tokens.json` (checked in, freshness test); Atkinson body face; lamp focus ring; vitest guard on colour literals across apps and packages (#8).
- 2026-11-03 P-004 done: NestJS 12 gateway (ESM) with `/v1/health` (version, git sha), zod env config, OpenAPI at `/docs`, pino with auth headers redacted; vitest e2e (#7).
- 2026-11-03 P-003 done: `services/minds` `/health` and `tools/citybake` CLI (stdlib argparse) as workspace members; python CI filter restored (#6).
- 2026-11-03 P-002 done: uv workspace (members by glob), twl_engine with a Numba smoke test, twl_tape skeleton, runner `/health` (#4).
- 2026-11-03 P-001 done: pnpm workspace, Turborepo, Biome, `packages/config`, justfile (#1); Next.js 16 app with the wordmark on dusk (#2).
- 2026-11-03 Repo seeded with planning docs, Claude Code agents, skills, hooks and CI from the starter kit.

## Gotchas
- Node 24 is pinned in `.nvmrc`, but this machine's nvm default is 22: run `nvm use` (or prefix commands with it) before pnpm or just.
- TypeScript is pinned to 6.0: the native 7.0 has no JS API yet, which Next's build type-check and the NestJS CLI use.
- Biome 2.6 also formats Markdown and YAML; `biome.json` excludes those, `.claude/` and `design/` (D-018 scopes Biome to TS/JS/JSON).
- `next dev` writes `apps/web/AGENTS.md` unless `agentRules: false` (set). Next 16's own docs ship in `apps/web/node_modules/next/dist/docs/`; read them before relying on remembered APIs.
- Tokens: after editing `design/tokens.json`, run `pnpm -F @twinlight/ui tokens`; the freshness test fails otherwise. The colour guard doesn't catch named colours (`white`), and `href="#add"` would be a false positive.
- Fonts: `next/font/local` paths must be relative to the calling file (package specifiers fail under Turbopack), and the font variable classes must stay on `<html>`. The rupee sign is only in the latin-ext subsets: add latin-ext faces (`unicode-range`, `preload: false`) before Brief or Plan show prices.
- Phone stills: headless Chrome `--window-size` below about 500 px lays out wider; use viewport emulation (Playwright or CDP).
- Disk: the Mac disk filled once (Docker VM about 38 GB) and Docker hung; restarting Docker means restarting the laptop. Check `df -h /` before big downloads. MinIO comes from `cgr.dev/chainguard/minio` pinned by digest (official images are gone).
- CI: `ci-ok` is the single status (it reports even when all jobs skip; docs-only PRs run only `changes`, `ci-ok` and the title check). The formats job matches only `packages/tape/**` and `fixtures/tape/**` until P-014 adds `py/tape/**`; P-025 adds `fixtures/pack/**`. The python matrix lists projects explicitly, so a new `tools/` dir must be added to it. `astral-sh/setup-uv` has no major tags; pin its exact release.
- P-010 port: the prototype's toy sim works in three.js axes (negative z is north), so pack `y = -z_proto` (ARCHITECTURE §5).
- Stage (Lane B): use `http://localhost:<port>`, not `127.0.0.1` (Next dev blocks it as cross-origin). Lane B's dev server runs on 3101. On an M4 the empty stage runs 16.7 ms p50; the prototype itself averages about 24 ms at 1837 px and DPR 1.75, so P-016 has to beat it. Shot baselines are macOS SwiftShader; CI needs Linux baselines (P-046).
- The CI `TWL_WORKERS` loop is gone; the first golden tests (P-011) must parametrise 1, 4 and 8 workers inside pytest (golden-test skill) so D-006 stays covered.
- Starlette's test client wants `httpx2`, not `httpx` (deprecation warning otherwise). pytest runs with `--import-mode=importlib`, so test module names may repeat across packages.
- The git hook rejects any Bash command containing a heredoc together with the words git and commit on one line; do file edits and commits in separate calls.
- Gateway: NestJS 12 is ESM-only, so imports carry `.js` suffixes. There's no Nest CLI: dev runs through `@swc-node/register` and tests through `unplugin-swc`, because both emit decorator metadata (esbuild and tsx don't, which breaks DI). Biome needs `unsafeParameterDecoratorsEnabled` for it (override in `biome.json`). `NODE_ENV` defaults to production, so set `NODE_ENV=development` in `.env` for pretty logs.
- `/docs` and `/docs-json` are public; gate or disable them in production when auth lands (P-050).
- pnpm 10 skips dependency install scripts; `pnpm-workspace.yaml` lists the ones deliberately ignored. A new native dependency that needs its script goes in `onlyBuiltDependencies`.
- No branch protection on `main` (owner's choice for now); the git hook and squash-only repo settings are the only guards.
- The git hook blocks multi-line commit messages, heredoc messages, `--no-verify`, force pushes, commits on `main` and badly named branches. Use a single `-m "type(scope): subject"`. Squash-merge deletes the branch, so each PR of a point gets its own branch.
- `design/prototype/twinlight-city.html` is the visual reference for the stage (WebGL, single file, toy sim inlined). It is not production code; match its look, not its structure. Its scale clamps (point size, fog distance) were tuned by eye and are a good starting point for P-016.
- Shaders: never `pow()` a value that can go negative; it produced NaN black blocks on real GPUs in the prototype that headless renders did not show (DESIGN.md, shader rules).
- Engine golden hashes are bit-exact only on the pinned CI platform; locally compare probes and aggregates (D-006).
- Workspace network may block CDNs and font hosts; stage visual tests must load three.js and fonts from node_modules.

## Open questions
- License: the repo is public with no license (owner accepted public; licence still to pick).
- Golden platform pinning (D-021 to propose at P-011, when the first goldens exist): `NUMBA_CPU_NAME=generic` plus NumPy CPU feature limits, and `just goldens` via a CI workflow.
- Voice provider for Hindi and Gujarati streaming (spike at P-066).
- Whether to keep 15-minute ticks once real occasion profiles exist (revisit after P-031 benchmarks).

## Demo log
- (add a line with a link or path whenever a milestone demo is recorded)
