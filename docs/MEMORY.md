# Memory

Working memory for coding sessions. Read at the start of every session; update at the end of every point or session with `/wrap`. Keep it short: current truth only, newest first in each section. Move anything stable into DECISIONS.md or ARCHITECTURE.md instead of letting it pile up here.

## Now
- Milestone: M1 First light (Lane A, B); Lane C on M2.
- Point: P-010 code merged (#20, #21, #23, #29); stays unticked until Lane B's toy-pack loader renders `data/cache/toy-48k`. Next Lane A point: P-011 engine core (`[deep]`: the owner switches to Fable first). Lane B: P-016 (sky, ground, fog, camera, roads, instanced buildings merged; residents layer, scrubber and day stills next, then the toy-pack loader). P-015 Firefox still unchecked (owner approved the WebKit download only). Lane C: P-020 code complete; P-021 next with D-021 footprints; P-024 per D-022.
- Branch: none
- Next: P-011 on Fable; meanwhile P-013 supply v0 or P-014 tape v1 can be prepared.

## Recent changes
- 2026-11-03 P-010 code: `twl_engine.rng` streams; toy city port (7,210 buildings, 4.50 ties per resident at 48k, 36 ms); v1 pack writer + CLI; independent tests from test-author; golden in `fixtures/golden/hashes.toml` (#20, #21, #23, #29). D-021 (VIDA footprints) and D-022 (graph travel times) recorded (#28).
- 2026-11-03 Lane B: stage sky tokens (#19), sky, ground, fog, camera (#24), roads and instanced buildings, 7 draw calls at 16.7 ms p50 on the M4 (#26), `just shots` (#13). Lane C: OSM fetch, clip and layers for Ahmedabad (#16, #18, #22), height defaults by footprint band (#27).
- 2026-11-03 P-005 done: `infra/compose/dev.yml` (Postgres 17 + PostGIS 3.7 + h3 4.2, Chainguard MinIO with packs and tapes, Redis 8.8, Temporal dev, otel-lgtm), `just infra`/`down` (#15); gateway `/v1/health/ready` probes Postgres and MinIO (#17).
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
- Dev stack ports: this Mac runs native Postgres and Redis on 5432 and 6379, so the stack maps Postgres to 15432 and Redis to 16379 (all ports bound to 127.0.0.1). A `.env` copied before that still points at 5432/6379; update `DATABASE_URL` and `REDIS_URL` from `.env.example`. Grafana is on 3001.
- Compose: `up --wait` exits 1 if any started container exits (even with 0), so the bucket one-shot sits behind `profiles: [init]` and `just infra` runs it after. Images pulled while the disk was full can be silently corrupt (zero-byte files) and Docker reuses the layers; switch tags rather than prune others' images.
- Toy city: pin seed 49 in count checks (the prototype's own building count varies about 2.7% across seeds). Toy ties are one-way, so a few residents have under 3. The toy keeps the prototype origin; its bbox excludes the river ring (±6 km). Regenerate `data/cache/toy-48k` and `toy-5k` after any pack change (`uv run python -m twl_engine.fixtures --seed 49 --residents N --out DIR`).
- Goldens: the 5k toy pack hash matched on macOS arm64 and CI Linux x86_64. CI checks hashes (`TWL_GOLDEN_STRICT=1`); elsewhere probes are compared with tolerance.
- Stage shaders (Lane B): use TSL `colorNode`/`opacityNode`, never `fragmentNode` (skips the sRGB output transform); mix shades on sRGB values, then convert. P-045 needs a sanitize pass before bloom (no per-fragment NaN guard exists) and output dither (dark gradients band on the real GPU). The colour guard also matches hex inside comments. three/webgpu and three/tsl load in Node, so vitest can test layers.
- Citybake (Lane C): data in `data/raw` and `data/cache/citybake/<city>/<stage>` (`TWL_DATA_DIR` overrides). `uv add --package <member>` uninstalls other members' deps; run `uv sync` after. WorldPop R2025A India is 778 MB with no range requests. pyosmium skips self-intersecting closed ways as invalid. `shapely.voronoi_polygons(ordered=True)` needs GEOS 3.12+.
- Ahmedabad data (Lane C): 100,488 OSM buildings, 81% with default heights, 37% of 1 km cells empty (hence D-021); 105 districts from OSM place Voronoi split at the Sabarmati; v1 water drops one lake island. Persona pool: 6,604 Ahmadabad-district personas in Nemotron-Personas-India, all 18+. Census HH-01 mean household 4.7.
- Stage inputs: roads as segments (x0, y0, x1, y1), split from `roads.bin` polylines by the pack loader; residents carry `building`. `@twinlight/stage/synthetic` mirrors two toy-sim constants (delivery radius 900 m, rain hours 264 to 336); change both together. `just shots` reuses a dev server on `SHOTS_PORT` (default 3101). Compare shots on p50 and p95 (SwiftShader mean fps swings 41 to 52). Playwright wipes its output dir each run (named after the branch, never `artifacts/shots/` itself), skips `undefined` options so `channel` is set per project, and `toHaveScreenshot` takes `stylePath` while ignoring unknown options.
- Stage (Lane B): use `http://localhost:<port>`, not `127.0.0.1` (Next dev blocks it as cross-origin). Lane B's dev server runs on 3101. On an M4 the empty stage runs 16.7 ms p50; the prototype itself averages about 24 ms at 1837 px and DPR 1.75, so P-016 has to beat it. Shot baselines are macOS SwiftShader; CI needs Linux baselines (P-046).
- The CI `TWL_WORKERS` loop is gone; the first engine goldens (P-011) must parametrise 1, 4 and 8 workers inside pytest (golden-test skill) so D-006 stays covered.
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
- Golden platform pinning (a new decision at P-011, when Numba kernels enter the hashed path): `NUMBA_CPU_NAME=generic` plus NumPy CPU feature limits, and `just goldens` via a CI workflow. The toy pack already matches across platforms.
- Voice provider for Hindi and Gujarati streaming (spike at P-066).
- Whether to keep 15-minute ticks once real occasion profiles exist (revisit after P-031 benchmarks).

## Demo log
- (add a line with a link or path whenever a milestone demo is recorded)
