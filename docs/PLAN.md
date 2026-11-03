# Plan

Points are worked in order within a milestone unless marked parallel. Each point becomes one or more micro PRs on a branch named after it (`feat/p032-residents-layer`).

Markers:
- `[tests]` automated tests required (see CLAUDE.md testing policy)
- `[deep]` hard reasoning; use the strongest model for this point (see KICKOFF.md)
- `Lane A/B/C` which parallel lane owns it

Milestones are vertical slices. M1 exists to get a city playing a launch on screen as early as possible; later milestones replace each toy part with the real one.

---

## M0 Foundations
- [x] **P-001 Monorepo scaffold** Lane A. pnpm workspaces, Turborepo, Biome, base tsconfig, `packages/config`, `apps/web` (Next.js 16) rendering the Twinlight wordmark on the dusk background. `justfile` with `setup`, `dev`, `lint`, `test`.
  AC: `just setup && just lint && just test` pass on a clean clone; web app starts.
- [x] **P-002 Python workspace** Lane A. Root `pyproject.toml` as a uv workspace; `py/engine` (`twl_engine`), `py/tape` (`twl_tape`) and `services/runner` (FastAPI `/health`) skeletons; NumPy, Numba, ruff, pyright and pytest wired; a Numba smoke test.
  AC: `just test-engine` runs; ruff and pyright clean; the Numba smoke test passes on Linux and macOS.
- [x] **P-003 Minds and citybake** Lane C. `services/minds` (FastAPI `/health`) and `tools/citybake` (CLI `citybake --help`) as members of the uv workspace, with ruff, pyright and pytest.
  AC: `uv run pytest` green in both.
- [x] **P-004 Gateway skeleton** Lane A. NestJS app with `/v1/health`, config validation, OpenAPI at `/docs`, pino logging.
  AC: health returns version and git sha.
- [ ] **P-005 Dev infrastructure** Lane A. `infra/compose/dev.yml`: Postgres 17 with PostGIS + h3, MinIO with buckets `packs` and `tapes`, Redis, Temporal dev server, otel-lgtm. `just dev` brings it up.
  AC: all containers healthy; gateway connects to Postgres and MinIO.
- [ ] **P-006 CI** Lane A. Path-filtered jobs: web (biome, tsc, vitest), engine (ruff, pyright, pytest, determinism across worker counts), python services (ruff, pyright, pytest), cross-language goldens, PR title check. Cache pnpm and uv.
  AC: a docs-only PR runs only the title check; an engine PR runs only engine + title.
- [x] **P-007 Tokens package** Lane B. `design/tokens.json` -> `packages/ui` CSS variables + TS constants; fonts loaded with `next/font`.
  AC: web app uses only token variables; a lint rule or test fails on hex literals in components.

## M1 First light (vertical slice on a toy city)
Goal: a procedural toy city of ~20k residents plays a 30-day launch, end to end through engine, tape, stage and timeline. No LLM, no OSM yet.
- [ ] **P-010 Toy city generator** Lane A. Port the prototype's procedural city (`design/prototype/toy-sim.mjs`: `buildCity`, `placeResidents`, `placeStores`, `buildTies`) to Python in `twl_engine.fixtures`, vectorised where it can be, using the engine's seeded streams. Sizes: 48,000 residents and 14 stores to match the prototype, 5,000 for fast tests. Writes a city pack (minimal subset of the real format).
  AC: same seed produces a byte-identical pack; at 48,000 residents the building count is within 3% of the prototype's 7,253 and ties per resident within 3% of its 4.5; pack loads in the stage. `[tests]`
- [ ] **P-011 Engine core** Lane A. Integer clock, `SeedSequence` streams, NumPy struct-of-arrays residents, vectorised tick loop, Numba event queue. `[tests] [deep]`
  AC: determinism golden across 1/4/8 worker processes; empty scenario produces a valid empty tape.
- [ ] **P-012 Awareness and adoption v0** Lane A. Paid campaign exposure, word of mouth over the graph, logistic purchase decision, repeat purchases. `[tests] [deep]`
  AC: Bass-limit test (MODEL.md section 4) within 2%; monotonicity property tests (hypothesis) pass; the prototype's launch scenario (`simulate` in `design/prototype/toy-sim.mjs`) lands within 3 points of its numbers on days 6, 13 and 30 (23/4.5, 42/11, 63/23 percent heard/ordered; `node design/prototype/toy-sim.mjs` prints them).
- [ ] **P-013 Supply v0** Lane A. Store stock, (s,S) restock with fixed lead time, picking queue, riders with straight-line travel, as Numba kernels. `[tests]`
  AC: conservation tests pass; a scenario with low stock produces stockout intervals.
- [ ] **P-014 Tape v1** Lane A. Writer in `py/tape`, decoder in `packages/tape`, fixture tapes and cross-language goldens. `[tests] [deep]`
  AC: TS decodes Python-written fixtures with matching checksums and probe values; streaming decode of day chunks works.
- [ ] **P-015 Stage core** Lane B. `Stage.create`, WebGPU with WebGL2 fallback, render loop, resize, DPR caps, quality tiers, perf HUD (dev only).
  AC: runs on Chrome, Safari 26, Firefox; HUD shows backend and frame time.
- [ ] **P-016 City and residents layers** Lane B. Instanced buildings from the pack, roads, sky, fog; residents as GPU points evaluated from event times at `uTime`, including the amber flare. `[deep]`
  AC: scrubbing anywhere in 30 days updates instantly; 20k residents at 60 fps on the reference laptop.
- [ ] **P-017 Stores layer** Lane B. Beams, radius rings, red flicker during stockout intervals from the tape.
- [ ] **P-018 Timeline and readouts** Lane B. Docked timeline with play/pause, speeds, scrubber with day ticks, event markers and adoption sparkline; readouts card.
  AC: keyboard accessible; works at 390 px wide.
- [ ] **P-019 Fixture route** Lane B. `/dev/fixture` loads the toy city pack and fixture tape from MinIO (or local files) without auth.
  AC: the M1 demo: open the route, press play, watch the city warm. Record a 20-second capture for MEMORY.md.

## M2 Real city
- [ ] **P-020 OSM bake** Lane C. Geofabrik extract clipped to the Ahmedabad bbox -> buildings with heights, road graph, river polygon, districts. `[tests]`
  AC: counts and height distribution reported; pack renders in the stage.
- [ ] **P-021 Population synthesis** Lane C. WorldPop raster -> households per building with iterative proportional fitting to district marginals. `[tests] [deep]`
  AC: district totals within 1% of marginals; no building over capacity.
- [ ] **P-022 Persona assignment** Lane C. Sample from Nemotron-Personas-India (Gujarat slice, Ahmedabad district first) to residents by age, sex, education and occupation marginals; derive segments. `[tests]`
  AC: segment shares within tolerance of source marginals; attribution written to the manifest.
- [ ] **P-023 Social graph** Lane C. Households, neighbours (distance decay), friends (homophily on segment), work ties (long range), target mean degree 5 to 8. `[tests] [deep]`
  AC: degree distribution, clustering and share of cross-river ties reported and within targets.
- [ ] **P-024 Store placement and travel times** Lane C. Greedy max-coverage placement within 2.5 km radii; OSRM travel times store -> H3 cell per time band. `[tests]`
  AC: coverage of residents >= 90%; travel table complete.
- [ ] **P-025 City pack v1** Lane C + A. Full format from ARCHITECTURE.md section 5 with meshopt-compressed buildings; Python and TS loaders. `[tests]`
  AC: Ahmedabad pack under 6 MB compressed; loads in under 1.5 s on the reference laptop.

## M3 Full engine
- [ ] **P-030 Campaign model** Lane A. Budgets, CPM, targeting, frequency caps, multiple channels. `[tests]`
- [ ] **P-031 Occasions and time-of-day** Lane A. Segment occasion profiles, evening peaks, weekend shift. `[tests]`
- [ ] **P-032 Out-of-stock behaviour** Lane A. Spillover, wait, give up, satisfaction. `[tests]`
- [ ] **P-033 Riders and routing** Lane A. Shifts, assignment, travel from `travel.bin`, rain slowdown, late flags. `[tests]`
- [ ] **P-034 Scenario events** Lane A. rain, festival, match night, price change, store closed. `[tests]`
  AC: event isolation test (MODEL.md section 10).
- [ ] **P-035 Runner service** Lane A. FastAPI `POST /runs` backed by a process pool, streams day chunks to MinIO, progress to Redis, returns tape URI and hash. `[tests]`
- [ ] **P-036 Benchmarks** Lane A. pytest-benchmark suite and py-spy profiles; move hot loops into Numba; meet ARCHITECTURE.md section 9 engine budgets on 100k residents.

## M4 Full stage (the jaw-drop)
- [ ] **P-040 Chatter arcs layer** Lane B. GPU arcs from tape chatter, travelling heads with tails, per-moment cap. `[deep]`
- [ ] **P-041 Riders layer** Lane B. GPU riders on L-shaped or road-following paths with trails.
- [ ] **P-042 Weather layer** Lane B. Rain particles, darker fog, wet road and river sheen, all driven by `rain_intensity` from the tape.
- [ ] **P-043 Camera director** Lane B. Intro flight with district-by-district light-up, focus on selection, idle drift, scripted shots; reduced-motion cuts.
- [ ] **P-044 Picking** Lane B. GPU id-buffer picking for residents and stores; CPU spatial-hash fallback. `[tests]`
- [ ] **P-045 Post-processing and tiers** Lane B. Bloom, tone mapping, vignette; auto tier selection from intro frame timings.
- [ ] **P-046 Visual regression** Lane B. Playwright stills (desktop + phone, three sim times) compared with tolerance; wired into `just shots` and CI for stage PRs. `[tests]`
- [ ] **P-047 Build-the-city transition** Lane B. Buildings rise in a wave from the river outward and residents light up when a new city pack loads.

## M5 Product shell
- [ ] **P-050 Auth and orgs** Lane A. Google OAuth + email magic link, sessions, orgs. `[tests]`
- [ ] **P-051 Brief composer** Lane B. Sentence-style form ("I'm launching [product] at [price] in [city] on [apps] with [budget] on [channels]") -> scenario via contracts. `[tests]`
- [ ] **P-052 Runs API and live progress** Lane A. Create run, WebSocket progress, seed 1 streams live while others run. `[tests]`
- [ ] **P-053 City view** Lane B. Stage + timeline + layers + readouts + legend, wired to real runs.
- [ ] **P-054 Resident and store cards** Lane B. Stories derived from tape events; store stock sparkline with stockout spans.
- [ ] **P-055 Plan view** Lane B + A. Fan chart of adopters (P10 to P90), per-store stock plan table, next-city ranking, calibration panel, CSV export. `[tests]`
- [ ] **P-056 Compare view** Lane B. Two scenarios as two cities side by side with a shared timeline (e.g. Rs 49 vs Rs 59). `[deep]`
- [ ] **P-057 Share links** Lane A. Read-only run pages with signed links. `[tests]`

## M6 Minds
- [ ] **P-060 Deep persona sampling** Lane C. Stratified sample of ~300 per city run from the persona pool. `[tests]`
- [ ] **P-061 Elicitation** Lane C. Instrument, JSON schemas, Batches API client, cache, cost guard. `[tests]`
- [ ] **P-062 Calibration** Lane C. Answer-to-parameter mapping with intent deflator; NumPyro hierarchical model; posterior draws to object storage; diagnostics. `[tests] [deep]`
- [ ] **P-063 Backtests and honesty meter** Lane C. CSV upload, fit days 1 to 10, forecast 11 to 30, coverage and MAPE. `[tests] [deep]`
- [ ] **P-064 Resident life logs** Lane C. Compile a resident's tape events + profile into a retrievable memory. `[tests]`
- [ ] **P-065 Text interviews** Lane C + B. Grounded replies that cite tape events; uncited claims dropped; interview UI in the resident card. `[tests]`
- [ ] **P-066 Voice interviews** Lane C + B. Streaming STT -> reply -> streaming TTS, barge-in, Hindi, Gujarati and English; latency budget. Provider chosen by a spike at the start of this point. `[deep]`
- [ ] **P-067 Narration** Lane C. Run story where every sentence links to tape events; shown at the top of the Plan view. `[tests]`

## M7 Orchestration and scale
- [ ] **P-070 Temporal LaunchRun** Lane A. Workflow + activities per ARCHITECTURE.md section 3; cancel and progress. `[tests]`
- [ ] **P-071 Monte Carlo aggregation** Lane C. DuckDB bands, stock plan, ranking. `[tests]`
- [ ] **P-072 Caching by content hash** Lane A. Packs, elicitations, calibrations, runs. `[tests]`
- [ ] **P-073 Observability** Lane A. OTel across services; dashboards for runs, LLM cost, frame-time reports.
- [ ] **P-074 Quotas and cost guard** Lane A. Per-org daily caps on seeds and LLM spend. `[tests]`

## M8 Launch
- [ ] **P-080 Landing page** Lane B. Live city hero on a lightweight demo tape, sentence-style brief as the call to action.
- [ ] **P-081 Public demo** Lane B. One preset scenario anyone can open without signing in.
- [ ] **P-082 Film capture** Lane B. Camera director script rendered to a 60-second video for the README and socials.
- [ ] **P-083 Deploy** Lane A. Images, Coolify services, domain, TLS, backups, smoke tests.
- [ ] **P-084 Write-up** Lane A. README polish, architecture diagram, a case-study post on the hybrid engine and event-time rendering.
