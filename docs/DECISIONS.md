# Decisions

Append-only. To change a decision, add a new one that supersedes it and mark the old one `Superseded by D-xxx`. Format: decision, why, alternatives considered, consequences. Use `/decide` to add one.

---

### D-001 Monorepo with pnpm, Turborepo and one uv workspace
Status: Accepted (2026-11-03)
- **Decision:** One repo, two languages. TypeScript via pnpm workspaces + Turborepo; Python via a single uv workspace (root `pyproject.toml`) whose members live under `py/`, `services/` and `tools/`. `just` is the single entry point for cross-language tasks.
- **Why:** Contracts, tape formats and goldens cross the TS/Python boundary; one repo keeps them in lockstep and lets one PR change writer and reader together. One Python workspace means one lockfile and shared tooling for engine, runner, minds and citybake.
- **Alternatives:** Separate repos per service (drift, cross-repo PRs); Nx (heavier, JS-centric).
- **Consequences:** CI path-filters jobs so a docs PR runs nothing but the title check.

### D-002 Stage is an imperative three.js engine outside React
Status: Accepted (2026-11-03)
- **Decision:** `packages/stage` is plain TypeScript + three.js with a small imperative API. React mounts it once and sends it commands. No React Three Fiber.
- **Why:** The frame loop, GPU buffers and camera director must not depend on React reconciliation. Keeps the stage testable headless and reusable in the landing page and the film capture.
- **Alternatives:** React Three Fiber (pleasant composition, but per-frame work and GPU lifetimes leak into React state for this workload).
- **Consequences:** UI overlays that must track 3D points (labels, rings) read projected positions from a stage event, throttled.

### D-003 WebGPU renderer with WebGL2 fallback, materials in TSL
Status: Accepted (2026-11-03)
- **Decision:** `WebGPURenderer` from three.js, which falls back to WebGL2 automatically. All custom materials and the residents/arcs/riders shaders are written in TSL.
- **Why:** WebGPU ships in all major browsers since Safari 26 (Sept 2025); compute and lower CPU overhead matter at 100k entities. TSL means one shader source for both backends.
- **Alternatives:** WebGL2 only with GLSL (simpler, leaves performance on the table); raw WebGPU (no fallback, more code).
- **Consequences:** Pin the three.js version; upgrades are their own `chore/` PR with `/scene-check`.

### D-004 Render from event times, not frames
Status: Accepted (2026-11-03)
- **Decision:** The engine emits per-entity event times (aware_at, first_order_at, chatter (from, to, at), deliveries). The client uploads them once and shaders evaluate state at `uTime`.
- **Why:** Scrubbing to any moment is O(1); a 30-day run fits in a few MB instead of gigabytes of frames; live runs stream as appended chunks.
- **Alternatives:** Per-frame state snapshots (huge, slow to scrub); keyframes + deltas for everything (complex for monotonic state that event times express directly).
- **Consequences:** Non-monotonic state (stock levels) ships as time series. New visual states need a new event field in the tape (format version bump).

### D-005 Simulation engine in Python with NumPy and Numba
Status: Accepted (2026-11-03)
- **Decision:** `py/engine` (`twl_engine`) in Python 3.13. Residents are NumPy struct-of-arrays advanced with vectorised operations each 15-minute tick; the logistics event queue and other loops that do not vectorise are Numba `@njit` functions over preallocated arrays. Monte Carlo parallelism is across seeds, one seed per worker process.
- **Why:** One backend language shared with runner, minds and citybake (shared types, one lockfile, the team's strongest stack). Vectorised NumPy plus Numba gets within a small factor of compiled code for this workload. Seeds are independent, so process-level parallelism is simple and keeps each seed deterministic.
- **Alternatives:** Rust (faster and stricter, but a third language and slower iteration for this team); plain Python loops (far too slow at 100k residents); JAX (strong for vectorised maths, awkward for an event queue and integer state).
- **Consequences:** Budgets in ARCHITECTURE.md section 9 assume about 10 s per seed. Profile with py-spy and move hot loops into Numba before considering anything else. Revisit with a new decision only if benchmarks miss budgets by more than 2x after optimisation.

### D-006 Determinism contract
Status: Accepted (2026-11-03)
- **Decision:** `numpy.random.Generator` streams spawned from `SeedSequence(run_seed)`, one per subsystem and passed explicitly; integer minute clock; no set iteration in sim paths; no `parallel=True`, `prange` or `fastmath=True` in Numba sim code; parallelism only across seeds. Golden tape hashes are bit-exact on the pinned CI platform (Linux x86_64, locked NumPy and Numba) and checked across 1, 4 and 8 worker processes; other platforms check probe values and aggregates within tolerance.
- **Why:** Every number in the product must be reproducible from a seed, and reproducibility is what makes debugging and the narration's event links trustworthy. NumPy's SIMD transcendental functions can differ in the last bit across CPU generations, so bit-exactness is promised per platform rather than everywhere.
- **Alternatives:** Best-effort determinism (unreviewable drift); disabling SIMD globally (slow, and still not portable across builds).
- **Consequences:** `just goldens` regenerates hashes inside the pinned CI container image. Any intended output change updates goldens in the same PR with a reason.

### D-007 Gateway in NestJS, minds in FastAPI
Status: Accepted (2026-11-03)
- **Decision:** NestJS (TypeScript) for the API edge and WebSockets; FastAPI (Python) for personas, calibration, interviews and voice.
- **Why:** Matches the team's strongest stacks. TS at the edge shares zod contracts with the web app; Python owns the ML and Bayesian tooling.
- **Alternatives:** One FastAPI backend (loses shared contracts with web); all TypeScript (weak for NumPyro and data work).
- **Consequences:** Contracts in `packages/contracts`; minds exposes an OpenAPI spec the gateway client is generated from.

### D-008 Temporal for run orchestration, introduced in M7
Status: Accepted (2026-11-03)
- **Decision:** `LaunchRun` is a Temporal workflow (TypeScript worker) with idempotent activities. Until M7 the gateway calls the runner directly for single-seed and small runs.
- **Why:** Multi-step, long-running, fan-out work with retries, cancellation and visibility. Deferring it keeps the first vertical slice small.
- **Alternatives:** BullMQ (simpler, weaker for workflows); a hand-rolled state machine in Postgres.
- **Consequences:** Activities must be idempotent from day one (keyed by content hashes) so the migration in M7 is mechanical.

### D-009 Postgres + PostGIS + h3 for data; object storage for packs and tapes; DuckDB for run analytics
Status: Accepted (2026-11-03)
- **Decision:** Relational and geo data in Postgres 17 with PostGIS and h3-pg; city packs, tapes and posteriors in S3-compatible storage (MinIO locally); aggregation with DuckDB over Parquet inside minds. No ClickHouse.
- **Why:** Run outputs are immutable blobs read by id; analytics are per run and small enough for DuckDB.
- **Alternatives:** ClickHouse for all telemetry (operational weight for little gain at this scale).
- **Consequences:** Cross-run analytics (later) can add ClickHouse without touching the write path.

### D-010 Drizzle ORM with SQL migrations
Status: Accepted (2026-11-03)
- **Decision:** Drizzle in the gateway; migrations as SQL files; PostGIS and h3 through custom column types.
- **Why:** SQL-first, typed, handles geo types without fighting the ORM.
- **Alternatives:** Prisma (weak PostGIS support); TypeORM (heavier, decorators everywhere).
- **Consequences:** Geo queries are hand-written SQL in a repository layer.

### D-011 The LLM is never on the tick path
Status: Accepted (2026-11-03)
- **Decision:** LLM calls happen only in elicitation (before a run), interviews and narration (after). Calls are cached by `(persona_id, scenario_hash, prompt_version, model)`, batched where possible, and go through a cost guard. Default models are configurable: a small fast Claude model for bulk elicitation through the Batches API, a stronger one for interviews and narration.
- **Why:** Cost and determinism. A run makes thousands of LLM calls, not millions, and re-running a seed costs zero LLM calls.
- **Alternatives:** LLM agents per resident per tick (unaffordable, non-reproducible, and 2026 research favours hybrid designs).
- **Consequences:** Behaviour richness comes from calibrated parameters and the event model, not live LLM reasoning.

### D-012 Synthetic people only
Status: Accepted (2026-11-03)
- **Decision:** Residents are synthetic, derived from census-aligned synthetic persona datasets (Nemotron-Personas-India, CC BY 4.0) and population rasters. No real customer data enters the simulation; brand uploads are aggregate sales only. Voices are generated, never cloned from real people.
- **Why:** Privacy, DPDP Act exposure, and honesty about what the product is.
- **Consequences:** Attribution notices in the app and in each city pack manifest.

### D-013 City data from OpenStreetMap and WorldPop, baked into immutable packs
Status: Accepted (2026-11-03)
- **Decision:** `tools/citybake` builds content-addressed city packs from OSM extracts and WorldPop rasters. Dark-store locations are synthetic (coverage optimisation), not scraped from any platform.
- **Why:** Open licences, reproducible builds, no dependency on private platform data.
- **Consequences:** Building heights are estimated where OSM lacks them (levels x 3.2 m, else district defaults).

### D-014 Every number has an interval
Status: Accepted (2026-11-03)
- **Decision:** The UI never shows a bare point estimate for an outcome. Bands (P10 to P90) everywhere, the calibration state on every Plan view, and the product's language is "rehearsal" and "likely range", never "prediction".
- **Why:** Synthetic research is criticised for self-graded accuracy; showing uncertainty and backtest error is the differentiator.
- **Consequences:** Charts, cards and exports carry bands; copy review checks wording.

### D-015 Hybrid time stepping: 15-minute behaviour ticks, discrete events for logistics
Status: Accepted (2026-11-03)
- **Decision:** See docs/MODEL.md section 1.
- **Why:** Behaviour does not need minute resolution; queues and riders do.
- **Consequences:** Event ordering within a tick is by `(minute, seq)`.

### D-016 Testing for big features only, defined precisely
Status: Accepted (2026-11-03)
- **Decision:** Tests are required for `[tests]` plan points and for changes to engine semantics, formats, public contracts, calibration math, auth, quotas and cost guards. Small changes do not need tests.
- **Why:** Keeps velocity high on UI iteration while protecting the parts where a silent bug changes every number.
- **Consequences:** The reviewer subagent flags missing tests only against this list.

### D-017 Git conventions
Status: Accepted (2026-11-03)
- **Decision:** Branch prefixes `feat/ fix/ chore/ docs/ refactor/ perf/ test/`; one-line Conventional Commits with no body or trailers; micro PRs squash-merged with the PR title as the commit; no attribution footers. Enforced by `.claude/hooks/git-guard.mjs`, `attribution` settings, and a PR title check in CI.
- **Why:** A main history that reads as a clean changelog of small steps.
- **Consequences:** Repo settings: squash merge only, default squash message = PR title, delete head branches, branch protection on `main` with required checks.

### D-018 Lint and format: Biome, ruff and pyright
Status: Accepted (2026-11-03)
- **Decision:** Biome for TS/JS/JSON; ruff format + lint and pyright basic for Python. A PostToolUse hook formats files Claude edits.
- **Why:** Fast, few configs, no style debates in review.

### D-019 Visual identity
Status: Accepted (2026-11-03)
- **Decision:** Dusk indigo ground; light as data (tubelight mint for supply and the unaware city, lilac for word of mouth, sodium amber for orders, sindoor red for stockouts, monsoon blue for rain); Big Shoulders display with Atkinson Hyperlegible Next body. Details in docs/DESIGN.md.
- **Why:** Grounded in the subject: an Indian city at dusk, sodium lamps, tube lights. The city warming from cool to amber is the product's signature image.
- **Consequences:** No literal colours in components; tokens only.

### D-020 Deploy on a single VPS with Coolify first
Status: Accepted (2026-11-03)
- **Decision:** Docker images per service deployed with Coolify on one VPS; S3-compatible storage abstracted so it can move to R2/S3.
- **Why:** Cheap, already familiar, enough for demos and early users.
- **Consequences:** Runner replicas limited by the box; Monte Carlo default seeds tuned to fit.
