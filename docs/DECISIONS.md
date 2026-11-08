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
Status: Accepted (2026-11-03); streaming by appended chunks dropped by D-033
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
Status: Accepted (2026-11-03); refined by D-023
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
Status: Accepted (2026-11-03); amended by D-021, D-022 and D-025
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

### D-021 Fill OSM building gaps from VIDA's combined footprints
Status: Accepted (2026-11-03); refined by D-024
- **Decision:** Citybake adds building footprints from VIDA's combined Google + Microsoft + OSM dataset (source.coop `vida/google-microsoft-osm-open-buildings`, `country_iso=IND` GeoParquet), reading only the city bbox with DuckDB and caching it under `data/cache`. Footprints are deduplicated across all three sources with priority OSM > Google > Microsoft: a footprint is dropped when more than 30% of its area overlaps a higher-priority one (VIDA does not deduplicate). Heights: OSM `height` or `building:levels` when present, else the median Google Open Buildings 2.5D Temporal 2023 `building_height` (4 m grid) inside the footprint, else the district default. Synthetic boxes in populated cells with no footprint come last and are flagged `synthetic` in the pack.
- **Why:** OSM has about 100k buildings in the Ahmedabad bbox and almost none in the walled city and the south (Vatva, Narol, Lambha), where WorldPop puts dense population, so P-021's "no building over capacity" cannot hold on OSM alone. The combined dataset gives real shapes there under open licences.
- **Alternatives:** Synthetic boxes only (keeps D-013 unchanged, but whole districts look procedural rather than like the city); leaving the gaps and relaxing the capacity check (residents pile into the few buildings that exist).
- **Consequences:** Pack manifests and the app's attribution credit VIDA, Google Open Buildings and Microsoft Global ML Building Footprints alongside OSM; check each licence's terms when implementing and record them in the manifest. The bake report shows per district the source mix (OSM, Google, Microsoft, synthetic) and the share of WorldPop population with no footprint. Reproducibility depends on the pinned dataset release and cache, like the pinned OSM extract.

### D-022 Travel times on the baked road graph instead of OSRM
Status: Accepted (2026-11-03)
- **Decision:** P-024 computes store-to-cell travel minutes in citybake with `scipy.sparse.csgraph.dijkstra` over the directed road graph (oneway tags respected), all stores as sources, one pass per time-of-day band with per-road-class speeds. A travel-backend interface keeps OSRM available as a later backend.
- **Why:** Everything stays in Python and reproducible, with no routing container in CI or on the laptop. Within a 2 to 3 km delivery radius, OSRM's better turn handling barely matters next to the model's own travel noise (lognormal, sigma 0.15), and riders on two-wheelers don't follow perfect routes anyway.
- **Alternatives:** OSRM in Docker during the bake (better routing realism, but a container dependency for every bake and in CI, on a machine where Docker is fragile).
- **Consequences:** One-off check when P-024 lands: about 30 store-to-cell pairs compared by hand with Google Maps two-wheeler times, median error under 20%, result noted in MEMORY.md. `travel.bin` keeps its format; only its source changes.

### D-023 Pin CPU features for strict goldens
Status: Accepted (2026-11-03)
- **Decision:** The CI engine job, and `just goldens` when P-011 adds it, run with `NPY_ENABLE_CPU_FEATURES=X86_V3` (an allow-list, so current and future AVX-512 groups stay off), `NUMBA_CPU_NAME=generic`, `NUMBA_DISABLE_INTEL_SVML=1`, `OPENBLAS_CORETYPE=Haswell` and `OPENBLAS_NUM_THREADS=1`. A guard test, active when `TWL_GOLDEN_STRICT=1`, asserts through `numpy._core._multiarray_umath.__cpu_features__` that `X86_V3` is on and `X86_V4` is off, and CI prints `np.show_runtime()`.
- **Why:** GitHub's Linux runners mix CPUs with and without AVX-512, and NumPy dispatches different SIMD kernels for `exp` and `log` on each, which can flip the last bit and make strict goldens flaky. NumPy 2.5 groups features as `X86_V2`/`X86_V3`/`X86_V4` and silently ignores old names such as `AVX512F`, so the guard makes a renamed group fail loudly instead of flaking.
- **Alternatives:** No pinning, fixing the first flake when it appears (slow to diagnose); the pinned container image D-006 originally described (it does not exist, and an image does not pin the CPU).
- **Consequences:** "Pinned CI platform" in D-006 now means Linux x86_64 at `X86_V3` with these variables. A small CI-only speed cost. Strict hashes come from CI; other machines compare probes with tolerance. Check the group names on every NumPy upgrade.

### D-024 Correct 2.5D raster heights against OSM storeys
Status: Accepted (2026-11-04)
- **Decision:** Citybake corrects raster-sourced building heights with a monotone fit to OSM-tagged storeys. The tagged sample is binned by raster median height (1 m bins), each bin taking the median of `building:levels x 3.2` weighted by count; `scipy.optimize.isotonic_regression` fits the bins; corrected heights interpolate linearly between bin centres (no plateaus), and above the top bin they keep the top bin's ratio instead of clipping. The correction applies only to raster-sourced heights. The raw raster value stays as a column, floors are derived as corrected height / 3.2 for every source, and the fitted knots go into the pack manifest.
- **Why:** Google's 2.5D heights run low against OSM storeys (median ratio 0.82 over 18,557 tagged buildings; 0.96 at 1 to 2 storeys, 0.71 at 5 to 7, 0.65 at 13+), so tall buildings rendered about 30% short and held too few floors for residents.
- **Alternatives:** Raw raster heights (towers about 30% short); one global factor of 1 / 0.82 (over-corrects the low-rise majority and under-corrects towers).
- **Consequences:** Validation holds out whole districts, not a random sample, and the bake report gives the median absolute percentage error per storey band (1 to 2, 3 to 4, 5 to 7, 8 to 12, 13+) before and after correction. The fit is deterministic given the pinned OSM extract and raster tiles. Refined 2026-11-04 on real data (the owner): bin mean instead of median (the median snapped to storey multiples and put 70% of buildings at 6.4 m), the sparsest top bins pooled until each holds at least 30 buildings, the above-top ratio floored at 1.0 so a tall raw reading never shrinks, floors = max(1, round(height / 3.2)), and the report shows median signed error by raster bin (near zero) and by storey band (a low-high tilt is expected from a mean-calibrated curve). Below the lowest bin the bottom bin's ratio applies.

### D-025 Population from GHS-POP, with a sourced-total override
Status: Accepted (2026-11-04); amended by D-030
- **Decision:** Citybake takes population from JRC GHS-POP R2023A (2025 epoch, 3 arcsec, EPSG:4326, CC BY 4.0, © European Union, JRC) instead of WorldPop. The `[worldpop]` city-config section becomes a generic `[population_raster]` (url, md5, zip member, source and licence). The city config also carries an optional `population_total` with `source`, `polygon` (an OSM boundary id, e.g. the AMC limit or a UA polygon) and `as_of`; when set, the raster is rescaled inside that polygon, not the bbox, so a Census 2027 provisional total can replace a projection. For Ahmedabad no rescale is applied: GHS-POP's 8.08M in the bbox already matches the city.
- **Why:** WorldPop R2025A put 3.74M people in the bbox (AMC alone had 5.58M in 2011) and no 100 m cell above 130 people, while the walled city is 50k+/km²; it under-concentrates dense cores (Mumbai shows the same). GHS-POP is built from built-up volume: 8.08M in the bbox, walled city 55.5k/km², resident weight 14.8 against the design's "about 16". Cell-level correlation between the two is only 0.69.
- **Alternatives:** Rescale WorldPop to a sourced total (keeps the smoothing and the core/suburb imbalance); keep WorldPop (weights 6.5, absolute counts about half). Fallback if a future city's GHS-POP also under-concentrates: distribute the sourced total by residential floor area (footprints × D-024 floors, excluding commercial and industrial) instead of rescaling the raster.
- **Consequences:** Attribution adds GHS-POP. The feasibility check for a new city compares the bbox total and the core-to-suburb density ratio against census ward counts where available. GHS-POP's 2025 epoch is a projection from a 2020 baseline, like WorldPop's.

### D-026 Six persona segments by an ordered rule table
Status: Accepted (2026-11-04)
- **Decision:** `segment` is one of six, assigned by an ordered first-match rule table over Nemotron persona fields, versioned in the city config: `senior` (age 60+), `young_single` (under 30, never married), `homemaker` (female, no occupation), `working_graduate` (occupation and graduate or technical diploma), `working_other` (occupation, below graduate), `other_adult` (the rest). `other_adult` stays separate: partial pooling handles its size, and merging it would give non-working men work-hours occasion profiles and work ties. Household couples: the first adult draws from the pool; if married, the second adult is a married persona of the other sex, with the husband usually older (a gap of about 0 to 8 years, rarely reversed).
- **Why:** MODEL.md hangs `alpha_s`, `ref_price_s` and the occasion profiles on `s(r)`. These six separate the obvious quick-commerce behaviours, every group is large enough to calibrate (Ahmadabad pool: 32% working_other, 28% homemaker, 14% young_single, 11% senior, 11% working_graduate, 3% other_adult), and the rules are transparent, so calibration can split one later.
- **Alternatives:** Five segments with other_adult folded into working_other (wrong occasion profile and ties for non-working men); segments from clustering (opaque, unstable across cities).
- **Consequences:** Persona marginals are district-wide, so east and west of the city would behave alike; a household affluence tier (three levels from built form: building height, footprint, district, plus ward literacy if available) that shifts `alpha` and `beta` and conditions persona sampling is a plan point before P-062. Pinned by the HF dataset commit sha and the `en_IN` parquet list; only persona uuids enter the pack, never the text.

### D-027 Anchor tests for awareness and adoption: Bass limit and prototype parity
Status: Accepted (2026-11-05)
- **Decision:** The Bass-limit test runs a reduced configuration, because the full model cannot produce a Bass curve: 50,000 identical residents; word-of-mouth targets drawn uniformly from all residents (`wom_mixing = "uniform"`); awareness decay, talk decay, hearsay and seen-nearby off; flat time-of-day; one exposure enough to be aware; aware residents order in the same tick; no supply limit. `p` is impressions per resident per tick and `q` expected talks per adopter per tick, chosen so the 50% point falls between day 10 and 15. Pass: the mean of 8 seeds is within 1 percentage point (absolute) of the Bass curve at the end of every day. A second configuration with `q = 0` must match `1 - exp(-p t)` within 0.5 points, so a paid-term bug cannot hide behind a word-of-mouth bug. The prototype-parity test uses a checked-in scenario, `fixtures/scenarios/prototype_launch.toml`, on the 48,000-resident toy city (day-1 ad burst, daily drip, a creator burst at hour 186 aimed at Satellite, Bodakdev and Vastrapur, unlimited stock, no rain). At most three parameters are fitted (paid reach, word-of-mouth rate, purchase intercept), on days 6 and 13 only; everything else stays at the MODEL.md section 11 defaults. Pass: the mean of seeds 1 to 8 is within 3 points of the prototype's heard and ordered shares on days 6, 13 and 30, with day 30 out of sample.
- **Why:** MODEL.md's "within 2%" was not testable as written (relative error is dominated by seed noise near zero, and talk decay, awareness decay, hearsay, the 0.9 talk gain and occasion waits all break the closed form). A correct engine sits within 0.3 points of the curve, while a 10% error in `q` shows as up to 7 points and a 20% error in `p` as about 5, so 1 point separates them. Fitting three parameters on two days and checking the third day tests that the dynamics carry forward, not that a fit can hit six numbers. Stockouts and rain move the prototype's numbers by at most 0.5 points, so the comparison holds without the supply model.
- **Alternatives:** Relative 2% on every day (needs far more seeds to pass honestly); fitting all six numbers with free parameters (proves little); asserting the east/west split now (MODEL.md has no district-level affinity until P-061b).
- **Consequences:** Fitted values are checked in next to the scenario; CI never refits, and a refit is a script run whose diff is in the PR. The prototype's east/west ordered split (about 33% and 12%) is reported and asserted in an `xfail` test tied to P-061b. MODEL.md section 11 lists the v1 defaults and pins the readings the formulas left open (rates per hour, tick order, what `eps` and `t_o` mean, crossing attribution). Monotonicity is tested on medians over seeds with stated step sizes and a small margin, since runs with different prices do not share random draws.

### D-028 The tape records the talks that mattered: heard and nudged
Status: Accepted (2026-11-05)
- **Decision:** Every talk raises the listener's awareness strength, but the tape's CHATTER section records only two kinds, in a new `u8 kind` column: `1 heard`, the talk that takes a resident over the awareness threshold (the one `heard_from` points at), and `2 nudged`, the last talk a resident received in the 72 hours before their first order, when that is a different talk.
- **Why:** Recording every talk would be 100k to 250k rows per seed, most of the 3 MB tape budget, and most of those arcs would mean nothing. With these two kinds every arc in the city means something, interviews can cite the nudge, and the section stays at about one row per aware resident plus at most one per adopter.
- **Alternatives:** Every talk (tape budget); heard only (loses the nudge that interviews and narration want); heard plus a random sample (adds rows without meaning).
- **Consequences:** MODEL.md section 3 and ARCHITECTURE.md section 4 say which talks are recorded; the stage can draw nudged arcs fainter. A nudged row carries its talk's time, which can be up to 72 hours before the order that reveals it, so rows are sorted when the tape is written, and P-014's day chunks must either accept chatter rows up to three days late or hold chatter back by three days.

### D-029 Word of mouth: one full-strength talk is enough, and adopters keep talking
Status: Accepted (2026-11-05)
- **Decision:** A talk adds `talk_gain * w_uv` to the listener's awareness with `talk_gain = 1.0` (was 0.9), so one talk over a tie of strength 1 reaches `theta_aware`. An adopter's talk rate fades from its level at the latest order to a floor: `lambda_uv(t) = q_u * w_uv * talk_kind * (talk_floor + (1 - talk_floor) * exp(-(t - t_o) / 72h))` with `talk_floor = 0.35`; hearsay from aware non-adopters still fades to zero. Both are named MODEL.md section 11 defaults that calibration (P-062) may move.
- **Why:** With the model as first written and three parameters fitted on days 6 and 13, the prototype-parity test's day-30 heard share came out at 54.3% against the prototype's 63.1% (ordered was on target). Two structural differences explained it: a 0.9 gain sits below the 1.0 threshold, so word of mouth alone needed two talks inside the awareness half-life; and talk faded to zero 72 hours after an order. Refitted on days 6 and 13 only, the gain change alone gave 58.3% (this already includes the fade restarting at every repeat order, which the engine does), the floor alone 57.5%, and both together 61.1% heard and 24.3% ordered. The owner's rule was: adopt the gain; prefer the repeat-order restart to a floor if day 30 then passed; otherwise add the 35% floor, which is the prototype's own constant.
- **Alternatives:** Restart at repeat orders without a floor (ties sustained talk to behaviour, but day 30 stayed 4.8 points low); the floor without the gain change (5.6 points low); keeping the model and dropping the day-30 assertion (long-run word of mouth weaker than the prototype's).
- **Consequences:** Day 30 informed this choice, so it is no longer a clean out-of-sample check. Day 20 (prototype 51.9% heard, 16.1% ordered), never used for fitting or for choosing the model, is asserted from now on as the held-out check, within 3 points. Real packs have tie strengths well below 1 outside households (friend mean 0.60, neighbour 0.39, work 0.35), so there a single talk still does not reach the threshold; calibration sets `q` and may revisit the gain. Engine goldens regenerate.

### D-030 Correct the population raster with a built-form fit, checked on ward polygons
Status: Accepted (2026-11-06)
- **Decision:** Citybake corrects the population raster inside the city as a general step. It fits people per m² of residential floor area as a function of built form (height class, footprint-size class, building coverage in the cell) on reference polygons named in the city config (for Ahmedabad five OSM talukas), applies the fit across the bbox, combines it per cell as `raster^(1-w) * fit^w`, and rescales to the bbox total (8.08M). For Ahmedabad `w = 0.75`, with the passing range `[0.68, 0.84]` stored next to it. Buildings inside OSM `landuse=commercial|retail|industrial` and `amenity=university|college|school|hospital` areas are excluded from residential floor area. The correction is validated on DataMeet's 48 AMC ward polygons (CC BY-SA 2.5 IN; used for checks only, never shipped in a pack) against the 2015 delimitation's 118,001 people per ward in 2011: the walled-city wards (Dariyapur, Jamalpur, Khadia, Shahpur) as a group mean within 0.75 to 1.25, the 48-ward total within the city growth band of 1.27 to 1.6, plus bands for Ghatlodiya taluka, a north-west box and the part of Daskroi in the bbox. A failed check blocks personas, graph and stores.
- **Why:** GHS-POP put the five north-west wards at 0.32 of their own 2011 count (Bodakdev about 7,000 people/km² beside Navrangpura's 84,000 on near-identical building stock), so store placement left Bodakdev, Thaltej, Vastrapur, Memnagar, Sola and Gota uncovered. The step's cause is unknown; it is not the 2011 taluka boundary. A geometric blend with floor area kept too much of it, and the pure fit ran high (built form cannot see that western homes hold fewer people per m²).
- **Alternatives:** The raster as it is with more stores (the west stays under-populated in every run); a 50/50 blend with floor area (north-west box 27% under its band); the pure fit (15% over the taluka band, over-fills Daskroi); a crowding term for the old core (new modelling fitted to four wards).
- **Consequences:** At `w = 0.75` with the exclusion: walled-city mean 0.817, 48-ward total 1.300 of 2011, Ghatlodiya taluka 706,515 (band 405k to 830k), north-west box 1,085,718 (band 970k to 1.42M), Daskroi part 197,509 (up to 409k); north-west five 1.365 of 2011 (raster 0.321); ward spread (log std) 0.403 against the raster's 0.749; Navrangpura 2.69 (raster 5.79). Dariyapur (0.56) is a recorded known miss. Stores rise from 15 to 24 at the same 0.92 coverage target, 11 of them on the west bank. These are not independent validation: the rule (group mean in place of per-ward limits, the weight chosen by ward spread) changed after seeing the ward data, the same wards informed the weight, and the range's edges rest on rough bands. Minimising spread across wards assumes equal growth, so it likely understates the fast-growing north-west; it is not a default rule for other cities, which state their own weight and evidence. Class rates vary across the city (fold variation 25%) while the outcome is stable (15 refits, no band fails, shifts of at most 6%). Checks use integer households; before-figures use fractional people, and a single ward can differ by up to 0.16 between the two. Revisit `w` when AMC's ward-wise 2011 table or Census 2027 ward data is available.

### D-031 City pack budget: browser download, clustered small footprints kept, detail file
Status: Accepted (2026-11-06)
- **Decision:** The 6 MB pack budget is the first-load browser download (buildings, roads, residents, stores, compressed), not the whole pack. Footprints under 30 m² are dropped from the first-load buildings only when isolated; a small footprint in clustered fabric (touching or within about 3 m of another building) is kept, so the walled city's pol houses survive. Dropped footprints are written to an optional detail file in the pack, outside the first-load budget, for the stage to load on zoom-in. There is no confidence floor. Engine-only files (social graph, travel table, cells, engine resident columns) have their own soft budget of 10 MB, reported in the manifest. Residents whose building is dropped keep their position and are repointed to the nearest kept building.
- **Why:** D-021 took Ahmedabad to 1,054,627 buildings. Measured with the real meshopt and brotli encoders, everything is 6.05 MB and a plain 30 m² minimum 5.01 MB with 97.5% of floor area kept, but a size filter alone removes 5.3% of walled-city floor area and a confidence floor of 0.75 removes 44% of walled-city buildings, in the ward group that is already the weakest population check. The engine reads the graph and travel files; the browser never downloads them.
- **Alternatives:** A 50 m² minimum (4.2 MB, but 15.5% of walled-city floor lost); a Google confidence floor (thins the old city most); one 6 MB budget for every file (only about 440k to 515k buildings fit); keeping everything and raising the budget.
- **Consequences:** P-025 reports the walled-city floor area kept and confirms the first-load download stays within 6 MB with the clustering rule. The pack splits resident columns into a browser file and an engine file, and roads ship compact. No filter fixes frame rate: 100k buildings fully in frame already ran at 51 to 57 fps, so P-048 (building LOD and culling) must hold the full building set at 60 fps full-city on the M4 at the default tier. ARCHITECTURE.md section 9 states both budgets.

### D-032 City pack buildings and roads as delta-coded columns, not meshopt
Status: Accepted (2026-11-06)
- **Decision:** Buildings and roads ship as quantised, Morton-ordered, delta-coded columns like the rest of the v1 pack, not as meshopt-compressed glTF. The buildings file starts with a tile index (offset ranges over the Morton order) so the stage can cull and switch detail by contiguous ranges. Compression has one path: the pack uses zstd -19 if the first-load download still fits 6 MB and JS decoding meets the load target, because the tape decoder already decompresses zstd in the browser; otherwise brotli with objects stored as `Content-Encoding: br` so the browser decodes in the HTTP layer (never in JS: Chrome has no brotli `DecompressionStream`), verified in Chrome, Firefox and WebKit against local MinIO. The stage decodes columns in a Web Worker and transfers the typed arrays. Buildings and roads join the cross-language goldens (Python writes, TS decodes, checksums and probe buildings match).
- **Why:** Measured with the real encoders on the 1,026,145 first-load buildings (brotli 11): meshopt 4.78 MB against 4.05 MB for columns at 8-byte precision, 5.61 against 4.94 MB with the district, and roads 0.75 against 0.71 MB. Morton order makes position deltas nearly monotonic. Columns need no encoder: the Python meshopt wrapper is an alpha third-party package with no wheels or licence metadata, so meshopt meant running the JS encoder from citybake.
- **Alternatives:** meshopt in a .glb as first planned (fits 6 MB at 8-byte precision by only 60 KB, with the encoder problem above).
- **Consequences:** No glTF container; the stage reads buildings as typed-array columns. At zstd -19 the first-load total is 5.58 MB with 0.5 m sizes and a district column (5.19 MB with brotli); 0.1 m precision does not fit. JS zstd decode time for about 11 MB is measured at the start of P-014 and can reverse the zstd choice, so the pack writer keeps the compressor a single switch. P-025's text and ARCHITECTURE.md section 5 drop meshopt and the .glb.

### D-033 No tape streaming in v1
Status: Accepted (2026-11-06)
- **Decision:** A tape is one immutable object written when its seed finishes. The append-and-range-fetch design in ARCHITECTURE.md is dropped, and the tape's CHUNKS section stays reserved and empty. While seed 1 runs (a second or two), the client shows the city from the pack and "day N of 30" from the existing Redis progress events, covered by the intro camera move; seed 1 plays as soon as its tape lands, and the readouts' ranges tighten as seeds 2 to 50 arrive.
- **Why:** S3 and MinIO objects cannot be appended to, and a tape's section offsets, checksums and final hash are not known until the run ends. A seed takes about a second, so the city can appear from the pack at once and nothing is lost by waiting for the whole tape.
- **Alternatives:** One object per simulated day applied in order by the client (works, but adds a second format, a writer, a reader and an equivalence test for a wait the intro already covers); specifying day chunks now and building them later.
- **Consequences:** P-014 loses its streaming-decode criterion; P-035's runner uploads finished tapes; P-052 becomes "seed 1 plays as soon as its tape lands". Revisit when seed 1 takes longer than the intro on the production box, for example with bigger cities or longer horizons.

### D-034 Campaign channels
Status: Accepted (2026-11-08)
- **Decision:** A campaign names one of five channels: `social` (Instagram, Facebook), `video` (YouTube, OTT), `creators`, `in_app` (ads inside quick-commerce apps) and `outdoor` (hoardings). Responsiveness is `kappa_r,c = kappa_s * channel_kappa[c][s]` and presence is the campaign's own hourly weights times the channel's presence per segment; both channel factors are neutral (1) until calibration supplies them, except two rules that define a channel: `in_app` presence is each resident's ordering-occasion profile, because an in-app ad is only seen inside an app session, and an `outdoor` campaign must be targeted by district, because hoardings stand in a place. Each channel has a default CPM in rupees with its source (MODEL.md section 11), used when the brief gives none. Budgets stay per day per campaign, and the frequency cap stays per campaign per day.
- **Why:** The brief names channels and MODEL.md names `kappa_r,c` but no channels. There is no behavioural data per channel yet; calibration elicits each persona's channel habits (MODEL.md section 8). Neutral defaults avoid inventing behaviour, which leaves cost as the one real difference between channels available now.
- **Alternatives:** Three channels (social, creators, in-app); free-form channel names (the brief and calibration could not share keys); hand-set placeholder skews (invented numbers would read as findings); a 30-day budget schedule or automatic pacing (a burst and a drip are already two campaigns, split by the scenario builder); a cap per channel across campaigns (no evidence it matters yet).
- **Consequences:** Every campaign in a scenario carries `channel`, so scenario hashes and the golden tape hashes change once (probes do not). While channel defaults are neutral, the Plan view (P-055) labels any channel-versus-channel comparison uncalibrated instead of presenting it as a finding. The brief composer (P-051) uses the same five names. Showing hoardings where they stand in the stage would need site positions, not just districts.

