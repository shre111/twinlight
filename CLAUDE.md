# Twinlight

Twinlight rehearses a product launch inside a living synthetic city before it happens. A brand describes a launch (product, price, city, channels, budget). Twinlight builds the city (buildings, roads, 100k+ synthetic residents, dark stores, riders), runs the launch for 30 simulated days many times over, and shows it as a 3D city at dusk that warms from cool to amber as people hear about the product and order it. Output: which neighbourhoods adopt first, which dark stores stock out and when, how much to stock per store, which city to launch in next, all as ranges, plus the ability to interview any resident by voice about why they did or did not buy.

Two things make this project what it is. Protect both in every change:
1. **The experience.** A cinematic, interactive 3D city that feels alive. Visual quality and smoothness are features, not polish.
2. **The engine.** A deterministic, calibrated hybrid simulation (LLM personas inform a classical agent-based model; the LLM is never on the per-tick path). Every number shown is reproducible from a seed and carries an interval.

## Read before working
- `docs/PLAN.md` the point you are on and its acceptance criteria
- `docs/MEMORY.md` current state, gotchas, open questions
- `docs/DECISIONS.md` settled decisions; do not relitigate them, add a new one to change course
- `docs/ARCHITECTURE.md` system shape, formats, APIs
- `docs/MODEL.md` before touching `py/engine`, `services/minds` calibration, or anything that changes a number
- `docs/DESIGN.md` before touching anything visible

## Repo map
```
apps/web          Next.js 16 app (UI shell, pages); mounts the stage
apps/gateway      NestJS API + WebSocket (auth, scenarios, runs, plans)
apps/worker       Temporal TypeScript worker (workflows, activities)
packages/stage    three.js scene engine, framework-agnostic (WebGPU with WebGL2 fallback, TSL materials)
packages/tape     TS decoder for run tapes + types
packages/ui       React components, design tokens as CSS vars
packages/contracts zod schemas shared by web, gateway, worker
packages/config   shared tsconfig presets
py/engine         Python simulation core, twl_engine (NumPy + Numba, deterministic)
py/tape           Python tape writer/reader, twl_tape
services/runner   Python FastAPI service that executes runs and uploads tapes
services/minds    Python FastAPI: personas, elicitation, calibration, interviews, voice
tools/citybake    Python: OSM + population + personas -> city packs
infra/            docker compose, otel, deploy
fixtures/         golden hashes, fixture tapes, packs and scenarios
design/           tokens.json, City prototype + toy sim, screen designs (see design/README.md)
docs/             the documents listed above
```

## Commands
```
just setup          install toolchains and deps (pnpm, uv)
just dev            just infra + all apps in watch mode
just infra          compose up (postgres, minio, redis, temporal, otel-lgtm), wait healthy, create buckets
just down           stop the dev infrastructure (volumes kept)
just lint           biome + ruff + pyright
just test           all test suites
just test-engine    uv run pytest py (engine + tape, includes determinism goldens)
just bake <city>    build a city pack, e.g. just bake ahmedabad
just fixture        run the fixture scenario end to end and print the tape hash
just fixture-pack   build the toy city pack that /dev/fixture loads, into data/cache/packs/toy
just goldens        regenerate golden hashes and probe files (only for intended output changes)
just shots          stage visual regression (Playwright stills, desktop + phone); --update to rebaseline
```
Commands that do not exist yet are created by the plan point that needs them. Keep this list current.

## Git workflow (enforced by `.claude/hooks/git-guard.mjs`)
- Never commit to `main`. One branch per plan point or fix: `feat/p032-residents-layer`, `fix/tape-offset-overflow`, `chore/bump-three`. Allowed prefixes: `feat/ fix/ chore/ docs/ refactor/ perf/ test/`. Lowercase, hyphens.
- **Micro commits.** Each commit is one small, coherent step that builds. Message is a single line, Conventional Commits, max 72 chars, lowercase start, no trailing period, **no body, no trailers, no co-author lines**: `feat(stage): add residents points layer`.
  Types: feat fix chore docs refactor perf test build ci style. Scopes: web gateway worker stage tape ui contracts engine runner minds citybake infra design docs plan memory decisions ci deps.
- **Micro PRs.** Aim for under ~300 changed lines excluding tests, fixtures and generated files. One concern per PR. A plan point may take several PRs.
- PR title = the same one-line convention. PR body = at most three short lines (what, why, how it was tested). No generated-by footers.
- Flow per PR: branch -> micro commits -> `git push -u origin <branch>` -> `gh pr create` -> wait for CI (`gh pr checks --watch`) -> `gh pr merge --squash --delete-branch` -> `git switch main && git pull` -> next. The `/ship` skill does this.
- Never `--no-verify`, never force-push, never rewrite `main`.

## Code rules
- **Minimal comments.** Code explains itself through names and small functions. Comment only non-obvious math, a protocol detail, or a workaround with its reason. No banner comments, no commented-out code, no restating what the next line does, no TODO without a plan point id.
- TypeScript strict, no `any` without a reason. Biome formats and lints TS/JS/JSON. Python 3.13 in one uv workspace (root `pyproject.toml`), ruff (format + lint), pyright basic, type hints on every public function.
- Engine code: residents are NumPy struct-of-arrays advanced with vectorised operations each tick. Loops that cannot vectorise (the logistics event queue, rider assignment) go in Numba `@njit` functions over preallocated arrays. No per-resident Python loops in the tick path.
- Shared shapes live in `packages/contracts` (zod) and are the single source for web, gateway and worker.
- No new dependency without a one-line reason in the PR body. Prefer the platform and existing deps.
- Config through env vars validated at startup. Secrets never in code, tapes, logs or the client.

## Testing policy
Automated tests are required for **big features**, not for every small change.
- Required: any plan point marked `[tests]`, and any change to engine semantics, tape or city-pack formats, public API contracts, calibration math, auth, quotas or cost guards.
- Not required: copy, styling, small UI tweaks, refactors covered by existing tests, docs.
- Kinds: engine pytest unit tests + hypothesis property tests + determinism goldens (same seed twice gives the same tape hash; golden hashes checked in). Cross-language goldens (Python writes, TS decodes, checksums match). TS vitest. Gateway e2e (supertest against a test DB). Python pytest (+ hypothesis for calibration). Stage visual regression with Playwright stills.
- Write the test to fail first when practical. Use the `test-author` subagent for big features so the implementation is not grading itself.

## Determinism rules (engine, tape, citybake)
- All randomness from `numpy.random.Generator` streams spawned from `SeedSequence(run_seed)`, one child stream per subsystem, passed in explicitly. Never the global `np.random`, `random`, wall clock, `os.urandom` or `hash()` in the sim.
- Integer sim clock in minutes. No floating time accumulation.
- No iteration over `set`s in sim code paths; use arrays, lists or sorted keys.
- No `parallel=True`, `prange` or `fastmath=True` in Numba sim code. Parallelism is across seeds (separate processes), never inside one seed.
- Golden hashes are bit-exact on the pinned CI platform (Linux x86_64, locked NumPy and Numba). Other machines check probe values and aggregates within tolerance, because NumPy's SIMD math can differ in the last bit across CPUs.
- Any intended change to sim output updates the golden hashes in the same PR, with the reason in the PR body.

## Design rules (details in `docs/DESIGN.md`)
- Use tokens from `design/tokens.json` via `packages/ui`; no literal colours or fonts in components.
- Display face Big Shoulders, body face Atkinson Hyperlegible Next. Sentence case. No all-caps labels, no monospace labels, no emoji, no decorative gradients, no "A · B" meta strings.
- The city is the hero. UI chrome stays quiet, docked to edges, never covering the centre of the city on desktop.
- 60 fps budget at the default quality tier; check with `just shots` and the perf HUD before merging stage work.
- Every insight shown in 3D also exists as text or a table somewhere (Plan view). Respect `prefers-reduced-motion`.

## LLM rules
- The LLM never runs inside the simulation tick loop. It elicits deep-persona answers before a run, writes interview replies and narrations after.
- Every LLM call is cached by `(persona_id, scenario_hash, prompt_version, model)`, goes through the cost guard, and is logged with tokens and cost.
- Interview answers and narrations must cite tape events; uncited claims are dropped.
- Synthetic people only. Never real customer PII, never cloned real voices.

## Status files
- `docs/PLAN.md`: tick a point only when its acceptance criteria hold and CI is green.
- `docs/MEMORY.md`: update at the end of every point or session, terse. Current point, what changed, gotchas, next.
- `docs/DECISIONS.md`: append-only. New decision IDs continue the sequence. Use `/decide`.
- In parallel lanes only the lead lane edits these three files; other lanes write to `docs/lanes/<lane>.md` and the lead folds them in.

## Parallel lanes
Use git worktrees (`claude --worktree <name>`, files under `.claude/worktrees/`, gitignored). `.worktreeinclude` copies `.env` files into new worktrees.
- Lane A, lead (main checkout): engine, tape, contracts, status files.
- Lane B, stage (worktree `stage`): `packages/stage`, `apps/web` scene pages.
- Lane C, minds (worktree `minds`): `services/minds`, `tools/citybake`.
Lanes own disjoint paths. Cross-lane contract changes go through Lane A first (`packages/contracts`, tape format).

## Subagents and skills
Subagents in `.claude/agents/`: `engine-smith` (Python sim engine), `scene-director` (3D and motion), `persona-lab` (personas, calibration, interviews), `test-author` (tests for big features), `reviewer` (read-only review against these rules). Use them for focused work and run independent ones in parallel.
Skills in `.claude/skills/`: `/start-point`, `/ship`, `/decide`, `/golden-test`, `/scene-check`, `/wrap`.

## Definition of done for a point
Acceptance criteria met; tests added if required; CI green; `reviewer` has no blocking findings; for visible work `/scene-check` passes; MEMORY.md and PLAN.md updated; PRs merged with one-line squash commits.
