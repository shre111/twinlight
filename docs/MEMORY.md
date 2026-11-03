# Memory

Working memory for coding sessions. Read at the start of every session; update at the end of every point or session with `/wrap`. Keep it short: current truth only, newest first in each section. Move anything stable into DECISIONS.md or ARCHITECTURE.md instead of letting it pile up here.

## Now
- Milestone: M0 Foundations
- Point: P-002 Python workspace (not started)
- Branch: none
- Next: P-002 first PR (root uv workspace, members by glob), then in parallel the rest of P-002, P-003, P-004 and P-007; P-005 after P-004; P-006 last

## Recent changes
- 2026-11-03 P-001 done: pnpm workspace, Turborepo, Biome, `packages/config`, justfile (#1); Next.js 16 app with the wordmark on dusk (#2).
- 2026-11-03 Repo seeded with planning docs, Claude Code agents, skills, hooks and CI from the starter kit.

## Gotchas
- Node 24 is pinned in `.nvmrc`, but this machine's nvm default is 22: run `nvm use` (or prefix commands with it) before pnpm or just.
- TypeScript is pinned to 6.0: the native 7.0 has no JS API yet, which Next's build type-check and the NestJS CLI use.
- Biome 2.6 also formats Markdown and YAML; `biome.json` excludes those, `.claude/` and `design/` (D-018 scopes Biome to TS/JS/JSON).
- `next dev` writes `apps/web/AGENTS.md` unless `agentRules: false` (set). Next 16's own docs ship in `apps/web/node_modules/next/dist/docs/`; read them before relying on remembered APIs.
- P-007 must replace the stopgaps in `apps/web/app/globals.css` (hand-copied colour vars, literal font fallback stack) and the relative `node_modules` font path in `apps/web/app/fonts.ts`.
- `ci.yml` was written before the code: the engine job's `pytest -m golden` exits 5 when no golden tests exist, and the python matrix lints `services/minds` and `tools/citybake` together. P-002 and P-003 keep their jobs green; P-006 reworks CI with a single `ci-ok` gate.
- No branch protection on `main` (owner's choice for now); the git hook and squash-only repo settings are the only guards.
- The git hook blocks multi-line commit messages, heredoc messages, `--no-verify`, force pushes, commits on `main` and badly named branches. Use a single `-m "type(scope): subject"`. Squash-merge deletes the branch, so each PR of a point gets its own branch.
- `design/prototype/twinlight-city.html` is the visual reference for the stage (WebGL, single file, toy sim inlined). It is not production code; match its look, not its structure. Its scale clamps (point size, fog distance) were tuned by eye and are a good starting point for P-016.
- Shaders: never `pow()` a value that can go negative; it produced NaN black blocks on real GPUs in the prototype that headless renders did not show (DESIGN.md, shader rules).
- Engine golden hashes are bit-exact only on the pinned CI platform; locally compare probes and aggregates (D-006).
- Workspace network may block CDNs and font hosts; stage visual tests must load three.js and fonts from node_modules.

## Open questions
- License: the repo is public with no license (owner accepted public; licence still to pick).
- Golden platform pinning (proposed D-021 at P-006): `NUMBA_CPU_NAME=generic` plus NumPy CPU feature limits, and `just goldens` via a CI workflow.
- Voice provider for Hindi and Gujarati streaming (spike at P-066).
- Whether to keep 15-minute ticks once real occasion profiles exist (revisit after P-031 benchmarks).

## Demo log
- (add a line with a link or path whenever a milestone demo is recorded)
