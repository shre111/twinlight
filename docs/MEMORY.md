# Memory

Working memory for coding sessions. Read at the start of every session; update at the end of every point or session with `/wrap`. Keep it short: current truth only, newest first in each section. Move anything stable into DECISIONS.md or ARCHITECTURE.md instead of letting it pile up here.

## Now
- Milestone: M0 Foundations
- Point: P-001 Monorepo scaffold (not started)
- Branch: none
- Next: P-001, then P-002 and P-003 can run in parallel lanes

## Recent changes
- 2026-11-03 Repo seeded with planning docs, Claude Code agents, skills, hooks and CI from the starter kit.

## Gotchas
- The git hook blocks multi-line commit messages, heredoc messages, `--no-verify`, force pushes, commits on `main` and badly named branches. Use a single `-m "type(scope): subject"`.
- `design/prototype/twinlight-city.html` is the visual reference for the stage (WebGL, single file, toy sim inlined). It is not production code; match its look, not its structure. Its scale clamps (point size, fog distance) were tuned by eye and are a good starting point for P-016.
- Shaders: never `pow()` a value that can go negative; it produced NaN black blocks on real GPUs in the prototype that headless renders did not show (DESIGN.md, shader rules).
- Engine golden hashes are bit-exact only on the pinned CI platform; locally compare probes and aggregates (D-006).
- Workspace network may block CDNs and font hosts; stage visual tests must load three.js and fonts from node_modules.

## Open questions
- License before the repo goes public (owner decides).
- Voice provider for Hindi and Gujarati streaming (spike at P-066).
- Whether to keep 15-minute ticks once real occasion profiles exist (revisit after P-031 benchmarks).

## Demo log
- (add a line with a link or path whenever a milestone demo is recorded)
