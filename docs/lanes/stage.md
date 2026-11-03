# Lane B (stage)

Worktree `.claude/worktrees/stage`. Lead folds this into MEMORY.md at each wrap.

## Now
- P-015: `just shots` is the last code PR (`chore/p015-just-shots`). The browser AC is still open: Chrome passes (WebGPU, HUD shows backend and frame time); Safari 26 and Firefox need Playwright's WebKit and Firefox builds (`SHOTS_ENGINES=1 just shots`), waiting on the owner's go-ahead for the download.
- Next: P-016 (`[deep]`, scene-director on Fable) on a synthetic city and tape behind one adapter over `design/prototype/toy-sim.mjs`, dev route only. Planned PRs: synthetic inputs and `loadCity`/`setTape`/`setTime`; city layer (sky, ground, fog, roads, instanced buildings); residents layer (event-time points with the flare); dev scrubber plus `#still-day1/6/13` stills.

## Gotchas
- `just shots` scope today: `/dev/stage` stills on desktop and phone, baseline compare, and a per-tier `frame-report.json`. The scene-check skill's day stills arrive with P-016, card stills with P-054, CI wiring with P-046.
- Compare shots on p50 and p95. Mean fps under SwiftShader swings 41 to 52 between identical runs.
- Shots reuse a dev server on `SHOTS_PORT` (default 3101). If `just dev` already runs the web app, pass its port (`SHOTS_PORT=3000 just shots`) instead of starting a second `next dev` in `apps/web`.
- Playwright wipes its output directory each run. The run directory is named after the branch, with a `detached` fallback, so it can never be `artifacts/shots/` itself (which holds the baselines). In CI, missing baselines fail instead of being written.
- Playwright merges project options over top-level ones but skips `undefined`, so `channel: undefined` cannot unset a top-level channel. Set `channel` per project.
- `toHaveScreenshot` takes `stylePath`, not `style`, and silently ignores unknown options. Shots hide the Next dev badge with `page.addStyleTag`.
