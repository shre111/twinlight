# Lane B (stage)

Worktree `.claude/worktrees/stage`. Lead folds this into MEMORY.md at each wrap.

## P-019
- Ready to tick. `/dev/fixture` plays the engine's fixture tape over the v2 toy pack, both written by `just fixture` (`data/cache/packs/toy`, `data/cache/tapes/toy.twlt`). `TWL_PACK_DIR` and `TWL_TAPE_DIR` default to those folders and `just shots` runs `just fixture` first. No auth, dev only; MinIO can come later. The stage reads the tape through `fetchTape` (`decodeTape` from `@twinlight/tape`, hash checked), and `fetchPack` decodes zstd packs by default.
- Capture: `artifacts/captures/p019-fixture-route.mp4` in the stage worktree, not committed (7.5 MB). 20 s, 1440x900, H.264 at 30 fps, from headed Chrome on WebGPU through a CDP screencast: open the route, press 16x and Play, Day 1 to Day 13.
- What it shows: Day 5 1,215 ordered and 21% heard; Day 9 2,204 and 29%; Day 13 4,177 and 40%, with 3 of 14 stores out and their beacons red. The first stockout is on Day 10 at 3:30 pm.
- Frames on the M4 in Chrome, WebGPU, tier high, 16x on the real tape with the readouts card and timeline visible: p50 16.7, p95 17.6 ms, 60 fps, 12 draw calls, and the same while recording. On battery at 65% with Low Power Mode off.
- The fixture route also passes in Firefox and Playwright WebKit (`SHOTS_ENGINES=1`, `-g fixture`).
- The `just shots` stills and frame report now use the fixture run (`/dev/stage?fixture=toy`), rebaselined once. The frame report measures 48k residents instead of 20k, so software GL timings start from a new reference (desktop high p50 583 ms against 350). The timeline label test stays on the toy run, since only it has a monsoon.
- Next: drop `pack-v1.ts`, since nothing writes v1 now.
- Timeline event labels use short names below 1400 px, a breakpoint tuned to the toy run's two campaigns and monsoon. When real events feed the timeline, labels need collision handling from their measured widths, and every event needs a `short` name.

## River layer
- The stage reads `river.bin` and draws any water ring: a dark ribbon whose slow sheen fades out toward the banks. Reduced motion holds the sheen. Rain sheen waits for P-042.
- `/dev/fixture` also passes in Firefox (WebGL2, p50 16.7 ms) and Playwright WebKit (WebGPU, 17.0 ms). The Safari app itself has not been opened.
- Frames on the M4 in Chrome, WebGPU, tier high, full runs at 16x with the river and readouts card: 48k p50 16.7, p95 18.4 ms; 100k p50 16.7, p95 18.2 ms; both 60 fps. With the river switched off under the same conditions, 48k read p95 18.5 ms, so the river adds nothing measurable.

## For Lane C
- `decodePack` now requires `river.bin` and the manifest's `riverPolygons` and `riverPoints` (ARCHITECTURE section 5). Only the toy fixture writes them so far; the citybake pack writer needs them too.

## Gotchas
- Open: Playwright WebKit once failed `Stage.create` at load above 15, logging `WebGL Device Lost` and a null-array `TypeError`, and showed "needs WebGPU or WebGL 2". Six reruns at load 8 passed. A context lost during creation reads as an unsupported browser, with no retry; not reproduced, so not changed.
- The first `next dev` compile of `/dev/fixture` on this branch reloaded the page about 1,500 times before it settled; later loads were normal. If a dev page stays blank, look for repeated GETs in the server log before suspecting the stage.
- Chrome caps animation at 30 fps on battery at low charge, so p50 33.3 ms can be the cap and not a regression. Time a page with no 3D before suspecting a change.
- `design/prototype/twinlight-city.html` shows its "needs WebGL" card over a working scene: the card's `display: grid` beats `hidden`. Hide `#fail` in devtools to see the reference.

## For P-040 (D-028)
- Add chatter with `kind` to `StageTape` and the synthetic tape, and draw nudged rows fainter.
- Do not assume time order: nudged rows can land up to 72 h late, across chunks.
