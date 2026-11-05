# Lane B (stage)

Worktree `.claude/worktrees/stage`. Lead folds this into MEMORY.md at each wrap.

## P-015
- Firefox passes: Playwright Firefox 157 took the WebGL2 fallback, HUD shows backend and frame time, 20k residents at p50 16.66, p95 17.28 ms, 59 fps, no console errors. Keyboard scrubbing works; Firefox rings the whole track rather than the thumb. Ready to tick.
- At 1440 px the dev HUD's last figure, pixel ratio, sits under the readouts card. Dev only.

## P-019
- `/dev/fixture` is in: `just fixture-pack` writes the toy pack to `data/cache/packs/toy`, the default for `TWL_PACK_DIR`, and `just shots` runs it first. No auth, dev only. Shows a message if the pack is missing.
- Not ticked. There is no TS tape decoder yet, so the tape is `synthesizeTapeFor(city)`: the toy sim's campaign over the pack's residents and stores, with traits drawn as the prototype draws them. Day 14 reads 43% heard, 11.5% ordered; day 30 reads 63%, 23%.
- To finish after P-014: swap `synthesizeTapeFor` in `apps/web/app/dev/stage/dev-stage.tsx` for the decoded fixture tape, move the shots stills to this route, then record the 20-second capture. Hold the capture until the tape has orders, so "Orders today" is not 0. MinIO can come later.
- Frames on the M4 in Chrome, WebGPU, tier high, one full run at 16x with the readouts card visible in every frame:
  - 48k residents on `/dev/fixture`: p50 16.7, p95 17.6, worst 17.8 ms, 60 fps, none over 20 ms.
  - 100k residents on `/dev/stage`: p50 16.7, p95 17.6, worst 33 ms, 60 fps, 2 of 2,697 over 20 ms.

## River layer
- The stage reads `river.bin` and draws any water ring: a dark ribbon whose slow sheen fades out toward the banks. Reduced motion holds the sheen. Rain sheen waits for P-042.
- `/dev/fixture` also passes in Firefox (WebGL2, p50 16.7 ms) and Playwright WebKit (WebGPU, 17.0 ms). The Safari app itself has not been opened.
- Frames on the M4 in Chrome, WebGPU, tier high, full runs at 16x with the river and readouts card: 48k p50 16.7, p95 18.4 ms; 100k p50 16.7, p95 18.2 ms; both 60 fps. With the river switched off under the same conditions, 48k read p95 18.5 ms, so the river adds nothing measurable.

## For Lane C
- `decodePack` now requires `river.bin` and the manifest's `riverPolygons` and `riverPoints` (ARCHITECTURE section 5). Only the toy fixture writes them so far; the citybake pack writer needs them too.

## Gotchas
- Chrome caps animation at 30 fps on battery at low charge, so p50 33.3 ms can be the cap and not a regression. Time a page with no 3D before suspecting a change.
- `design/prototype/twinlight-city.html` shows its "needs WebGL" card over a working scene: the card's `display: grid` beats `hidden`. Hide `#fail` in devtools to see the reference.

## For P-040 (D-028)
- Add chatter with `kind` to `StageTape` and the synthetic tape, and draw nudged rows fainter.
- Do not assume time order: nudged rows can land up to 72 h late, across chunks.
