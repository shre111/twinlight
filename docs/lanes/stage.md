# Lane B (stage)

Worktree `.claude/worktrees/stage`. Lead folds this into MEMORY.md at each wrap.

## Now
- P-007 stays with the lead (merged in #8); Lane B started at P-015.
- P-015 Stage core, three PRs:
  1. `feat/p015-stage-core`: `packages/stage` (three 0.186.1 `WebGPURenderer`, WebGL2 fallback, loop, resize, per-tier DPR cap) and the dev route `/dev/stage` with `<StageHost>`.
  2. `feat/p015-perf-hud`: frame-time stats (p50/p95 ring buffer, vitest), `stage.perf()`, dev perf HUD.
  3. `chore/p015-stage-shots`: `just shots` (Playwright stills desktop 1440x900 and phone 390x844, baseline compare, `frame-report.json` per tier).
- Next: P-016 on a synthetic in-memory city and tape (adapter over `design/prototype/toy-sim.mjs`, dev route only) until P-010 and P-014 land.

## Measurements
- Real GPU (Apple M4, Chrome 156, WebGPU, empty scene, DPR 1.75): p50 16.7 ms, p95 17.6 ms, 60 fps.
- Prototype on the same machine at a 1837 px wide window, DPR 1.75: about 24 ms mean frame, so the prototype itself is above budget there; P-016 has to beat it.
- SwiftShader shots (headless Chrome): the stage falls back to WebGL 2; desktop 33 ms, phone 16.7 ms on an empty scene. Software numbers, only for regression comparison.

## Gotchas
- Next dev treats `127.0.0.1` as cross-origin and blocks its dev resources, so the page never hydrates there. Use `http://localhost:<port>`.
- Playwright's `toHaveScreenshot` takes `stylePath`, not `style`; an unknown `style` option is silently ignored. Shots hide the Next dev badge with `page.addStyleTag` on `[data-nextjs-dev-overlay]`.
- The prototype's WebGL failure card always shows (`.overlay { display: grid }` beats `[hidden]`); hide `#fail` to see the city. Served without a charset, the rupee sign is mis-encoded.
- Lane B runs its dev server on port 3101 to stay clear of the lead's 3000.
- Disk is at 98% (4.4 GiB free). Playwright WebKit and Firefox builds are not downloaded yet; `SHOTS_ENGINES=1 just shots` needs them.
- Shots baselines are from macOS Chrome with SwiftShader. CI (P-046) will need its own Linux baselines.

## Agreed with Lane A (2026-11-03)
- Tape decode names (`packages/tape`, P-014): camelCase of ARCHITECTURE.md section 4, minutes, `0xFFFFFFFF` = never.
  - residents: `awareAt`, `firstOrderAt` (Uint32), `orderCount` (Uint16), `heardFrom` (Uint32, never = paid), `firstSource`, `outcomeFlags` (Uint8).
  - chatter: `from`, `to`, `at` (Uint32, sorted by `at`).
  - deliveries: `store` (Uint16), `resident`, `orderedAt`, `pickedAt`, `deliveredAt` (Uint32), `outcome` (Uint8).
  - stores: `stock` (Float32), `orders` (Uint16), flat row-major `[store * ticks + tick]`; stockouts as CSR `stockoutOffsets` (Uint32, storeCount + 1) into `stockoutIntervals` (Uint32 start, end pairs).
  - aggregates per tick: `aware`, `adopted`, `orders` (Uint32), `storesOut` (Uint16), `rainIntensity` (Float32).
  - header: `simMinutes`, `tickMinutes`, `seed`.
  - Streaming (day chunks) changes the container, not these names. The synthetic tape sits behind one small adapter so the swap is mechanical.
- Toy pack buildings (P-010): `buildings.bin`, little-endian f32 SoA `x, y, width, depth, height, rotation` (radians about up, counter-clockwise from east); count in `manifest.json`. `buildings.glb` waits for P-025.
- Axes: pack is local metres from the bbox centre, `x` east, `y` north, `z` up. Stage maps to three.js as `(x, z, -y)`, so north is -Z. The prototype's toy sim is already in three.js axes (negative z is north), so pack `y = -z_proto`.
- P-015's Safari and Firefox checks wait for the laptop restart and free disk.
