# Lane B (stage)

Worktree `.claude/worktrees/stage`. Lead folds this into MEMORY.md at each wrap.

## Now
- This PR: the camera, ground, sky, beam height, road lifts and wall-clearing nudge now scale from the pack's manifest bbox (`StageCity.bounds`, or the buildings if absent), relative to the toy city they were tuned on. The toy city is unchanged (all shots match).
- Lane C's `ahmedabad-osm` pack (P-020) frames as a whole city in `/dev/stage?pack=ahmedabad-osm`. With all of it in frame (100k buildings, about 238k road segments, 100k residents with glow), the M4 with WebGPU at DPR 1.75 stays at 16.7 ms p50 but drops frames at the high tier: 51 to 57 fps, p95 17.7 to 33.4 ms across three samples. Medium (DPR 1.5) runs 57 fps, p95 17.3 ms. The toy city's 100k holds 60 fps, so this is real-city scale, not residents alone, and about 1M D-021 buildings will need LOD or culling (P-025, P-045).
- Next: P-018 timeline and readouts.

## For P-045 (post-processing)
When bloom lands, remove the resident glow from `packages/stage/src/layers/residents.ts`:
- the second quad in `pointAndGlowQuads()` and its `withGlow` parameter, the `glow` attribute and `isGlow`, and the `glow` option of the `ResidentsLayer` constructor (and `TIERS[tier].bloomScale > 0` at its call in `stage.ts`);
- the glow branches of `sizeNode`, `positionNode` (including `offAxisCos` and the `cameraNear` import) and `colorNode`;
- `GLOW_SIGMA_PX`, `GLOW_QUAD_SIGMAS`, `GLOW_ENERGY`, `GLOW_REFERENCE_RESIDENTS`, `GLOW_PEAK_PER_ENERGY`, `glowDensity`, `settledGrowth`, `settledRadius`, `glowEnergy` and `glowPeak`;
- the `density` parameter of `#createMaterial`, which also scales the flare halo (`flareGain`), so decide then whether the halo stays at all;
- the tests for `glowDensity` and for the two-quad layout in `test/residents.test.ts`.
This pass also softened the flare (`FLARE_SPAN` 2 to 2.5, flare growth 1.6 to 1.2, halo falloff 0.9 to 0.5); retune those with bloom on. Keep `MIN_PHYSICAL_PX`, which is independent of bloom. If bloom lands at reduced strength, lower `GLOW_ENERGY` before deleting.

## Gotchas
- The glow quads sit on each resident's view ray just past the near plane, so a resident hidden behind a building adds faint haze over the face in front of it; real bloom only spreads visible light. The step along the ray is divided by the off-axis cosine because the near plane clips on view depth, not ray length.
- Flare halos also scale with `glowDensity`, so flares are dimmer at 100k than at 48k. That is deliberate: dense fresh-order blocks went hot without it.
- Building faces measured slightly darker than the render, not brighter. What made the skyline compete was the missing warm haze over lit districts.
- The resident point floor is physical: 1.3 logical px at DPR 1.75 or more, otherwise 2.3 physical px. The design renders use `POSES.hero` at DPR 2; `/dev/stage` uses `POSES.home`, so pixel comparisons by region are approximate.
- The shots baselines are SwiftShader stills at DPR 1 (desktop) and 3 (phone); they show the look, not GPU fill cost. `toHaveScreenshot` at a 1% pixel ratio passed a clearly visible light-only change; tighten it for stage stills in P-046.
