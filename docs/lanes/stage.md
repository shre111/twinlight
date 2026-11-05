# Lane B (stage)

Worktree `.claude/worktrees/stage`. Lead folds this into MEMORY.md at each wrap.

## Now
- This PR: the camera, ground, sky, beam height, road lifts and wall-clearing nudge now scale from the pack's manifest bbox (`StageCity.bounds`, or the buildings if absent), relative to the toy city they were tuned on. The toy city is unchanged (all shots match).
- Lane C's `ahmedabad-osm` pack (P-020) frames as a whole city in `/dev/stage?pack=ahmedabad-osm`. With all of it in frame (100k buildings, about 238k road segments, 100k residents with glow), the M4 with WebGPU at DPR 1.75 stays at 16.7 ms p50 but drops frames at the high tier: 51 to 57 fps, p95 17.7 to 33.4 ms across three samples. Medium (DPR 1.5) runs 57 fps, p95 17.3 ms. The toy city's 100k holds 60 fps, so this is real-city scale, not residents alone, and about 1M D-021 buildings will need LOD or culling (P-025, P-045).
- Next: P-018 timeline and readouts.

## For P-045 (post-processing)
- The resident glow's on/off is now a `glow` option of the `ResidentsLayer` constructor (next to a new `scale` option); remove that option rather than a positional argument when bloom lands.
