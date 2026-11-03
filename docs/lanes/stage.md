# Lane B (stage)

Worktree `.claude/worktrees/stage`. Lead folds this into MEMORY.md at each wrap.

## Now
- P-010's "pack loads in the stage" holds (#36): `/dev/stage?pack=<name>` with `TWL_PACK_DIR` set to the parent folder of generated packs.
- Lane C's `ahmedabad-osm` pack (P-020) decodes and passes the pack tests (sizes, road offsets, all 100,488 residents on their footprints). The render check in `/dev/stage` is next; the camera poses, target clamp and ground glow are tuned to the toy city and need fitting to a 27.6 x 24.4 km pack.
- This PR: a residents legibility pass on the lead's day 13 notes. A glow quad per resident stands in for bloom, so lilac and mint read as a field over the city. M4 WebGPU at day 13: 48k residents read like the render at full view; 100k has no blobs or rims and holds 16.7 ms p50 and 17.7 p95 in 8 draw calls.
- Next: P-018 timeline and readouts (stage clock with play, pause and speed in the render loop; docked timeline; readouts card; 390 px layout).

## For P-045 (post-processing)
- Remove the resident glow quad when bloom lands: the second quad in `pointAndGlowQuads()`, the `glow` attribute, the glow branch of the size, position and colour nodes, the `GLOW_*` constants and `glowDensity`. Then decide whether the flare halo stays. If bloom lands at reduced strength, lower `GLOW_ENERGY` before deleting it. Keep `MIN_PHYSICAL_PX`.
- The glow quads sit on each resident's view ray at 1.5 x near, so a resident hidden behind a building adds faint haze over the face in front of it. Real bloom only spreads visible light, so expect the city to read slightly less hazy after the swap.

## Gotchas
- Building faces measured slightly darker than the render, not brighter. What made the skyline compete in the day 13 still was the missing warm haze over lit districts, which the glow now supplies.
- The resident point floor is physical: 1.3 logical px at DPR 1.75 or more, otherwise 2.3 physical px. The prototype tuned it at DPR 1.75, and its stills render at DPR 2 from `POSES.hero`, while `/dev/stage` uses `POSES.home`. Pixel comparisons by region are approximate.
- `toHaveScreenshot` at a 1% pixel ratio passed a clearly visible light-only change. Tighten it for stage stills when P-046 wires CI.
- SwiftShader frame times did not move with twice the vertices and the extra glow fill; they say nothing about GPU fill cost.
