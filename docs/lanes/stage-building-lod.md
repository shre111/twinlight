# P-048 building LOD and culling: draft

Draft from Lane B for the owner, the lead and whoever builds P-048. It is not yet a plan of record. Numbers come from a throwaway spike on Lane C's bake of real Ahmedabad buildings; no stage code changed.

Terms: a *warm* city is one partway through a run, with lit residents. The *glow* is the second quad each resident draws, marked in `layers/residents.ts` as a bloom stand-in until P-045. *Cutoff* means the projected size below which a building is not drawn. A *level 6 tile* is one cell of the 64 x 64 grid over the city, about 433 x 381 m, in the pack's proposed tile index (`docs/lanes/minds.md`, "P-025 pack layouts").

## In short
- **The glow, not the buildings, is what stops the whole city holding 60 fps today.** On a warm city, cutting 1.05M buildings to about 90k leaves the median at 33.3 ms, though it does halve p95 from 65.7 to 34.3 ms. Turning the glow off with every building drawn gives 59.3 fps, p95 17.7 ms.
- **So the glow has to get cheaper before P-048's acceptance test can pass.** P-045's real bloom is the planned way, but its cost is unmeasured and it would change the look. Cheaper options are listed below, also unmeasured.
- **LOD is still worth building.** WebGL2, which Firefox uses, reaches only 45 fps at 1.05M. M1-class GPUs and phones have less to spare, and arcs, riders and rain have yet to arrive.
- **The approach:** one draw call over a compact list of visible buildings, rebuilt when the view changes. Tiles cull, a size cutoff drops what is too small to see, and a per-tier budget caps the total. The spike's list rebuild took 1.8 ms.

**How this sits with D-031 and P-048.** D-031 reasons that 100k buildings at 51 to 57 fps means the building count is the limit. These measurements suggest that shortfall was largely the glow. D-031's budget decision does not depend on this, but its reasoning needs a note, and moving P-045 changes the plan order, so both go through the lead. I read P-048's "holds the full building set" as: the whole set is loaded and available, and drawn with a size cutoff. Please correct me if the intent is every building drawn every frame. That is possible on WebGPU on the M4, but not on WebGL2.

## What was measured
M4 MacBook, Chrome, 1470 x 867 at pixel ratio 1.75, default tier, home pose unless stated. Scenes have the bake's 100,001 residents, 112,373 road segments, 24 stores and 63 water polygons unless stated. 835k is the plain 30 m² filter. 1.05M is every baked building, slightly above the 1,025,875 first-load buildings in Lane C's latest figures. The warm tape is the toy sim's launch run over the pack, days 13 to 15, playing at 4x, not a real engine tape. Machine load was 2 to 7, from other sessions. Single runs, so small differences are noise.

**Warm city, whole city in frame**

| Buildings drawn | Glow | Backend | fps | p50 ms | p95 ms |
|---|---|---|---|---|---|
| 1.05M, no cutoff (today) | on | WebGPU | 34 | 33.3 | 65.7 |
| 1.05M cut at 1.5 px, about 190k | on | WebGPU | 32.9 | 33.3 | 34.3 |
| 1.05M cut at 2 px, about 90k | on | WebGPU | 33.6 | 33.3 | 34.3 |
| 1.05M, no cutoff | off | WebGPU | 59.3 | 16.7 | 17.7 |
| 1.05M, no cutoff | off | WebGL2 | 45.4 | 16.7 | 33.6 |
| 1.05M cut at 1.5 px, about 190k | off | WebGL2 | 45.1 | 16.7 | 34 |
| 99k, the 835k set cut per building at 2 px | off | WebGL2 | 60 | 16.7 | 17.3 |

WebGL2 barely moves between 1.05M and 190k, then clears the budget by 99k. The last row is a different set from the others, and I don't have a point between them, so the WebGL2 limit is not pinned down.

**What each element costs, on an idle tape.** Each row is measured on its own against the base of sky, ground, river and stores. GPU pass time is from three's timestamps. On this pipelined GPU a pass can outlast the frame and still hold 60 fps, so read it as relative cost, not as budget.

| Scene | GPU pass ms | fps |
|---|---|---|
| Base only | 8.3 | 60 |
| Base and roads | 11.5 | 60 |
| Base and 100k residents, glow off | 12.0 | 60 |
| Base and 100k residents, glow on | 25.8 | 60 |
| Base and 835k buildings | 16.0 | 60 |
| Everything with 835k buildings, glow off | 19.5 | 60 |
| Everything with 835k buildings, glow on | 44.2 | 36 |

Each glow quad is 10 logical pixels wide, about 300 physical pixels, so 100k residents blend about 30M fragments a frame, 8 times the screen. It costs more with buildings in the scene (+24.7 ms against +13.8 ms alone), and I have not explained why.

**How big buildings look, and what culling gets.** The 835k set, with size as the larger of long side and height, in physical pixels. The first-load set adds about 190k small buildings, nearly all below the cutoff at a distance.

| Camera | Distance | In frame | Median size | 1 px or more | 2 px or more | 3 px or more |
|---|---|---|---|---|---|---|
| Overhead, zoomed out | 59 km | 835k, 100% | 0.4 px | 45k | 5.8k | 1.5k |
| Home | 23 km | 815k, 98% | 1.1 px | 456k | 95k | 33k |
| Half zoom | 11 km | 498k, 60% | 1.8 px | 466k | 219k | 95k |
| Quarter zoom | 5.6 km | 168k, 20% | 3.4 px | 168k | 153k | 101k |
| Closest | 2.6 km | 35k, 4% | 7.2 px | 35k | 35k | 35k |

Frustum culling does nothing for the acceptance view and most of the work up close. The size cutoff is the reverse. The busiest view is mid zoom, about 220k at 2 px, not the whole city.

**Draw calls are expensive in three.js.** JavaScript time per `render()` call, with 835k buildings and the glow off:

| Layout | Draw calls in the frame | render() JS ms |
|---|---|---|
| One mesh (today) | 12 | 0.8 |
| 2 km tiles, one mesh each | 188 | 4 |
| 1 km tiles, one mesh each | 658 | 26.7 |

So a level 6 tile cannot be a draw call: there are thousands of them. They can be ranges copied into one mesh instead. With all 1.05M buildings in 3,621 occupied 433 m tiles at a 2 px cutoff (the pack's real table has 3,988, so slightly more), that took 1.8 ms median and 2.6 ms p95, rebuilt every frame on purpose, including frustum tests and the cutoff search. This was with today's 68 bytes per building, not the 8-byte layout. The frame stayed at 12 draw calls and held 60 fps with p95 17.3 ms, on an idle tape with the glow off. Binning and sorting took 226 ms at load, on the main thread in the spike.

## Getting the glow's cost back
None of these is measured. They should be tried against the warm row above before committing to an order.
- **Real bloom (P-045).** The planned route, and a fixed full-screen cost instead of one per resident. It changes the look: today's glow draws in front of everything, so hidden residents still add haze, and bloom applied after the depth test loses that.
- **Half the quads at double gain.** `glowDensity` already scales gain by resident count, so this may look the same for half the fill.
- **A smaller quad.** It spans 2.5 sigma either side. Going to 2 cuts its area by about a third.
- **Skipping faint glows.** Collapse the quad in the vertex stage for far residents whose contribution is negligible.
- **A half-resolution glow pass.** A quarter of the fragments, with an upscale step.

## The approach

**At load, in the pack worker (D-032, which the lead is recording).** Decode the columns, then sort buildings by size inside each tile; tile ranges stay where they are, so the format is unaffected. Compute each tile's bounds. Keep a permutation back to pack order, because `residents.building` indexes it.

**One draw, compact instances.** Replace `InstancedMesh` matrices (64 bytes each, plus height) with the pack's 8-byte layout as instance attributes, decoded in the vertex shader. That cuts GPU memory for 1.03M buildings from about 70 MB to about 8 MB, and allocating only to the budget makes it smaller still. The per-building colour seed moves from `instanceIndex` to position, because indices now change as the list changes. That changes every building's shade, so stills get rebaselined.

**When the view changes.** Rebuild when the view matrix, projection, pixel ratio or budget changes, compared each frame, not on input events: camera damping keeps moving after input stops, and a later camera director may move it during playback. Walk the tile quadtree, skip what is outside the frustum, binary-search each visible tile's sorted sizes for the cutoff, copy those ranges into one buffer and upload the used part. Playback and scrubbing do not rebuild unless the camera moves.

**A budget, not a fixed cutoff.** Start at 2 physical pixels and raise the cutoff until the total fits the tier's budget. One mechanism handles the mid-zoom peak and slower backends. Budgets are guesses to tune: 250k on `high`, 120k on `medium` and WebGL2, 40k on `low`. Add hysteresis so the cutoff does not flicker at a boundary.

**Boxes only.** The home camera is 74 degrees from straight down, so walls fill more of a building than its roof, which rules out a roof-only far LOD. Removing the bottom face, which is never seen, saves a sixth of the triangles.

**Avoiding a pop.** The CPU picks a conservative cutoff per tile, using the tile's nearest point. The shader applies the same formula per building: full height above the band, shrinking to nothing at the cutoff. Because the band sits above the cutoff, it adds no instances beyond a hard cut at that size, though the far side of a tile draws some extra zero-height instances. Under reduced motion a building simply appears. Not built, and the extra instances are unmeasured, so this needs checking before it is relied on, in motion, not from stills.

**Keeping the city's mass.** On a cool city at home pose, 835k and 99k are nearly indistinguishable. On a warm city they are not quite the same: in dense districts more lights show through because the boxes that hid them are gone, and there is less dark mass. A size cutoff also drops small buildings first, which at a distance means the walled city's pol houses that D-031 went out of its way to keep. So a ground tint from a per-tile coverage map is part of this plan, not a fallback. Check warm stills at 1.5 and 2 px, with the walled city and the neighbourhood warming as the test.

**The detail file.** At 1,025,875 first-load buildings it holds at most about 29,000 footprints, so one fetch is enough, with no tile index. A 5 m footprint reaches 2 px at about 5 km, so fetch when the camera passes about 8 km, merge in the worker, and let the same shrink band grow them in. If it never loads, the city is still complete.

**What this touches.** Rendering only, so no simulation number changes. Residents keep their own positions and the layer does not read buildings. P-047's wave needs each building's distance from the river in the shader, perhaps from a distance texture. Its height factor multiplies with the shrink band, and a building that enters the list mid-wave must join at the right phase. That needs designing with P-047.

## Build order
Depends on the glow getting cheaper, above, for the acceptance test. The steps below can start before that.
1. Compact 8-byte instances, one mesh, no culling, on the P-025 loader. Same layout on screen, less memory, stills rebaselined for the colour seed.
2. Tile culling and the cutoff with a budget, the coverage tint, a frame report on the real bake.
3. The shrink band and hysteresis.
4. The detail file on zoom-in.

## What is not known yet
- Bloom's cost and look, and whether a cheaper glow is enough.
- Any GPU other than the M4: no M1, RTX 3060 or phone.
- Uploading on WebGL2, and the rebuild with the 8-byte layout and level 6 tiles. Timing is from WebGPU at 68 bytes and 433 m tiles.
- 8-byte attributes through TSL on both backends. Fallback is about 16 bytes, still far smaller than today.
- A real engine tape, with a real pattern of flares.
- Mid-zoom frame rate with the cutoff, which is the busiest view. The spike measured home pose and one zoomed-in view without it.

## For the owner
1. **The glow.** Try the cheaper options first, or go straight to P-045? I would measure the cheap ones first: a day's work could get a real city from about 33 fps to 60 on the M4 without waiting for P-045. Until then a warm real city runs at about 33 fps at the default tier.
2. **Closest zoom.** The camera's limits scale with the city, so on Ahmedabad it stops at 2.6 km, where the median building is only 7 px. Closer would suit resident cards, but makes the 8-byte rounding more visible; at about 700 m, 0.42 m is 1.3 px.
3. **Home pose.** It was tuned on the 7 km toy city. Scaled to Ahmedabad's 27 km, at this low angle much of the city reads as haze. Worth revisiting before the demo.
