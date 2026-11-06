# Lane C (minds)

Worktree `.claude/worktrees/minds`. Lead folds this into MEMORY.md at each wrap.

## D-030 population correction (merged #86 to #90; the lead records the decision)

GHS-POP gave the five north-west wards a mean of 0.32 of their 2011 size. `citybake population` now corrects each raster cell as `raster^0.25 * fit^0.75`, with `fit` from people per housing floor by built form on five OSM talukas, rescaled to 8.08M. Buildings in OSM commercial, retail, industrial, university, college, school and hospital areas house nobody. The weight was chosen after seeing the ward data, so the checks below are not out-of-sample.

| Check, real step | Result | Band | Raster before |
|---|---|---|---|
| Walled-city mean (4 wards) | 0.817 | 0.75 to 1.25 | 1.100 |
| 48-ward total / 2011 | 1.300 | 1.27 to 1.6 | 1.358 |
| Ghatlodiya taluka | 706,515 | 405,000 to 830,000 | 163,094 |
| North-west box | 1,085,718 | 970,000 to 1,420,000 | 359,444 |
| Daskroi part in the bbox | 197,509 | up to 409,000 | 116,359 |

- North-west five, reported only: 1.365 (raster 0.321): Gota 1.90, Thaltej 1.78, Bodakdev 1.53, Chandlodia 1.02, Ghatlodia 0.60.
- Navrangpura: raster 5.79, corrected 2.75, with the exclusion 2.69.
- 48 wards: median 1.09, p10 0.73, p90 1.98, min 0.56 (Dariyapur), max 2.78 (Nikol), log std 0.403 (raster 0.749). Known miss: Dariyapur 0.56. Jamalpur is 0.78, inside the band.
- Passing weight range on the real step: 0.68 to 0.84. 0.67 fails the box; 0.85 fails the walled city and the taluka. 0.75 sits near the middle, and is not an optimum.
- Stability: fold variation 25.1% floor-weighted, max 78.3%, so class rates differ across the city. Removing any reference taluka moves the walled mean and the north-west five by at most 3.0%, Navrangpura by 6.3%, and fails no band.
- Population stage on real data: 8,080,001 people against 8,080,000 (household rounding), 0 over capacity, IPF converged. All four stages are byte-identical across two runs on this machine.

Limits:
- Before and after use different measures: `raster_*` is fractional people, the checks are integer households. Same bake, a ward can differ by up to 0.158; Dariyapur is 0.60 against 0.53 before the exclusion.
- The bands rest on my growth and ward-count assumptions, and the 118,001 ward figure on search summaries attributed to the State Election Commission. AMC's ward-wise 2011 file or Census 2027 would tighten them.
- Minimising spread assumes equal growth, so the weight likely understates the north-west. Ghatlodia at 0.60 is low, and 5 wards sit under 0.75.
- The exclusion is by area: a home inside a mapped commercial or industrial polygon is removed. It took 13.8% of capacity and moved Ramol from 3.02 to 1.86 and Baherampura from 1.56 to 0.88.

## P-024 stores and travel times after D-030 (still needs the D-022 hand check)

`citybake stores ahmedabad`, `coverage_target` unchanged at 0.92: 24 stores cover 92.5%, west 11, east 13. It was 15 stores, west 5.

| District | Covered | Median nearest store | Before D-030 |
|---|---|---|---|
| Bodakdev | 100.0% | 905 m | 0% |
| Thaltej | 100.0% | 1,195 m | 0% |
| Vastrapur | 94.9% | 1,693 m | 0% |
| Memnagar | 100.0% | 1,961 m | 0% |
| Gota | 97.9% | 1,349 m | 0% |
| Sola | 34.5% | 2,726 m | 0% |

Districts are our own polygons. By the real ward polygons: Bodakdev 99.2%, Thaltej 91.1%, Chandlodia 97.3%, Ghatlodia 100%, Gota 63.4%. Gota is the largest uncovered ward (a 33.8 km2 ward), then Jodhpur 59.6%.

| Band | p50 | p90 | Max | Within radius p50 | Within radius p90 |
|---|---|---|---|---|---|
| morning_peak | 39 | 61 | 105 | 10 | 14 |
| midday | 31 | 49 | 84 | 8 | 11 |
| evening_peak | 44 | 70 | 120 | 11 | 15 |
| night | 25 | 39 | 67 | 6 | 9 |

- Table: 769,344 bytes raw, 99,012 zstd, 4.0% of pairs inside a radius. 4,007 hexes, median 19 residents.
- Hex to road node: p50 45 m, p90 140 m, max 1678 m (881 m before): the correction fills fringe hexes further from a road. 37 hexes are over 500 m from a road node, holding 174 residents. Minutes stop at the node.
- Load by nearest store, in represented shoppers: min 21,166, median 62,430, max 124,107. 5 of 24 stores stand on dead-end road nodes.
- Graph after the rebake, all targets met: mean degree 5.95, clustering 0.218, non-household cross-river 0.136, largest component 0.9996. Workers: west 25,583, east 27,699 (was 22,984 and 30,277).
- Evening travel alone is still p50 11 and p90 15 within radius, against MODEL's 10 to 20 minute promise including queue, picking and rider wait.
- Full travel table for P-025 is still small, and is the lead's call.

### Hand check for the owner (D-022)

Open each link, read Google's two-wheeler time, and compare with the band matching the time you look it up: morning 08:00 to 11:00, midday 11:00 to 17:00, evening 17:00 to 22:00, night 22:00 to 08:00. The gate is a median error under 20%. Bands and speeds stay assumptions until this is done. It tests road speeds, not store sites, so D-030 did not invalidate it: all 32 times below were recomputed on the current road graph and none differ. The store and hex numbers are from the bake before D-030; `check_pairs.json` now holds a fresh draw.

| # | Store to hex | km, straight | Morning / midday / evening / night (min) | Route |
|---|---|---|---|---|
| 1 | 0 to 2580 | 0.70 | 3 / 2 / 3 / 2 | [open](https://www.google.com/maps/dir/?api=1&origin=23.023203,72.562917&destination=23.022046,72.556164&travelmode=two-wheeler) |
| 2 | 2 to 207 | 0.74 | 7 / 6 / 8 / 5 | [open](https://www.google.com/maps/dir/?api=1&origin=23.054703,72.650593&destination=23.048058,72.650779&travelmode=two-wheeler) |
| 3 | 2 to 1715 | 0.79 | 8 / 6 / 9 / 5 | [open](https://www.google.com/maps/dir/?api=1&origin=23.054703,72.650593&destination=23.050674,72.656958&travelmode=two-wheeler) |
| 4 | 4 to 1488 | 0.73 | 5 / 4 / 6 / 3 | [open](https://www.google.com/maps/dir/?api=1&origin=23.010006,72.650687&destination=23.015715,72.654286&travelmode=two-wheeler) |
| 5 | 4 to 1607 | 0.80 | 5 / 4 / 6 / 4 | [open](https://www.google.com/maps/dir/?api=1&origin=23.010006,72.650687&destination=23.002759,72.650309&travelmode=two-wheeler) |
| 6 | 7 to 1963 | 0.89 | 8 / 6 / 9 / 5 | [open](https://www.google.com/maps/dir/?api=1&origin=22.982531,72.572988&destination=22.990438,72.574702&travelmode=two-wheeler) |
| 7 | 10 to 782 | 0.68 | 3 / 2 / 3 / 2 | [open](https://www.google.com/maps/dir/?api=1&origin=22.98695,72.499928&destination=22.989438,72.505935&travelmode=two-wheeler) |
| 8 | 10 to 2471 | 0.85 | 9 / 7 / 10 / 6 | [open](https://www.google.com/maps/dir/?api=1&origin=22.98695,72.499928&destination=22.994243,72.497331&travelmode=two-wheeler) |
| 9 | 11 to 800 | 0.36 | 3 / 3 / 4 / 2 | [open](https://www.google.com/maps/dir/?api=1&origin=22.977963,72.641649&destination=22.975546,72.639274&travelmode=two-wheeler) |
| 10 | 12 to 108 | 0.95 | 8 / 6 / 9 / 5 | [open](https://www.google.com/maps/dir/?api=1&origin=23.077662,72.659371&destination=23.07881,72.650137&travelmode=two-wheeler) |
| 11 | 2 to 1676 | 1.80 | 9 / 8 / 11 / 6 | [open](https://www.google.com/maps/dir/?api=1&origin=23.054703,72.650593&destination=23.039614,72.644157&travelmode=two-wheeler) |
| 12 | 2 to 1729 | 1.27 | 9 / 7 / 10 / 6 | [open](https://www.google.com/maps/dir/?api=1&origin=23.054703,72.650593&destination=23.043539,72.653423&travelmode=two-wheeler) |
| 13 | 3 to 1417 | 1.39 | 6 / 5 / 7 / 4 | [open](https://www.google.com/maps/dir/?api=1&origin=23.041205,72.606212&destination=23.03776,72.593207&travelmode=two-wheeler) |
| 14 | 4 to 1754 | 1.82 | 10 / 8 / 11 / 7 | [open](https://www.google.com/maps/dir/?api=1&origin=23.010006,72.650687&destination=23.009473,72.632897&travelmode=two-wheeler) |
| 15 | 5 to 951 | 1.39 | 10 / 8 / 12 / 7 | [open](https://www.google.com/maps/dir/?api=1&origin=22.969105,72.616446&destination=22.981244,72.612832&travelmode=two-wheeler) |
| 16 | 5 to 1075 | 1.10 | 8 / 7 / 9 / 5 | [open](https://www.google.com/maps/dir/?api=1&origin=22.969105,72.616446&destination=22.962472,72.608424&travelmode=two-wheeler) |
| 17 | 7 to 1965 | 1.27 | 8 / 6 / 9 / 5 | [open](https://www.google.com/maps/dir/?api=1&origin=22.982531,72.572988&destination=22.988541,72.583519&travelmode=two-wheeler) |
| 18 | 8 to 2390 | 1.83 | 11 / 9 / 12 / 7 | [open](https://www.google.com/maps/dir/?api=1&origin=23.059423,72.567449&destination=23.0745,72.560096&travelmode=two-wheeler) |
| 19 | 9 to 7 | 1.81 | 11 / 9 / 13 / 7 | [open](https://www.google.com/maps/dir/?api=1&origin=23.036942,72.680053&destination=23.026886,72.693987&travelmode=two-wheeler) |
| 20 | 12 to 101 | 1.87 | 10 / 8 / 12 / 7 | [open](https://www.google.com/maps/dir/?api=1&origin=23.077662,72.659371&destination=23.077798,72.641088&travelmode=two-wheeler) |
| 21 | 0 to 1337 | 2.65 | 12 / 9 / 13 / 8 | [open](https://www.google.com/maps/dir/?api=1&origin=23.023203,72.562917&destination=23.032228,72.586812&travelmode=two-wheeler) |
| 22 | 1 to 950 | 2.81 | 12 / 10 / 14 / 8 | [open](https://www.google.com/maps/dir/?api=1&origin=23.005089,72.606547&destination=22.979936,72.609747&travelmode=two-wheeler) |
| 23 | 1 to 1024 | 2.82 | 11 / 9 / 12 / 7 | [open](https://www.google.com/maps/dir/?api=1&origin=23.005089,72.606547&destination=22.994913,72.631789&travelmode=two-wheeler) |
| 24 | 3 to 1225 | 2.13 | 14 / 11 / 16 / 9 | [open](https://www.google.com/maps/dir/?api=1&origin=23.041205,72.606212&destination=23.022307,72.609969&travelmode=two-wheeler) |
| 25 | 3 to 1685 | 2.40 | 13 / 11 / 15 / 9 | [open](https://www.google.com/maps/dir/?api=1&origin=23.041205,72.606212&destination=23.035985,72.628937&travelmode=two-wheeler) |
| 26 | 4 to 1286 | 2.57 | 12 / 10 / 14 / 8 | [open](https://www.google.com/maps/dir/?api=1&origin=23.010006,72.650687&destination=23.003945,72.626502&travelmode=two-wheeler) |
| 27 | 8 to 2259 | 2.42 | 10 / 8 / 12 / 7 | [open](https://www.google.com/maps/dir/?api=1&origin=23.059423,72.567449&destination=23.037628,72.566298&travelmode=two-wheeler) |
| 28 | 8 to 2296 | 2.77 | 12 / 10 / 14 / 8 | [open](https://www.google.com/maps/dir/?api=1&origin=23.059423,72.567449&destination=23.041423,72.548649&travelmode=two-wheeler) |
| 29 | 11 to 1581 | 2.83 | 17 / 14 / 19 / 11 | [open](https://www.google.com/maps/dir/?api=1&origin=22.977963,72.641649&destination=22.997945,72.658903&travelmode=two-wheeler) |
| 30 | 13 to 578 | 2.52 | 14 / 11 / 16 / 9 | [open](https://www.google.com/maps/dir/?api=1&origin=23.099669,72.582775&destination=23.102903,72.607102&travelmode=two-wheeler) |

Not part of the 30, but worth a look: these are long detours in our graph, and may mean a missing road link.

| # | Store to hex | km, straight | Morning / midday / evening / night (min) | Route |
|---|---|---|---|---|
| A | 13 to 566 | 2.00 | 42 / 34 / 48 / 27 | [open](https://www.google.com/maps/dir/?api=1&origin=23.099669,72.582775&destination=23.091535,72.600259&travelmode=two-wheeler) |
| B | 7 to 2681 | 1.95 | 25 / 20 / 29 / 16 | [open](https://www.google.com/maps/dir/?api=1&origin=22.982531,72.572988&destination=22.995540,72.560152&travelmode=two-wheeler) |

## Building budget for P-025 (measured, for the owner's decision; no code)

Recommendation: **a 30 m2 minimum area and no confidence floor, with the 6 MB budget applied to what the browser downloads.** That is 835,384 buildings at 5.01 to 5.72 MB, keeping 97.5% of floor area and re-homing 125 of 100,001 residents.

Bytes are real: the meshopt 1.1.1 vertex codec then brotli 11, on quantised instanced boxes in Morton order, from the current bake. The glTF container is not included.

| Filter | Buildings | Floor kept | Walled-city floor lost | Residents to re-home (p99 distance) | Buildings MB, 8 / 12-byte | Browser total MB, 8 / 12-byte |
|---|---|---|---|---|---|---|
| Everything | 1,054,627 | 100.0% | 0.0% | 0 | 4.92 / 5.77 | 6.05 / 6.90 |
| Google confidence 0.70 or more | 908,483 | 96.8% | 6.4% | 1,633 (59 m) | 4.25 / 5.00 | 5.38 / 6.13 |
| Google confidence 0.75 or more | 712,861 | 90.2% | 19.8% | 6,865 (61 m) | 3.38 / 3.97 | 4.51 / 5.10 |
| Drop raster-less Microsoft | 1,045,461 | 99.5% | 0.3% | 443 (165 m) | 4.88 / 5.72 | 6.01 / 6.85 |
| Minimum area 20 m2 | 938,922 | 99.0% | 1.8% | 84 (179 m) | 4.35 / 5.14 | 5.48 / 6.27 |
| **Minimum area 30 m2** | 835,384 | 97.5% | 5.3% | 125 (166 m) | 3.88 / 4.59 | 5.01 / 5.72 |
| Minimum area 50 m2 | 653,300 | 93.1% | 15.5% | 620 (143 m) | 3.07 / 3.63 | 4.20 / 4.76 |
| Drop non-residential areas | 980,645 | 85.6% | 12.2% | 0 | 4.55 / 5.34 | 5.68 / 6.47 |
| 0.75, raster-less, 30 m2 together | 643,661 | 88.9% | 21.5% | 7,327 (89 m) | 3.04 / 3.59 | 4.17 / 4.72 |

Browser total is buildings plus 1.13 MB of compact roads, residents, stores, river and districts. 8-byte instances are x, y at 0.42 m and width, depth, height at 0.5 m, with rotation; they clip 336 widths, 22 depths and 1 height at 127.5 m and have no district. 12-byte instances use 0.1 m and carry the district.

What this shows:

1. **The other files matter as much as the filter.** As specified, the browser files other than buildings are 2.67 MB, leaving room for about 600,000 to 700,000 buildings. Quantising roads to u16 with meshopt takes them from 1.80 to 0.79 MB, and a browser-only residents file (no `hex`, `persona_ref` or `weight`) from 0.85 to 0.32 MB. Those are format changes for Lane A.
2. **Which budget is it?** The stage loads buildings, roads, residents and stores, not `graph.bin` or `travel.bin`, which the engine reads (2.46 MB with `hexes.bin`). If 6 MB must also cover those, only 2.42 MB is left for buildings, about 440,000 to 515,000. Even then, strength as u8 would save only 0.74 MB.
3. **Confidence floors cost the most.** They remove the fine grain of the old city: 0.75 drops 44% of walled-city buildings and 20% of its floor area, against 3.7% in Navrangpura, and moves 6,865 residents (6.9% overall, 13.9% in the walled city). The walled city is already our weakest check (mean 0.817). A 50 m2 minimum beats a 0.75 floor on bytes, floor kept, residents moved (620 rather than 6,865) and walled-city floor lost, but keeps fewer boxes (653,300 against 712,861) and drops more walled-city boxes (51% against 44%).
4. **Keep non-residential buildings on screen.** Dropping them saves about 0.4 MB, re-homes nobody (D-030 already empties them), but removes 14.4% of floor area: the GIDC estates, hospitals and 18.6% of Navrangpura's floor. They still count as workplaces.
5. **Raster-less Microsoft is a wash.** It saves 0.04 MB and moves 443 residents. Keep or drop on data-quality grounds only.
6. **Small-footprint loss is uneven even at 30 m2.** The walled city loses 26% of its buildings but only 5.3% of its floor area. At 50 m2 it loses 51% of its buildings, which may show; judge it visually.

Re-homing can be cheap and need not touch the sim. Filter when the pack is written, not in the population stage, and repoint `building` to the nearest kept building. Residents keep their x, y and z, so the ward checks stay as they are. For the recommended filter 92% land within 100 m and 96% in the same district; p99 is 166 m, max 179 m.

Limits:
- **Frame rate is not solved by any of this.** 835,384 boxes is 8 times the 100k pack Lane B measured at 51 to 57 fps fully in frame on an M4, and even 0.75 leaves 7 times. LOD or culling is needed whichever filter is chosen. I have not measured any frame rate or the 1.5 s load target.
- **Assumes brotli-compressed objects.** With gzip the recommended filter is about 5.38 / 6.08 MB, so 12-byte instances would just miss the budget.
- **Few residents move partly because of how we allocate.** Households are placed by capacity and rounded, so small buildings rarely hold any: 21% of buildings go at 30 m2 but only 0.1% of residents. In reality such buildings may house more people, so read the re-homing counts as a cost to the bake, not proof that those buildings are empty.
- Morton order matters: file order costs 7.63 MB against 5.77 MB for everything at 12 bytes. It means reindexing `building` when writing the pack.
- Extruded footprints are out of budget alone: 9.08 MB for rings, before any index buffer.
- Numbers shift a little if the population or footprints are rebaked.

## P-025 pack layouts (proposal for ARCHITECTURE section 5; no code yet)

For the lead to write into section 5, with Lane B choosing the instance precision. D-031's rule fits in 6 MB, but only just with meshopt. I propose replacing meshopt with plain delta-coded columns, which is smaller and needs no encoder: the first-load download is **5.19 MB** with district and 0.5 m sizes, against 5.94 MB with meshopt and no district. This changes "meshopt-compressed" in P-025 and section 5, so it needs your decision.

### The D-031 rule, measured

A footprint goes to the detail file when its area is under 30 m2 and its polygon lies more than 3.0 m from every other footprint, measured against all 1,054,627, small ones included.

| Rule | First-load buildings | Detail file | Walled floor kept | Walled buildings kept | All floor kept | Residents repointed |
|---|---|---|---|---|---|---|
| Keep everything | 1,054,627 | 0 | 100.00% | 100.0% | 100.00% | 0 |
| Plain 30 m2 | 835,384 | 219,243 | 94.65% | 73.7% | 97.48% | 125 |
| Touching only | 962,902 | 91,725 | 98.64% | 92.7% | 99.01% | 91 |
| Within 1 m | 1,004,048 | 50,579 | 99.68% | 98.1% | 99.47% | 76 |
| **Within 3 m (D-031)** | 1,026,145 | 28,482 | 99.88% | 99.3% | 99.71% | 57 |
| Within 5 m | 1,035,692 | 18,935 | 99.93% | 99.6% | 99.81% | 49 |

- 190,761 of the 219,243 small footprints are within 3 m of another building, so the rule keeps 97.3% of all buildings. It protects the pol houses as intended, but removes little: 1,026,145 boxes, 23% more than a plain 30 m2 for P-048 to hold at 60 fps.
- The 57 residents repointed stand for 4,723 of 8.08M people; p50 47 m, max 179 m.
- The result barely moves between 2 m and 5 m, so "about 3 m" is not a sensitive choice.

### Why columns instead of meshopt

| 1,026,145 buildings, brotli 11 | Meshopt | Columns |
|---|---|---|
| 8-byte precision | 4.78 MB | 4.05 MB |
| 12-byte precision, with district | 5.61 MB | 4.94 MB |
| Roads, 350,594 points | 0.75 MB | 0.71 MB |

- Morton order makes the sorted position code nearly monotonic, so its deltas compress far better than the meshopt vertex filter does here.
- No encoder is needed. The Python `meshoptimizer` package is an alpha third-party wrapper (0.2.30a0, no wheels, no licence metadata), so meshopt would otherwise mean running the JS encoder from citybake.
- Decoding is plain arithmetic in TypeScript and Python. For all 1,026,145 buildings on this machine in Node: 40 ms for brotli and 3 to 9 ms for positions. Not measured in a browser.
- What it gives up: no glTF container, and Lane B decodes into instance attributes itself. If Lane B has a reason to keep meshopt, 8-byte meshopt still fits, with only 60 KB of margin.

### Instance precision, for Lane B to pick

All at 1,026,145 buildings. Browser total adds 1.12 MB for roads, residents, stores, river and districts.

| Attributes after x, y | Buildings MB | Browser total MB | With gzip | Notes |
|---|---|---|---|---|
| x, y; w, d, h at 0.5 m; rotation | 4.05 | **5.17** | 5.66 | clips 336 widths, 22 depths, 1 height at 127.5 m |
| the same plus u8 district | 4.07 | **5.19** | 5.68 | same clipping; district costs 22 KB |
| w, d on a square-root scale; h at 0.5 m | 4.06 | **5.18** | 5.67 | no clipping; 0.37 m at 5 m wide, 1.67 m at 100 m |
| w, d at 0.25 m in u16; h at 0.5 m | 4.30 | **5.42** | 5.90 | no width or depth clipping |
| w, d, h at 0.1 m in u16 | 4.92 | **6.04** | 6.51 | misses, narrowly |

### Exact layouts

Positions are quantised to the manifest `bbox`: `q = round((v - min) / (max - min) * 65535)`, about 0.42 m by 0.37 m on Ahmedabad. The bbox must enclose every coordinate in the pack. A byte-plane column writes all the lowest bytes, then the next, and so on, little-endian. Each column stays padded to 8 bytes.

First-load, the `first_load` size report:

- `buildings.bin` replaces `buildings.glb`, with `counts.buildings` rows in ascending Morton order, which defines the building index:
  - `u32 position_delta` in byte planes: the Morton code of (qx, qy) with x in the even bits, as a delta from the previous row, the first from 0.
  - Then Lane B's chosen attributes. Rotation is `u8`, radians = value / 255 * pi. Boxes repeat every pi.
- `roads.bin`, no longer `f32`:
  - `u16 points[roads]`: points per road; the longest has 114.
  - `u8 class[roads]`: unchanged, 0 street, 1 arterial, 2 bridge.
  - `u16 dx[roadPoints]`, `u16 dy[roadPoints]` in byte planes: zigzag delta of the quantised position, across the whole column, the first from 0.
- `residents.bin`, order is the resident index:
  - `u16 x[n]`, `u16 y[n]`: quantised, not delta-coded.
  - `u8 floor[n]`: z = (floor + 0.5) * 3.2 m. The baked z is mid-floor.
  - `u32 building[n]`: the index in `buildings.bin`, repointed for dropped buildings.
  - `u8 segment[n]`.
- `stores.bin` and `districts.json`: unchanged.
- `river.bin`: see below.

Engine-only, the `engine` size report:

- `residents_engine.bin`: `u32 hex[n]`, `u32 persona_ref[n]`, `f32 weight[n]`, same order. 0.21 MB.
- `graph.bin`, `travel.bin`, `hexes.bin`: unchanged.
- Total 2.67 MB against the soft 10 MB.

Optional, the `detail` size report:

- `buildings_detail.bin`: the same layout as `buildings.bin`, with its own Morton order and `counts.detailBuildings` rows. No resident points to it. 0.12 to 0.14 MB.

`manifest.json`:

- `counts` gains `detailBuildings` and `riverRings`, alongside `hexes` and `bands`.
- `sizes` is new: `{"first_load": {"files": [...], "bytes": n, "budget": 6000000}, "engine": {"files": [...], "bytes": n, "budget": 10000000}, "detail": {"files": [...], "bytes": n}}`. `bytes` is brotli 11, and the manifest itself is not counted.
- Objects are stored brotli-compressed with `Content-Encoding: br`. `files` hashes stay over the uncompressed bytes.

### Holes in water (the owner's request)

- `river.bin` becomes: `u32 offsets[riverRings + 1]` into points, `f32 x[riverPoints]`, `f32 y[riverPoints]`, `u8 kind[riverPolygons]`, `u16 rings[riverPolygons]`.
- Each polygon takes its next `rings[p]` rings: the first is the exterior, counter-clockwise, and the rest are holes, clockwise. `rings[p] >= 1` and they sum to `riverRings`. Toy packs write all 1.
- The stage passes the holes to `ShapeUtils.triangulateShape`, whose indices run over the exterior then the holes. `distanceToRing` must include hole rings too, or the sheen will not fade at an island's shore.
- Ahmedabad has one hole, 15.5 KB for the whole file: Lapkaman Lake (relation/13498663), 0.34 ha, with no buildings or residents. It is not Kankaria. OSM draws Kankaria Lake (way/29340465) as a single outline running around Naginawadi, which overlaps our water by 0.00 ha, so that island is dry with or without this change.
- The case for it is other cities and getting the format right once, not Ahmedabad.

### Costs and limits

- The toy pack changes with all of this, so its `pack_hash` and the golden tapes that carry it regenerate, with no change to simulation output. Worth folding into one regeneration.
- `brotli` (Google, MIT, with wheels) becomes a citybake dependency, for writing objects and the size report. A version bump can change `sizes` and so `pack_hash`, so it stays pinned.
- The budget assumes brotli. With gzip the 0.5 m options still fit, at about 5.7 MB.
- D-031 names buildings, roads, residents and stores; I count river and districts too, since the stage loads the river. Together they are 22 KB.
- Frame rate is still P-048's: this is ten times the 100k pack Lane B measured.
- Quantised positions apply to the engine too if it reads them. That is your call as the reader's owner.
- Found while checking water, not changed: 331 building centroids and 43 residents fall inside water polygons. The largest groups are the river (95 buildings, 24 residents) and one 3.1 ha lake at 72.6413, 22.9873 (75 buildings, 14 residents). Boxes will stand in water there. I can filter them when writing the pack if you want.
