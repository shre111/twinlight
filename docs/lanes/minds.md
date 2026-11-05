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
