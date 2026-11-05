# Lane C (minds)

Worktree `.claude/worktrees/minds`. Lead folds this into MEMORY.md at each wrap.

## P-023 social graph (ready to tick)

`citybake graph ahmedabad` on the real city, no parameter tuned. All approved targets hold. 100,002 residents, 297,531 undirected ties (household 132,032, neighbour 79,057, friend 59,937, work 26,505).

| Measure | Result | Target |
|---|---|---|
| Mean degree | 5.95 | 5 to 8 |
| Average local clustering, all ties | 0.2175 | 0.10 to 0.50 |
| Cross-river share, non-household ties | 0.139 | 0.05 to 0.25 |
| Largest component | 0.9992 | at least 0.95 |
| Isolated residents | 0.0004 | at most 0.02 |

- Degree: p10 3, p50 6, p90 9, p99 12, max 18. By kind: household 2.64, neighbour 1.58, friend 1.20, work 0.53.
- Friend homophily: 0.521 same segment against 0.231 under random mixing.
- Strength means: household 0.849, friend 0.601, neighbour 0.390, work 0.350. Per resident mean 0.646 (p10 0.539, p90 0.759).
- Workplaces: 6,803 buildings (OSM tagged 2,656, untagged footprints 4,147). Workers: east 30,277, west 22,984.
- 1.3 s, 5.8 MB, byte-identical on rerun on this machine. `layer.json` records the digests of residents, assignment and buildings, and which buildings stage was used.

Questions and limits for the lead:

- **Clustering target measures households, not social closure.** Without household ties clustering is 0.005 (transitivity 0.0045). Friends of friends rarely know each other, so word of mouth may spread faster than in a real city once it leaves a household. The 0.10 to 0.50 target passes because of household cliques. Triadic closure would be a model change, so I have not added it. Worth checking against the first P-012 adoption curves.
- **Influence is mostly household.** Share of summed w^2: household 71.6%, friend 16.7%, neighbour 9.2%, work 2.6%; share of summed w: 59.6%, 19.1%, 16.4%, 4.9%. Both before `talk_kind`.
- **No single talk reaches the awareness threshold at talk_gain 0.9**: spouse 0.90, housemate 0.72, friend mean 0.54, neighbour mean 0.35, work mean 0.31. Relevant to the possible strength or gain normalisation.
- **The east's workplaces rest on the untagged rule.** Untagged footprints are 90.3% of the east's workplace floor area (36.4% in the west). The east's worker share depends on `workplace_min_area_m2`, which nothing anchors: 2,000 m2 gives 46.7%, 1,000 m2 gives 56.8%, 500 m2 gives 62.4%.
- **3,872 residents (3.87%) live in footprints that also count as workplaces.** P-021 treats unknown footprints as 90% housing. A small double use, recorded rather than changed.
- Work ties cross the river at 0.379 against 0.442 for random mixing, a mild preference for the home bank from the distance decay.
- Byte-identity was checked on one machine only.

`docs/MEMORY.md` on main cites D-027, D-028 and MODEL.md section 11, but DECISIONS.md there stops at D-026 and MODEL.md ends at section 10. Possibly not pushed yet.

## P-024 stores and travel times (ready to tick; results provisional on the D-025 question)

`citybake stores ahmedabad` on the real city. Both acceptance criteria hold: 92.3% of residents within 2.5 km, and the travel table is complete.

- 15 stores: west 5, east 10. 2,982 hexes, median 22 residents.
- Table: 357,840 bytes raw, 48,603 zstd, 5.8% of pairs inside a radius.
- Road graph: 99.6% of 88,744 nodes in the main strong component. Hex to road node: p50 39 m, p90 110 m, max 881 m.
- About 2 s, byte-identical on rerun on this machine. `layer.json` records every input digest and the store and travel config.
- Greedy matches a brute-force run on the real candidates, and a separate heap Dijkstra matched the table for store 0 in all bands (persona-lab). 38 deliberate bugs across the six modules are all caught by the tests.

| Band | p50 | p90 | Max | Within radius p50 | Within radius p90 |
|---|---|---|---|---|---|
| morning_peak | 34 | 55 | 90 | 9 | 13 |
| midday | 27 | 44 | 72 | 8 | 11 |
| evening_peak | 39 | 63 | 103 | 11 | 15 |
| night | 22 | 35 | 58 | 6 | 9 |

### Questions for the lead

1. **Population step (D-025), sent separately.** GHS-POP gives Bodakdev 7,069 and Navrangpura 83,819 people per km2 on almost the same building stock (158 vs 15 m2 of floor area per person), with a sharp north-south edge between them. P-021 reproduces the raster faithfully, so the step comes from the raster. Not confirmed from outside: the AMC ward-wise census page fails TLS verification and I did not bypass it. Next step: compare the AMC 2011 ward table with baked counts.
2. **Western corridor at 0% coverage.** In Bodakdev, Thaltej, Vastrapur, Memnagar, Sola, Gota, South Bopal, Chanakyapuri and Santej, no resident has a store within 2.5 km. I kept `coverage_target` at 0.92 for now, because a corrected population should move stores there by itself. If the raster stays, I recommend 0.95: 18 stores, 95.5% coverage.
3. **Keep the full travel table for P-025.** At 48,603 bytes zstd it is already small, so an in-radius table saves little. Your call.
4. **Radius definition for P-025 and the engine.** Coverage is by resident point; `within_radius_*` is by hex centre. 0.80% of weight is covered by point but its hex centre is outside every radius, and 0.58% the reverse (persona-lab). Pick one.
5. **Travel crowds the promise.** MODEL's 10 to 20 minutes covers queue, picking, rider wait and travel. Evening travel alone is p50 11 and p90 15, so a 10-minute promise fails for much of the city in the evening before any queueing.
6. **Load per store is uneven.** By nearest store, represented shoppers per store are min 31,007, median 87,447, max 214,517. Placement ignores capacity, so stock and rider pools must absorb it.

Smaller limits: minutes stop at the road node, with no snap distance added. Rounding up adds about 0.5 minute on average, 6 to 7% of a median pair. 4 of 15 stores stand on dead-end road nodes, and greedy can place a store for ring coverage with few people nearby. `travel.complete` can only be false for an empty table, because a missing pair raises. Byte-identity is checked on one machine only.

### Hand check for the owner (D-022)

Open each link, read Google's two-wheeler time, and compare with the band matching the time you look it up: morning 08:00 to 11:00, midday 11:00 to 17:00, evening 17:00 to 22:00, night 22:00 to 08:00. The gate is a median error under 20%. Bands and speeds stay assumptions until this is done. It tests road speeds, not store sites, so it stays valid if the population changes. Pairs are store to hex in `check_pairs.json`, 10 in each distance stratum.

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
