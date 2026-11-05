# Lane C (minds)

Worktree `.claude/worktrees/minds`. Lead folds this into MEMORY.md at each wrap.

## Now
- D-024 merged (#55). D-025 population from GHS-POP: this PR. Next: P-022, then footprint cache keys.

## D-025 real run (Ahmedabad, combined corrected buildings)
- 8,080,000 people in the bbox, 7,959,302 synthesised, 120,700 in cells without buildings.
- 104 districts at 0.00% error, 0 over capacity, IPF at most 8 sweeps.
- 1,678,715 households, mean size 4.74.
- 100,002 residents from 30,595 households, weight 14.83, about 1.48M adults represented.
- `population_total` checked by hand on real boundaries:
  - Maninagar taluka (relation 16736464) rescales to the total, with cells outside bit-identical;
  - Ahmadabad district (relation 1953566) is refused because it reaches past the bbox;
  - an unknown id is refused.
- `tools/citybake/README.md` holds the new-city population check and the floor-area fallback.
- Still crediting WorldPop, outside this lane's paths: README.md lines 17 and 25, `.claude/agents/persona-lab.md` line 18, `design/screens/source/*.html`, and MEMORY's history lines. Over to Lane A.
- I deleted `data/raw/worldpop` from this worktree (about 778 MB, a local cache, re-downloadable over FTP).

## D-024 results (Ahmedabad, 18,557 tagged buildings, whole-district 5-fold hold-out)
| Storeys | n | Abs error, raw to corrected | Signed after | Aggregate after |
|---|---|---|---|---|
| 1-2 | 9,519 | 0.207 to 0.170 | +0.167 | +0.259 |
| 3-4 | 3,207 | 0.243 to 0.222 | -0.097 | -0.024 |
| 5-7 | 3,717 | 0.289 to 0.142 | -0.032 | -0.045 |
| 8-12 | 1,674 | 0.312 to 0.149 | -0.070 | -0.099 |
| 13+ | 440 | 0.354 to 0.192 | -0.191 | -0.265 |
| All | 18,557 | 0.247 to 0.172 | +0.069 | +0.001 |

- By raster height bin, the aggregate error is within 1.3% in every bin. The median signed error is within 0.067 except in the 6 to 9 m bin at +0.177: most of those buildings are 6.4 m with a tall tail, so the mean sits above the median building. `aggregate_after` is in the report because the aggregate is what a mean-calibrated curve guarantees.
- 38 knots from 1.5 to 52.3 m. The largest single raster height holds 1.4% of raster buildings (70.6% with the median). 81% of all buildings have 2 floors. Floor area is 344.5 km2.
- The top ratio is 0.902, so the floor at 1 binds and makes one upward jump at 52.3 m: 104 raster buildings above it keep their raw height (max 87.3 m), and 80 with raw 46 to 52 m stay shrunk by up to 5 m. The map is still monotone. I left this as the decision reads.
- Capacity on corrected heights is 75.3M, a ceiling rather than a count.

## Open follow-ups
- Footprint cache keys: `vida.parquet` and `heights.tif` should record what they were built from (reviewer should-fix on #44, #45).
