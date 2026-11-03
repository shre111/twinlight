# Lane C (minds)

Worktree `.claude/worktrees/minds`. Lead folds this into MEMORY.md at each wrap.

## Now
- P-020: all ACs hold. Lane B rendered a v1 preview pack of the OSM layers in `/dev/stage` (2026-11-04): 100,488 buildings, 238k road segments and 100k stand-in residents in 8 draw calls, 16.7 ms p50 and 17.5 ms p95 on the M4. Stills are in Lane B's scratchpad. The pack was written by `twl_engine.fixtures.pack.write_pack` from a one-off script; P-025 brings the real writer.
- D-021 footprints: merged (#44, #45).
- P-021: the ACs hold on the real Ahmedabad run (#46, see below).
- This PR adds the district names table (approved):
  - display names prefer `name:en`, with the OSM `name` kept as `name_local`;
  - `[districts]` gains `extra_seeds`, `merge` and `rename`, and any key that matches nothing fails loudly;
  - a merged seed's catchment joins its target.
  - Ahmedabad now has 104 districts. Vatva is an extra seed on its railway station node (37.8k buildings). Koteshwar Village is merged into Koteshwar. The four Gujarati-script names now display in English.
  - "Jantanagar -Ramol" and "Ramol -Jantanagar" stay separate (renamed Jantanagar and Ramol): their nodes are 2.85 km apart, and merging them made a two-part Ramol.
- Decided, to build next: the D-024 refinement (bin mean, top bins pooled to at least 30 buildings, top ratio of at least 1, floors = round, median signed error by raster bin and by storey band). The code is parked on `feat/p020-height-correction`.
- Waiting on the owner: the population raster (GHS-POP recommended, see below) and the P-022 segment set (6 segments proposed).
- Next: footprint cache keys (`vida.parquet` and `heights.tif` should record what they were built from), then P-022.

## GHS-POP check (for the WorldPop scale decision)
Ahmedabad: WorldPop R2025A against GHS-POP R2023A, 2025 epoch, 3 arcsec, CC BY 4.0, one 10-degree tile R7_C26:

| Area | WorldPop | GHS-POP |
|---|---|---|
| bbox | 3.74M | 8.08M |
| walled city density | 12.2k/km2 | 55.5k/km2 |
| Vatva to Isanpur density | 9.9k/km2 | 34.3k/km2 |

The two grids correlate at 0.69 cell by cell. P-021 on GHS-POP, unchanged:
- 7.96M people placed, 105 districts at 0.00% error, 0 over capacity.
- 120.7k people unplaced (WorldPop: 234k).
- Resident weight 14.8.

## P-021 real run (Ahmedabad, combined buildings, WorldPop R2025A, about 4 s)
- All 105 districts are synthesised at 0.00% error against their marginals. 0 buildings over capacity; IPF converged everywhere in at most 12 sweeps.
- 740,171 households, mean size 4.74 (census 4.74).
- 100,001 residents from 30,465 households; weight 6.54.
- 234k WorldPop people sit in 20,558 cells with no buildings, so they are unplaced.
- WorldPop scale: R2025A puts only 3.74M people in the bbox (AMC alone was 5.58M in 2011; the UA is about 8 to 9M in 2025). Its densest cell holds 130 people per 100 m, so it under-concentrates dense cores; Mumbai's island bbox shows 4.99M against over 12M. Options sent to the lead:
  - rescale to a sourced city total;
  - switch to GHS-POP R2023A;
  - both.
- Lead-accepted assumptions, now explicit in the config and report:
  - `shopper_share` 0.27, with the mean weight in `layer.json`;
  - the adult share 0.69 with its C-14 source;
  - `[households].size_shrink` (default 1.0; an exponential tilt of the census sizes);
  - edge-cell people beyond the bbox in the report.

## D-021 bake (Ahmedabad, cached rerun 23 s, peak RSS about 3.2 GB, byte-identical reruns)
- Buildings: 1,054,627 in total.
  - OSM: 100,488.
  - Google: 936,451 kept of 1,056,667. 120,064 were dropped by dedupe (overlap counted once where higher footprints overlap each other) and 152 fell outside the bbox.
  - Microsoft: 17,688 kept of 18,338. 645 were dropped and 5 fell outside.
- Heights by source: 14 from `height` tags, 18,776 from `building:levels`, 919,024 from the raster and 116,813 defaults. The raw raster median is kept in `raster_height_m` (NaN means no raster support).
- Raster against OSM levels x 3.2 m on 18,557 tagged buildings: median ratio 0.817 and median absolute error 2.17 m. The ratio falls with height: 0.96 at 1 to 2 storeys, 0.77 at 3 to 4, 0.71 at 5 to 7, 0.69 at 8 to 12 and 0.65 at 13 or more. D-024 corrects this.
- Coverage: 1 km cells without buildings drop from 250 of 672 to 0. Jamalpur, Khadia, Dariyapur, Isanpur, Lambha and Narol are now over 98% Google footprints.
- Synthetic P-021 check: capacity-bound districts drop from 10 to 0 with the combined buildings. Zero-capacity cells with people drop from 82 to 12, holding about 6k people in temples and sheds.
- Kept but flagged for P-025's filter:
  - Google footprints with confidence below 0.7: 15.6% of those kept.
  - Microsoft footprints without raster support: 9,166.
- Attribution: Microsoft footprints are credited "ODbL 1.0, via VIDA's 2024-08-27 snapshot" (Microsoft's repo has been CDLA Permissive 2.0 since 2026-03). `layer.json` records VIDA's URL, ETag and snapshot date, plus the 2.5D tile list.

## Gotchas
- DuckDB installs its httpfs extension on the first fetch. GeoParquet is read as raw WKB with `enable_geoparquet_conversion = false`.
- The footprints stage re-runs the OSM place and river pass (`osm.extract.district_seeds`) so new footprints use the same district seeds and banks. It raises an error if the PBF's district list differs from `osm/layers/districts.json`.
- The population stage reads `buildings/layers` when it exists, otherwise `osm/layers`.
