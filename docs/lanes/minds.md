# Lane C (minds)

Worktree `.claude/worktrees/minds`. Lead folds this into MEMORY.md at each wrap.

## Now
- P-020: all ACs hold. Lane B rendered a v1 preview pack of the OSM layers in `/dev/stage` (2026-11-04): 100,488 buildings, 238k road segments and 100k stand-in residents in 8 draw calls, 16.7 ms p50 and 17.5 ms p95 on the M4. Stills are in Lane B's scratchpad. The pack was written by `twl_engine.fixtures.pack.write_pack` from a one-off script; P-025 brings the real writer.
- D-021 footprints: this PR completes `citybake footprints <city>`.
- Next:
  - D-024 height correction.
  - Ahmedabad's population TOML sections. The WorldPop raster is downloaded: md5 `89905eed942228829cf21eee28eb12df`, 778,106,191 bytes.
  - The lead's four P-021 follow-ups.
  - The real P-021 run.
  - District seed merges and names.
  - P-022.

## D-021 bake (Ahmedabad, cached rerun 23 s, peak RSS about 3.2 GB, byte-identical reruns)
- Buildings: 1,054,622 in total.
  - OSM: 100,488.
  - Google: 936,446 kept of 1,056,667. 120,069 were dropped by dedupe and 152 fell outside the bbox.
  - Microsoft: 17,688 kept of 18,338. 645 were dropped and 5 fell outside.
- Heights by source: 14 from `height` tags, 18,776 from `building:levels`, 919,020 from the raster and 116,812 defaults. The raw raster median is kept in `raster_height_m` (NaN means no raster support).
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
