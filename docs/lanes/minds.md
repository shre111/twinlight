# Lane C (minds)

Worktree `.claude/worktrees/minds`. Lead folds this into MEMORY.md at each wrap.

## Now
- P-020 OSM bake for Ahmedabad. Done: pinned Geofabrik `western-zone-260101` extract (md5 checked), clipped to the bbox 72.45,22.92,72.72,23.14 with pyosmium (6.7 MB, 61 s), local transverse Mercator centred on the bbox (#16). In progress: buildings with heights, road graph, river and lakes, Sabarmati banks, districts, bake report.
- Next: P-021 population synthesis (WorldPop R2025A 2025, 100 m, constrained), then P-022 to P-024.

## For Lane A (decisions needed)
1. **OSM building coverage gap (blocks P-021 AC).** The bbox has 100,504 OSM buildings, but the walled city (Khadia, Jamalpur, Dariyapur) and the south (Vatva, Narol, Isanpur, Lambha) have almost none, while WorldPop puts dense population there. "No building over capacity" cannot hold there. Options: (a) fill gaps from VIDA's combined Google v3 + Microsoft + OSM footprints (source.coop `vida/google-microsoft-osm-open-buildings`, GeoParquet by country and S2 cell, ODbL like OSM), adding only footprints that don't overlap OSM ones. Heights still come from district defaults. This is a new source, so it would amend D-013. (b) synthesise box buildings in populated cells without OSM footprints, flagged `synthetic` in the pack. Lane C proposes (a), with (b) as the fallback.
2. **Districts.** OSM has no ward boundaries for Ahmedabad, only talukas (admin_level 6). Lane C builds about 100 districts as Voronoi cells of named OSM place nodes (suburb, neighbourhood, quarter, village, town), split along the Sabarmati so no district spans both banks. The 48 AMC wards exist as open data (DataMeet `Municipal_Spatial_Data/Ahmedabad/Wards.geojson`). They have the official names but are CC BY-SA 2.5 India (share-alike) and would be a new source. Proposal: stay OSM-only per D-013. The district source is a city-config switch, so wards can be added later by a decision.
3. **Heights.** Only 16 buildings have `height` and 18.7% have `building:levels`. Defaults are the median levels x 3.2 m (D-013), taken over the district and building kind, falling back to the city-wide kind, then the city.
4. **P-024 travel times without OSRM.** OSRM needs a container, and Docker is off-limits until a laptop restart. Proposal: compute store-to-hex travel times on the baked road graph (scipy Dijkstra, per-class speeds per time-of-day band), with OSRM as an optional backend later. This changes the plan's wording, so it needs a decision.
5. **Pack byte layouts (P-025).** ARCHITECTURE section 5 gives no offsets for `roads.bin` or `river.bin`. Proposal: `roads.bin` = u32 n, u32 offsets[n+1], f32 xy[2m], u8 class[n], with class = stage `ROAD_KIND` (0 street, 1 arterial, 2 bridge). `river.bin` = u32 p, u32 poly_offsets[p+1] (into rings), u32 ring_offsets[r+1] (into vertices), f32 xy[2k], u8 kind[p] (0 river, 1 lake), with the exterior ring first.

## Data notes
- Persona pool (P-022): Nemotron-Personas-India `en_IN` has 1,000,000 personas: 53,077 in Gujarat and 6,604 in Ahmadabad district. All are aged 18 or over. There are 1,216 distinct occupation titles, and 47% are "No Occupation / Retired / Homemaker". DuckDB reads the demographic columns over HTTP (about 12 s), with no need to download the 5 GB of text.
- WorldPop R2025A India is 778 MB and its server ignores HTTP range requests, so citybake downloads it whole.

## Gotchas
- Bake data lives in `data/raw` (downloads) and `data/cache/citybake/<city>/<stage>` (gitignored). `TWL_DATA_DIR` overrides the root.
- `uv add --package <member>` syncs only that member and uninstalls the others' deps; run `uv sync` afterwards.
