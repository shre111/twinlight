# Lane C (minds)

Worktree `.claude/worktrees/minds`. Lead folds this into MEMORY.md at each wrap.

## Now
- P-020 OSM bake for Ahmedabad: code complete with this PR (#16, #18, #22, then this one). `citybake fetch ahmedabad` downloads and clips the pinned extract; `citybake osm ahmedabad` writes layers and `report.json` to `data/cache/citybake/ahmedabad/osm/layers` in about 4 s, byte-identical across runs. Pack rendering waits for P-025 (the pack writer) and P-016 (the city layer).
- Follow-up in review: default heights also conditioned on footprint band (below 250, 250 to 700, above 700 m2), and coincident district seeds collapsed.
- Next: P-021 on cells that have OSM buildings (the lead's instruction until the footprint decision lands). The WorldPop download is slow, about 100 KB/s.

## P-020 bake report (Ahmedabad, bbox 72.45,22.92,72.72,23.14, Geofabrik western-zone-260101)
- Buildings: 100,488. By kind: unknown (`building=yes`) 85,531, residential 11,985, commercial 1,832, civic 509, industrial 315, other 273, religious 43.
- Height sources: `height` tag 14, `building:levels` 18,776, default 81,698 (81%).
- Default levels with footprint bands: 1 level 96 buildings, 2 levels 67,305, 3 levels 11,026, 4 levels 2,113, 5 to 10 levels 1,158. Before bands they were nearly all 2.
- Height distribution: p5 to p75 are all 6.4 m (2 levels), p95 16.0 m, max 137.6 m (43 levels).
- Footprint: p5 44 m2, p50 133 m2, p95 625 m2.
- Roads: 88,744 nodes and 112,373 edges in 21 weak components, the largest holding 99.8% of nodes. Length: local 4,865 km, tertiary 1,556 km, service 831 km, secondary 431 km, primary 392 km, major 254 km. 454 bridge edges.
- Water: the Sabarmati covers 14.9 km2 in 3 polygons; 60 lakes total 3.0 km2.
- Banks: the Sabarmati splits the bbox; the west bank holds 64.3% of buildings.
- Districts: 105 (57 seeds west, 48 east). Buildings per district: min 0, median 482, max 6,936 (Thakkarabapa Nagar).
- Coverage: 250 of 672 one-km cells (37%) have no buildings. These are the walled city and the south.

## Lead answers (2026-11-03)
- Approved within D-013: districts from OSM place-node Voronoi cells split along the Sabarmati, with the source kept a city-config switch; and the height defaults (district and kind, then kind, then city).
- Pack layouts: use ARCHITECTURE section 5 "Binary layout (v1)" from P-010. `roads.bin` and `river.bin` have separate x and y columns, counts in `manifest.json`, and one exterior ring per polygon. The water layer here keeps holes. In Ahmedabad the river polygons have none and one lake has one (likely Kankaria's island), so v1 dropping holes costs one island.
- With the owner: VIDA footprints for the coverage gap, and P-024 travel times on the baked graph versus OSRM. Docker works again. P-024 goes behind a backend interface.

## Data notes
- Persona pool (P-022): Nemotron-Personas-India `en_IN` has 1,000,000 personas: 53,077 in Gujarat and 6,604 in Ahmadabad district. All are aged 18 or over. There are 1,216 distinct occupation titles, and 47% are "No Occupation / Retired / Homemaker". DuckDB reads the demographic columns over HTTP in about 12 s.
- Household sizes (P-021): Census 2011 HH-01 (Gujarat, `DDW-HH01-2400-2011.XLS`). Ahmadabad City sub-district: 1,175,757 households, 5,552,466 persons, mean 4.7. Bins: 1, 2, 3, 4, 5, 6, 7 to 10, 11 to 14, 15+.
- WorldPop R2025A India 2025 (100 m, constrained) is 778 MB, and its server ignores HTTP range requests, so citybake downloads it whole.
- Levels rise with footprint size for residential and commercial buildings: about 2 below 150 m2, 5 to 6 at 250 to 700 m2, 8 to 10 above 700 m2. Unknown buildings stay near 2 to 3. This is why defaults use footprint bands.

## Gotchas
- Bake data lives in `data/raw` (downloads) and `data/cache/citybake/<city>/<stage>` (gitignored). `TWL_DATA_DIR` overrides the root.
- `uv add --package <member>` syncs only that member and uninstalls the others' deps; run `uv sync` afterwards.
- pyosmium hands self-intersecting closed ways over as empty areas, and its WKB factory raises `RuntimeError` on them; they count as `skipped_invalid`.
- `shapely.voronoi_polygons(ordered=True)` needs GEOS 3.12 or later. The shapely 2.2 wheels include it.
- Tests use `--import-mode=importlib`, so `tests/conftest.py` puts the tests directory on `sys.path` for the shared `osm_fixture` builder.
