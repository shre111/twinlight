# citybake

Bakes a city pack from open data. Each stage writes under `data/cache/citybake/<city>/<stage>` and is byte-identical on rerun. `TWL_DATA_DIR` overrides `data`.

```
citybake fetch <city>        pinned OSM extract (clipped to the bbox) and population raster
citybake osm <city>          buildings, roads, water, banks, districts
citybake footprints <city>   VIDA footprints merged in, 2.5D heights corrected (D-021, D-024)
citybake population <city>   households per building, residents (D-025)
citybake personas <city>     Nemotron persona uuids and segments per resident (D-026)
citybake graph <city>        social graph: household, neighbour, friend and work ties
citybake stores <city>       synthetic dark stores (D-013), travel minutes per store, hex and band (D-022)
```

City settings live in `src/twl_citybake/cities/<city>.toml`.

## Adding a city: population check
GHS-POP (D-025) comes first, but it has to be checked for each city. Before baking:

1. Sum the raster over the bbox and compare it with a sourced city total.
2. Compare the core-to-suburb density ratio against census ward counts, where ward boundaries exist. A core far flatter than the wards means the raster under-concentrates. WorldPop R2025A did this in Ahmedabad: 12k/km2 in the walled city, against 55k/km2 in GHS-POP and about 50k/km2 known.

If the total is off but the shape is right, set `[population_total]` (`people`, `source`, `polygon`, `as_of`). It rescales the cells inside that OSM boundary, not the bbox, so a census figure can replace a projection. The boundary must lie inside the raster window (the bbox out to whole cells), because a total cannot be spread over part of its polygon.

If the shape is also wrong, the documented fallback (D-025, not built) is to distribute the sourced total by residential floor area: footprints x D-024 floors, excluding commercial and industrial buildings, instead of rescaling the raster.
