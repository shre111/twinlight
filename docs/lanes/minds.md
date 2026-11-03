# Lane C (minds)

Worktree `.claude/worktrees/minds`. Lead folds this into MEMORY.md at each wrap.

## Now
- P-021 population synthesis: rng, capacity and IPF (#33) and the WorldPop window plus allocation (#35) are merged. This PR adds households, residents and `citybake population <city>`. Still to come: Ahmedabad's `[worldpop]`, `[population]` and `[households]` TOML sections, which wait on the raster md5 (the download is still running, see below), then the real run.
- Next: the D-021 footprints stage, then P-021 rerun on the merged buildings with synthetic boxes last, then P-022.

## P-021 numbers (synthetic check)
Real P-020 buildings with a synthetic raster of 7.5M people spread over the cells that have buildings:
- IPF converges in at most 7 sweeps everywhere.
- Household sizes match the census to 3 decimals (mean 4.742).
- 100,001 residents from about 30,800 households, weight about 13.8.
- 0 buildings over capacity.
- Districts land on their placed people exactly, because the singles column absorbs rounding.
- 10 districts are capacity-bound on OSM buildings alone (Isanpur, Kalapi nagar, DaniLimda, Lambha and others), so they miss their WorldPop marginal. D-021 footprints fix this.

## Assumptions to confirm (Lane A)
- `shopper_share = 0.27`: the share of adults the agents stand for. It reproduces DESIGN's "100,000 residents, each about 16 people" but has no data behind it yet.
- Adults per household: households of one or two are all adults. Larger ones have two adults plus r x (size - 2), with r = 0.47 so the census reaches 69% adults (Census 2011 C-14, Ahmadabad district urban). Fractions are rounded with the seeded stream, which keeps couples for P-023's household ties.
- Census vintage: household sizes are from 2011 (mean 4.7) but applied to 2025 WorldPop. Households have likely shrunk since.
- Edge cells: WorldPop cells that straddle the bbox are read whole, and all their people go to the buildings inside the bbox.

## D-021 and D-022 feasibility (checked 2026-11-03)
- VIDA: Ahmedabad is one S2 level-6 file (`country_iso=IND/4134022982949404672.parquet`, 702 MB, ETag `d4fb270fd5ebbc6db795b886a3a213f6-84`). DuckDB with a `bbox` filter reads the city in about 40 s. It holds 1,056,667 Google footprints (all at confidence 0.65 or more; 899k at 0.7 or more, 688k at 0.75 or more) and 18,338 Microsoft ones. VIDA's own OSM rows are only 2,105, so the pinned OSM extract is the OSM tier.
- Google 2.5D Temporal 2023: public cloud-optimised GeoTIFFs in `gs://open-buildings-temporal-data/v1`. The manifest `39_EPSG_32643_2023_06_30.json` lists 15 tiles over the bbox, each 25,000 px at 0.5 m with overviews. The height band at 4 m reads over HTTP in about 1.6 s per tile.
- Licences: VIDA ODbL 1.0; Google Open Buildings and 2.5D Temporal CC BY 4.0 or ODbL 1.0 (user's choice, and the 2.5D data asks for a citation of Sirko et al. 2023); Microsoft ODbL 1.0.

## Gotchas
- The only randomness in citybake is `twl_citybake.rng.stream(seed, Stream.X)`. Households use no RNG; residents draw from `Stream.POPULATION` in a fixed order.
- WorldPop's HTTPS server ignores range requests and drops long transfers. `ftp://ftp.worldpop.org/...` resumes with `curl -C -`.
- Don't run `citybake fetch ahmedabad` while a manual WorldPop download writes to the same `.tif.part` path.
- Pyright with the NumPy 2.5 stubs needs integer index arrays built with an explicit `np.intp` dtype.
