from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import osmium
import shapely
from osmium.geom import WKBFactory
from osmium.osm import Area

from twl_citybake.population.raster import repaired_polygon


@dataclass(frozen=True)
class NonResidential:
    """Buildings standing in areas OSM marks as not housing, how many areas of each tag, and how
    many matching areas could not be assembled."""

    mask: npt.NDArray[np.bool_]
    areas: dict[str, int]
    skipped: int


def nonresidential_buildings(
    pbf: Path,
    lon: npt.ArrayLike,
    lat: npt.ArrayLike,
    landuse: Sequence[str],
    amenity: Sequence[str],
) -> NonResidential:
    """Which building centroids lie inside an area tagged with one of these values."""
    factory = WKBFactory()
    geometries: list[shapely.Geometry] = []
    counts: Counter[str] = Counter()
    skipped = 0
    for obj in osmium.FileProcessor(str(pbf)).with_areas():
        if not isinstance(obj, Area):
            continue
        tag = _matching_tag(obj.tags.get("landuse"), obj.tags.get("amenity"), landuse, amenity)
        if tag is None:
            continue
        try:
            geometries.append(repaired_polygon(shapely.from_wkb(factory.create_multipolygon(obj))))
        except RuntimeError:
            skipped += 1
            continue
        counts[tag] += 1
    lons, lats = np.asarray(lon, np.float64), np.asarray(lat, np.float64)
    mask = np.zeros(len(lons), np.bool_)
    if geometries:
        inside, _ = shapely.STRtree(geometries).query(
            shapely.points(lons, lats), predicate="within"
        )
        mask[inside] = True
    return NonResidential(mask, dict(sorted(counts.items())), skipped)


def _matching_tag(
    landuse_value: str | None,
    amenity_value: str | None,
    landuse: Sequence[str],
    amenity: Sequence[str],
) -> str | None:
    if landuse_value in landuse:
        return f"landuse={landuse_value}"
    if amenity_value in amenity:
        return f"amenity={amenity_value}"
    return None
