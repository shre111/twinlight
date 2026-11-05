from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from twl_citybake.config import TravelBand

STRATA_KM = ((0.0, 1.0), (1.0, 2.0), (2.0, 3.0))
PER_STRATUM = 10
COORDINATE_DECIMALS = 6
DISTANCE_DECIMALS = 3


@dataclass(frozen=True)
class Places:
    """Coordinates of the table's stores and hexes."""

    store_lat: NDArray[np.float64]
    store_lon: NDArray[np.float64]
    hex_lat: NDArray[np.float64]
    hex_lon: NDArray[np.float64]


def draw_check_pairs(
    distance_m: NDArray[np.float64],
    holds_residents: NDArray[np.bool_],
    minutes: NDArray[np.uint16],
    bands: Sequence[TravelBand],
    places: Places,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    """Store-to-hex pairs for the owner's hand check against Google Maps (D-022): up to PER_STRATUM
    in each distance stratum, hexes holding residents only. Each stratum draws from its own child
    stream, so one stratum's size cannot move another's draws."""
    pairs: list[dict[str, Any]] = []
    for (low, high), stratum_rng in zip(STRATA_KM, rng.spawn(len(STRATA_KM)), strict=True):
        eligible = (distance_m >= low * 1000) & (distance_m < high * 1000) & holds_residents
        flat = np.flatnonzero(eligible)
        drawn = stratum_rng.choice(flat, min(PER_STRATUM, len(flat)), replace=False)
        for s, h in zip(*np.unravel_index(np.sort(drawn), distance_m.shape), strict=True):
            pairs.append(_pair(int(s), int(h), distance_m, minutes, bands, places))
    return pairs


def _pair(
    s: int,
    h: int,
    distance_m: NDArray[np.float64],
    minutes: NDArray[np.uint16],
    bands: Sequence[TravelBand],
    places: Places,
) -> dict[str, Any]:
    store_lat = round(float(places.store_lat[s]), COORDINATE_DECIMALS)
    store_lon = round(float(places.store_lon[s]), COORDINATE_DECIMALS)
    hex_lat = round(float(places.hex_lat[h]), COORDINATE_DECIMALS)
    hex_lon = round(float(places.hex_lon[h]), COORDINATE_DECIMALS)
    return {
        "store": s,
        "hex": h,
        "store_lat": store_lat,
        "store_lon": store_lon,
        "hex_lat": hex_lat,
        "hex_lon": hex_lon,
        "distance_km": round(float(distance_m[s, h]) / 1000, DISTANCE_DECIMALS),
        "minutes": {band.name: int(minutes[s, h, b]) for b, band in enumerate(bands)},
        "maps_url": (
            "https://www.google.com/maps/dir/?api=1"
            f"&origin={store_lat},{store_lon}&destination={hex_lat},{hex_lon}"
            "&travelmode=two-wheeler"
        ),
    }
