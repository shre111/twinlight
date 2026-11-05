from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import zstandard
from numpy.typing import NDArray

from twl_citybake.config import StoresConfig, TravelBand
from twl_citybake.osm.banks import EAST, WEST
from twl_citybake.stores.placement import Placement

REPORT_DECIMALS = 3
TABLE_ZSTD_LEVEL = 19


def stores_section(
    placement: Placement,
    config: StoresConfig,
    district: NDArray[np.integer],
    bank: NDArray[np.integer],
    district_names: Sequence[str],
) -> dict[str, Any]:
    by_district = np.bincount(district, minlength=len(district_names))
    by_bank = np.bincount(bank, minlength=2)
    return {
        "count": len(placement.node),
        "radius_m": config.radius_m,
        "coverage": coverage(placement),
        "coverage_by_store": placement.covered_share.tolist(),
        "by_district": {name: int(by_district[i]) for i, name in enumerate(district_names)},
        "by_bank": {"west": int(by_bank[WEST]), "east": int(by_bank[EAST])},
    }


def coverage(placement: Placement) -> float:
    return float(placement.covered_share[-1]) if len(placement.covered_share) else 0.0


def hexes_section(
    residents_per_hex: NDArray[np.integer], node_distance_m: NDArray[np.floating]
) -> dict[str, Any]:
    """`residents_per_hex` covers every hex in the table; hexes added only because a store stands
    in them hold nobody and stay out of the median."""
    return {
        "count": len(residents_per_hex),
        "residents_per_hex_p50": _percentile(residents_per_hex[residents_per_hex > 0], 50),
        "node_distance_m": {
            "p50": _percentile(node_distance_m, 50),
            "p90": _percentile(node_distance_m, 90),
            "max": _percentile(node_distance_m, 100),
        },
    }


def travel_section(
    table: NDArray[np.uint16],
    bands: Sequence[TravelBand],
    in_radius: NDArray[np.bool_],
    holds_residents: NDArray[np.bool_],
    backend_name: str,
    speed_kmh: Mapping[str, float],
) -> dict[str, Any]:
    """`within_radius_*` is what riders see: pairs inside the store's radius, on hexes with
    residents. `table_in_radius_share` sizes an in-radius table for P-025, so it counts every pair
    in the table."""
    riders_see = in_radius & holds_residents
    raw = table.astype("<u2").tobytes()
    return {
        "backend": backend_name,
        "bands": {
            band.name: _band_minutes(table[:, :, b], riders_see) for b, band in enumerate(bands)
        },
        "complete": is_complete(table),
        "speed_kmh": dict(speed_kmh),
        "table_bytes": len(raw),
        "table_zstd_bytes": len(zstandard.ZstdCompressor(level=TABLE_ZSTD_LEVEL).compress(raw)),
        "table_in_radius_share": float(in_radius.mean()) if in_radius.size else 0.0,
    }


def is_complete(table: NDArray[np.uint16]) -> bool:
    """An unreachable pair raises in `travel_table`, so a table that reaches here is complete unless
    it has no stores or no hexes."""
    return bool(table.size and table.min() >= 1)


def graph_section(in_component: NDArray[np.bool_]) -> dict[str, Any]:
    return {"nodes": len(in_component), "main_component_share": float(in_component.mean())}


def within_targets(
    placement: Placement, target: float, table: NDArray[np.uint16]
) -> dict[str, bool]:
    checks = {"coverage": coverage(placement) >= target, "travel_complete": is_complete(table)}
    return {**checks, "all": all(checks.values())}


def rounded(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: rounded(v) for k, v in value.items()}
    if isinstance(value, list):
        return [rounded(v) for v in value]
    return round(value, REPORT_DECIMALS) if isinstance(value, float) else value


def _band_minutes(minutes: NDArray[np.uint16], riders_see: NDArray[np.bool_]) -> dict[str, float]:
    seen = minutes[riders_see]
    return {
        "p50": _percentile(minutes, 50),
        "p90": _percentile(minutes, 90),
        "max": _percentile(minutes, 100),
        "within_radius_p50": _percentile(seen, 50),
        "within_radius_p90": _percentile(seen, 90),
    }


def _percentile(values: np.ndarray, q: float) -> float:
    """0.0 over nothing, never NaN."""
    return float(np.percentile(values.astype(np.float64), q)) if values.size else 0.0
