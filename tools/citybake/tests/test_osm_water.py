import numpy as np
import pytest
from osm_fixture import TinyCity, ring_area

from twl_citybake.osm.extract import OsmLayers


def polygon_rings(layers: OsmLayers, polygon: int) -> list[np.ndarray]:
    water = layers.water
    first, last = water["poly_offsets"][polygon], water["poly_offsets"][polygon + 1]
    return [
        water["xy"][water["ring_offsets"][r] : water["ring_offsets"][r + 1]].astype(np.float64)
        for r in range(first, last)
    ]


def exterior_area(layers: OsmLayers, polygon: int) -> float:
    return ring_area(polygon_rings(layers, polygon)[0])


def test_water_encoding_has_the_specified_dtypes_and_shapes(layers: OsmLayers) -> None:
    water = layers.water
    assert water["poly_kind"].dtype == np.uint8
    assert water["xy"].dtype == np.float32
    assert water["xy"].ndim == 2 and water["xy"].shape[1] == 2
    assert np.issubdtype(water["poly_offsets"].dtype, np.unsignedinteger)
    assert np.issubdtype(water["ring_offsets"].dtype, np.unsignedinteger)


def test_water_offsets_are_monotone_and_terminal_offsets_are_consistent(layers: OsmLayers) -> None:
    water = layers.water
    polygons = len(water["poly_kind"])

    assert len(water["poly_offsets"]) == polygons + 1
    assert water["poly_offsets"][0] == 0
    assert np.all(np.diff(water["poly_offsets"].astype(np.int64)) >= 1)
    assert water["poly_offsets"][-1] == len(water["ring_offsets"]) - 1
    assert water["ring_offsets"][0] == 0
    assert np.all(np.diff(water["ring_offsets"].astype(np.int64)) >= 3)
    assert water["ring_offsets"][-1] == len(water["xy"])


def test_river_polygon_comes_first_and_lakes_follow(layers: OsmLayers) -> None:
    assert layers.water["poly_kind"].tolist() == [0, 1, 1]


def test_river_polygon_is_clipped_to_the_bbox(layers: OsmLayers, tiny_city: TinyCity) -> None:
    ring = polygon_rings(layers, 0)[0]
    ymin, ymax = tiny_city.extent[1], tiny_city.extent[3]

    assert ring[:, 1].min() >= ymin - 1.0
    assert ring[:, 1].max() <= ymax + 1.0
    assert ring[:, 1].max() < 1100
    assert ring[:, 0].min() == pytest.approx(-40, abs=0.5)
    assert ring[:, 0].max() == pytest.approx(40, abs=0.5)
    assert exterior_area(layers, 0) == pytest.approx(80 * tiny_city.clip_height_m, rel=0.02)


def test_lakes_above_threshold_are_kept_ordered_by_descending_area(layers: OsmLayers) -> None:
    assert exterior_area(layers, 1) == pytest.approx(200 * 200, rel=0.01)
    assert exterior_area(layers, 2) == pytest.approx(120 * 120, rel=0.01)
    assert exterior_area(layers, 1) > exterior_area(layers, 2)


def test_small_pond_is_dropped(layers: OsmLayers) -> None:
    centres = [
        polygon_rings(layers, p)[0].mean(axis=0) for p in range(len(layers.water["poly_kind"]))
    ]
    assert not any(np.allclose(c, (-600, -600), atol=30) for c in centres)


def test_lake_is_judged_by_area_after_clipping(layers: OsmLayers) -> None:
    centres = [
        polygon_rings(layers, p)[0].mean(axis=0) for p in range(len(layers.water["poly_kind"]))
    ]
    assert not any(c[0] > 900 and abs(c[1] - 250) < 100 for c in centres)


def test_water_report_matches_the_polygons(layers: OsmLayers, tiny_city: TinyCity) -> None:
    water = layers.report["water"]

    assert water["lakes"] == 2
    assert water["lake_area_km2"] == pytest.approx((40_000 + 14_400) / 1e6, abs=0.001)
    assert water["river_area_km2"] == pytest.approx(80 * tiny_city.clip_height_m / 1e6, rel=0.02)
