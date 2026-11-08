import math

import numpy as np
import pytest

from twl_pack.types import (
    Band,
    BuildingColumns,
    GraphColumns,
    PackContents,
    ResidentColumns,
    ResidentEngineColumns,
    RiverColumns,
    RoadColumns,
    StoreColumns,
)


def _river_from(polygons, kinds):
    """polygons: list of polygons, each a list of rings, each a list of (x, y)."""
    rings = [ring for polygon in polygons for ring in polygon]
    points = [p for ring in rings for p in ring]
    return RiverColumns(
        offsets=np.cumsum([0] + [len(r) for r in rings]),
        x=np.array([p[0] for p in points], dtype=np.float32),
        y=np.array([p[1] for p in points], dtype=np.float32),
        kind=np.array(kinds),
        rings=np.array([len(p) for p in polygons]),
    )


EXTERIOR = [(0, 0), (10, 0), (10, 10), (0, 10)]
HOLE = [(2, 2), (2, 8), (8, 8), (8, 2)]
LAKE = [(20, 20), (30, 20), (25, 30)]


@pytest.fixture
def river_from():
    return _river_from


@pytest.fixture
def city() -> PackContents:
    """Six buildings with Morton codes 0x55555555, 0xAAAAAAAA, 1, 2, 1, 0 in input order, so the
    stable pack order is input rows [5, 2, 4, 3, 0, 1]. The bbox is exactly 0..65535 on both
    axes, so quantised values equal the metres."""
    pi = math.pi
    return PackContents(
        kind="city",
        name="hand",
        crs="EPSG:32643",
        timezone="Asia/Dubai",
        weekend_days=[6, 7],
        segments=["young", "old"],
        bands=[Band("morning", 360), Band("evening", 1080)],
        attributions=["OSM", "Census", "OSM"],
        districts=[
            {"id": 0, "name": "A", "name_local": "ए", "x": 0.0, "y": 0.0, "east": 0.0},
            {"id": 1, "name": "B", "name_local": "B", "x": 5.0, "y": 5.0, "east": 1.0},
        ],
        buildings=BuildingColumns(
            x=np.array([65535, 0, 1, 0, 1, 0], dtype=float),
            y=np.array([0, 65535, 0, 1, 0, 0], dtype=float),
            width=np.array([4, 9, 16, 25, 36, 100], dtype=float),
            depth=np.array([100, 36, 25, 16, 9, 4], dtype=float),
            height=np.array([10, 20, 30, 40, 50, 60], dtype=float),
            rotation=np.array([0, pi / 4, pi / 3, 2 * pi / 3, pi, -pi / 4]),
            district=np.array([0, 1, 0, 1, 1, 0]),
        ),
        roads=RoadColumns(
            points=np.array([3, 1]),
            road_class=np.array([0, 2]),
            x=np.array([65535, 0, 40000, 7], dtype=float),
            y=np.array([0, 0, 0, 9], dtype=float),
        ),
        residents=ResidentColumns(
            x=np.array([10, 65535, 0], dtype=float),
            y=np.array([20, 0, 65535], dtype=float),
            floor=np.array([0, 3, 255]),
            building=np.array([0, 5, 2]),
            segment=np.array([0, 1, 1]),
        ),
        residents_engine=ResidentEngineColumns(
            hex=np.array([1, 0, 1]),
            persona_ref=np.array([5, 0xFFFFFFFF, 7]),
            weight=np.array([1.0, 0.5, 2.0]),
            tier=np.array([2, 0, 1]),
        ),
        stores=StoreColumns(
            x=np.array([100.5, 300.0]),
            y=np.array([200.25, 400.0]),
            hex=np.array([0, 1]),
            radius_m=np.array([1500.0, 2000.5]),
            district=np.array([0, 1]),
        ),
        river=_river_from([[EXTERIOR, HOLE], [LAKE]], [0, 1]),
        graph=GraphColumns(
            offsets=np.array([0, 2, 3, 3]),
            targets=np.array([1, 2, 0]),
            kind=np.array([0, 2, 3]),
            strength=np.array([0.5, 1.0, 0.25]),
        ),
        detail=BuildingColumns(
            x=np.array([5.0, 2.0]),
            y=np.array([5.0, 2.0]),
            width=np.array([4.0, 9.0]),
            depth=np.array([4.0, 9.0]),
            height=np.array([4.0, 9.0]),
            rotation=np.array([0.0, 0.0]),
            district=np.array([1, 0]),
        ),
        hexes=np.array([0x8828308281FFFFF, 0x8828308283FFFFF], dtype=np.uint64),
        travel=np.array([5, 6, 7, 8, 9, 10, 11, 12]),
        tile_level=1,
    )


@pytest.fixture
def toy() -> PackContents:
    return PackContents(
        kind="toy",
        name="toy",
        crs="local",
        timezone="Asia/Kolkata",
        weekend_days=(6, 7),
        segments=["a0", "a1", "a2", "a3", "a4"],
        bands=[],
        attributions=[],
        districts=[
            {"id": 0, "name": "West", "x": -100.0, "y": 0.0, "east": 0.0},
            {"id": 1, "name": "East", "x": 100.0, "y": 0.0, "east": 1.0},
        ],
        buildings=BuildingColumns(
            x=np.array([-300.0, 250.0, 10.0]),
            y=np.array([-200.0, 120.0, 0.0]),
            width=np.array([20.0, 15.0, 30.0]),
            depth=np.array([20.0, 15.0, 30.0]),
            height=np.array([9.0, 12.0, 30.0]),
            rotation=np.array([0.0, 1.0, 2.0]),
            district=np.array([0, 1, 0]),
        ),
        roads=RoadColumns(
            points=np.array([2]),
            road_class=np.array([1]),
            x=np.array([-300.0, 250.0]),
            y=np.array([-200.0, 120.0]),
        ),
        residents=ResidentColumns(
            x=np.array([-290.0, 240.0, 12.0, 0.0]),
            y=np.array([-190.0, 110.0, 3.0, 0.0]),
            floor=np.array([0, 1, 2, 0]),
            building=np.array([0, 1, 2, 2]),
            segment=np.array([0, 4, 2, 1]),
        ),
        residents_engine=ResidentEngineColumns(
            hex=np.zeros(4, dtype=int),
            persona_ref=np.full(4, 0xFFFFFFFF),
            weight=np.ones(4),
            tier=np.ones(4, dtype=np.uint8),
        ),
        stores=StoreColumns(
            x=np.array([0.0]),
            y=np.array([0.0]),
            hex=np.array([0]),
            radius_m=np.array([900.0]),
            district=np.array([1]),
        ),
        river=_river_from([[[(-6000, -6000), (6000, -6000), (6000, 6000), (-6000, 6000)]]], [0]),
        graph=GraphColumns(
            offsets=np.array([0, 1, 3, 3, 4]),
            targets=np.array([1, 0, 2, 0]),
            kind=np.array([0, 1, 2, 3]),
            strength=np.array([1.0, 0.5, 0.5, 0.25]),
        ),
        seed=7,
        tile_level=2,
    )
