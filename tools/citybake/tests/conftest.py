import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from osm_fixture import BBOX, OsmBuilder, TinyCity, clip_extent, populate_tiny_city

from twl_citybake.projection import LocalProjection

if TYPE_CHECKING:
    from twl_citybake.config import CityConfig
    from twl_citybake.osm.extract import OsmLayers


@pytest.fixture(scope="session")
def tiny_city(tmp_path_factory: pytest.TempPathFactory) -> TinyCity:
    projection = LocalProjection.for_bbox(BBOX)
    builder = OsmBuilder(projection)
    populate_tiny_city(builder)
    path = tmp_path_factory.mktemp("tiny-city") / "tiny.osm"
    builder.write(path)
    return TinyCity(path=path, projection=projection, extent=clip_extent(projection))


@pytest.fixture(scope="session")
def city_config() -> "CityConfig":
    from twl_citybake.config import CityConfig, DistrictConfig, OsmSource, RiverConfig

    return CityConfig(
        slug="tinyville",
        name="Tinyville",
        bbox=BBOX,
        osm=OsmSource(url="https://example.invalid/tiny.osm.pbf", md5="0" * 32),
        river=RiverConfig(names=("Sabarmati River", "Sabarmati")),
        districts=DistrictConfig(
            place_kinds=("suburb", "neighbourhood", "quarter", "village", "town")
        ),
    )


@pytest.fixture(scope="session")
def layers(tiny_city: TinyCity, city_config: "CityConfig") -> "OsmLayers":
    from twl_citybake.osm.extract import extract_osm

    return extract_osm(tiny_city.path, city_config)
