import dataclasses
import json
import re

import numpy as np
import pytest

from twl_pack.format import PackError
from twl_pack.types import Band
from twl_pack.writer import encode_pack

EXTERIOR = [(0, 0), (10, 0), (10, 10), (0, 10)]
HOLE = [(2, 2), (2, 8), (8, 8), (8, 2)]
LAKE = [(20, 20), (30, 20), (25, 30)]


def rep(c, section, **kw):
    return dataclasses.replace(c, **{section: dataclasses.replace(getattr(c, section), **kw)})


def a(*v):
    return np.array(v)


def districts(n):
    return [{"id": i, "name": str(i), "x": 0.0, "y": 0.0, "east": 0.0} for i in range(n)]


XY = r"\bx\b|\by\b|buildings|residents|roads|stores"

CITY_REFUSALS = {
    "buildings width short": (lambda c: rep(c, "buildings", width=c.buildings.width[:5]), "width"),
    "buildings depth long": (
        lambda c: rep(c, "buildings", depth=np.append(c.buildings.depth, 1.0)),
        "depth",
    ),
    "buildings district short": (
        lambda c: rep(c, "buildings", district=c.buildings.district[:5]),
        "district",
    ),
    "detail height short": (lambda c: rep(c, "detail", height=a(4.0)), "height"),
    "residents floor short": (lambda c: rep(c, "residents", floor=a(0, 1)), "floor"),
    "residents segment short": (lambda c: rep(c, "residents", segment=a(0, 1)), "segment"),
    "engine weight short": (lambda c: rep(c, "residents_engine", weight=a(1.0)), "weight"),
    "engine persona_ref short": (lambda c: rep(c, "residents_engine", persona_ref=a(1)), "persona"),
    "stores radius_m short": (lambda c: rep(c, "stores", radius_m=a(1.0)), "radius_m"),
    "roads road_class short": (lambda c: rep(c, "roads", road_class=a(0)), "road_class"),
    "roads y short": (lambda c: rep(c, "roads", y=a(0.0, 0, 0)), XY),
    "graph strength short": (lambda c: rep(c, "graph", strength=a(1.0)), "strength"),
    "graph kind short": (lambda c: rep(c, "graph", kind=a(0)), "kind"),
    "buildings x nan": (
        lambda c: rep(c, "buildings", x=np.array([np.nan, 0, 1, 0, 1, 0])),
        XY,
    ),
    "buildings height inf": (
        lambda c: rep(c, "buildings", height=np.array([np.inf, 1, 1, 1, 1, 1.0])),
        "height",
    ),
    "buildings rotation nan": (
        lambda c: rep(c, "buildings", rotation=np.array([np.nan, 0, 0, 0, 0, 0.0])),
        "rotation",
    ),
    "detail depth nan": (lambda c: rep(c, "detail", depth=a(np.nan, 1.0)), "depth"),
    "roads y inf": (lambda c: rep(c, "roads", y=a(0.0, 0, np.inf, 9)), XY),
    "residents x nan": (lambda c: rep(c, "residents", x=a(np.nan, 1.0, 2.0)), XY),
    "stores radius_m inf": (lambda c: rep(c, "stores", radius_m=a(np.inf, 1.0)), "radius_m"),
    "stores x nan": (lambda c: rep(c, "stores", x=a(np.nan, 1.0)), XY),
    "engine weight nan": (
        lambda c: rep(c, "residents_engine", weight=a(1.0, np.nan, 1.0)),
        "weight",
    ),
    "graph strength nan": (lambda c: rep(c, "graph", strength=a(1.0, np.nan, 1.0)), "strength"),
    "river x nan": (
        lambda c: rep(c, "river", x=np.where(np.arange(11) == 3, np.nan, c.river.x)),
        r"river|\bx\b",
    ),
    "kind unknown": (lambda c: dataclasses.replace(c, kind="village"), "kind"),
    "city with seed": (lambda c: dataclasses.replace(c, seed=1), "seed"),
    "city without hexes": (lambda c: dataclasses.replace(c, hexes=None), "hexes"),
    "city without travel": (lambda c: dataclasses.replace(c, travel=None), "travel"),
    "city without bands": (lambda c: dataclasses.replace(c, bands=[]), "bands"),
    "tile_level negative": (lambda c: dataclasses.replace(c, tile_level=-1), "tile_level"),
    "tile_level above max": (lambda c: dataclasses.replace(c, tile_level=9), "tile_level"),
    "too many districts": (lambda c: dataclasses.replace(c, districts=districts(256)), "district"),
    "district ids descending": (
        lambda c: dataclasses.replace(c, districts=districts(2)[::-1]),
        "district",
    ),
    "district ids repeated": (
        lambda c: dataclasses.replace(c, districts=[districts(1)[0]] * 2),
        "district",
    ),
    "building district unknown": (
        lambda c: rep(c, "buildings", district=a(0, 1, 0, 1, 1, 5)),
        "district",
    ),
    "detail district unknown": (lambda c: rep(c, "detail", district=a(1, 5)), "district"),
    "store district unknown": (lambda c: rep(c, "stores", district=a(0, 5)), "district"),
    "road with no points": (
        lambda c: rep(c, "roads", points=a(0, 4)),
        "points",
    ),
    "road with 65536 points": (
        lambda c: rep(
            c,
            "roads",
            points=a(65536),
            road_class=a(0),
            x=np.zeros(65536),
            y=np.zeros(65536),
        ),
        "points",
    ),
    "road points sum high": (lambda c: rep(c, "roads", points=a(3, 2)), "points"),
    "road points sum low": (lambda c: rep(c, "roads", points=a(2, 1)), "points"),
    "road_class 3": (lambda c: rep(c, "roads", road_class=a(0, 3)), "road_class"),
    "road_class negative": (lambda c: rep(c, "roads", road_class=a(-1, 0)), "road_class"),
    "floor negative": (lambda c: rep(c, "residents", floor=a(0, -1, 3)), "floor"),
    "floor 256": (lambda c: rep(c, "residents", floor=a(0, 256, 3)), "floor"),
    "building out of range": (lambda c: rep(c, "residents", building=a(0, 6, 2)), "building"),
    "building negative": (lambda c: rep(c, "residents", building=a(0, -1, 2)), "building"),
    "segment out of range": (lambda c: rep(c, "residents", segment=a(0, 2, 1)), "segment"),
    "segment negative": (lambda c: rep(c, "residents", segment=a(0, -1, 1)), "segment"),
    "persona_ref negative": (
        lambda c: rep(c, "residents_engine", persona_ref=a(-1, 0, 0)),
        "persona_ref",
    ),
    "persona_ref above u32": (
        lambda c: rep(c, "residents_engine", persona_ref=a(2**32, 0, 0)),
        "persona_ref",
    ),
    "weight zero": (lambda c: rep(c, "residents_engine", weight=a(1.0, 0.0, 1.0)), "weight"),
    "weight negative": (lambda c: rep(c, "residents_engine", weight=a(1.0, -1.0, 1.0)), "weight"),
    "resident hex out of range": (lambda c: rep(c, "residents_engine", hex=a(0, 2, 0)), "hex"),
    "store hex out of range": (lambda c: rep(c, "stores", hex=a(0, 2)), "hex"),
    "travel too short": (lambda c: dataclasses.replace(c, travel=np.arange(1, 8)), "travel"),
    "travel too long": (lambda c: dataclasses.replace(c, travel=np.arange(1, 10)), "travel"),
    "travel minute zero": (
        lambda c: dataclasses.replace(c, travel=a(5, 0, 7, 8, 9, 10, 11, 12)),
        "travel",
    ),
    "travel minute 65536": (
        lambda c: dataclasses.replace(c, travel=a(5, 65536, 7, 8, 9, 10, 11, 12)),
        "travel",
    ),
    "travel minute negative": (
        lambda c: dataclasses.replace(c, travel=a(5, -3, 7, 8, 9, 10, 11, 12)),
        "travel",
    ),
    "bands descending": (
        lambda c: dataclasses.replace(c, bands=[Band("a", 500), Band("b", 400)]),
        "bands",
    ),
    "bands tied": (
        lambda c: dataclasses.replace(c, bands=[Band("a", 400), Band("b", 400)]),
        "bands",
    ),
    "band start 1440": (
        lambda c: dataclasses.replace(c, bands=[Band("a", 0), Band("b", 1440)]),
        "bands",
    ),
    "band start negative": (
        lambda c: dataclasses.replace(c, bands=[Band("a", -1), Band("b", 10)]),
        "bands",
    ),
    "graph offsets wrong length": (lambda c: rep(c, "graph", offsets=a(0, 2, 3)), "offsets"),
    "graph offsets not from zero": (lambda c: rep(c, "graph", offsets=a(1, 2, 3, 3)), "offsets"),
    "graph offsets decreasing": (lambda c: rep(c, "graph", offsets=a(0, 2, 1, 3)), "offsets"),
    "graph offsets end short": (lambda c: rep(c, "graph", offsets=a(0, 2, 3, 2)), "offsets"),
    "graph offsets end long": (lambda c: rep(c, "graph", offsets=a(0, 2, 3, 4)), "offsets"),
    "graph target out of range": (lambda c: rep(c, "graph", targets=a(1, 3, 0)), "targets"),
    "graph target negative": (lambda c: rep(c, "graph", targets=a(1, -1, 0)), "targets"),
    "graph kind 4": (lambda c: rep(c, "graph", kind=a(0, 4, 3)), "kind"),
    "river rings entry zero": (lambda c: rep(c, "river", rings=a(3, 0)), "rings"),
    "river rings sum mismatch": (lambda c: rep(c, "river", rings=a(1, 1)), "rings"),
    "river kind 2": (lambda c: rep(c, "river", kind=a(0, 2)), "kind"),
    "river offsets not from zero": (
        lambda c: rep(c, "river", offsets=a(1, 4, 8, 11)),
        "offsets",
    ),
    "river offsets end short": (lambda c: rep(c, "river", offsets=a(0, 4, 8, 10)), "offsets"),
    "river offsets decreasing": (lambda c: rep(c, "river", offsets=a(0, 8, 4, 11)), "offsets"),
    "residents floor fractional": (lambda c: rep(c, "residents", floor=a(0.5, 1.0, 0.0)), "floor"),
    "travel fractional": (lambda c: dataclasses.replace(c, travel=np.arange(8) + 1.5), "travel"),
    "buildings width negative": (
        lambda c: rep(c, "buildings", width=np.array([-1.0, 1, 1, 1, 1, 1])),
        "width",
    ),
    "districts repeat an id": (
        lambda c: dataclasses.replace(c, districts=[c.districts[0], *c.districts]),
        "districts",
    ),
    "district id past u8": (
        lambda c: dataclasses.replace(c, districts=[*c.districts, {**c.districts[0], "id": 256}]),
        "districts",
    ),
    "district id past int64": (
        lambda c: dataclasses.replace(c, districts=[*c.districts, {**c.districts[0], "id": 2**64}]),
        "districts",
    ),
    "district holds nan": (
        lambda c: dataclasses.replace(
            c, districts=[{**c.districts[0], "x": float("nan")}, *c.districts[1:]]
        ),
        "districts",
    ),
}


@pytest.mark.parametrize(("mutate", "column"), CITY_REFUSALS.values(), ids=CITY_REFUSALS.keys())
def test_city_refusal_names_the_column(city, mutate, column):
    with pytest.raises(PackError, match=re.compile(column, re.IGNORECASE)):
        encode_pack(mutate(city))


def test_pack_without_buildings_is_refused(city):
    empty = {f.name: np.zeros(0) for f in dataclasses.fields(city.buildings)}
    with pytest.raises(PackError, match="building"):
        encode_pack(dataclasses.replace(city, buildings=type(city.buildings)(**empty)))


@pytest.mark.parametrize("key", ["id", "name", "x", "y", "east"])
def test_district_missing_key_is_refused(city, key):
    bad = [{k: v for k, v in d.items() if k != key} for d in city.districts[:1]]
    with pytest.raises(PackError, match=re.compile(rf"{key}|district", re.IGNORECASE)):
        encode_pack(dataclasses.replace(city, districts=bad + city.districts[1:]))


RIVER_REFUSALS = {
    "ring under three points": ([[[(0, 0), (1, 0)]]], [0], "ring"),
    "closed ring": ([[[*EXTERIOR, EXTERIOR[0]]]], [0], "closed|ring"),
    "exterior clockwise": ([[EXTERIOR[::-1]]], [0], "counter|exterior|ring"),
    "hole counter-clockwise": ([[EXTERIOR, HOLE[::-1]]], [0], "clockwise|hole|ring"),
    "lake exterior clockwise": ([[LAKE[::-1]]], [1], "counter|exterior|ring"),
    "exterior with no area": ([[[(0, 0), (1, 1), (2, 2)]]], [0], "exterior"),
    "hole with no area": ([[EXTERIOR, [(2, 2), (3, 3), (4, 4)]]], [0], "hole"),
}


@pytest.mark.parametrize(
    ("polygons", "kinds", "column"), RIVER_REFUSALS.values(), ids=RIVER_REFUSALS.keys()
)
def test_river_ring_refusal_names_the_column(city, river_from, polygons, kinds, column):
    bad = dataclasses.replace(city, river=river_from(polygons, kinds))
    with pytest.raises(PackError, match=re.compile(column, re.IGNORECASE)):
        encode_pack(bad)


TOY_REFUSALS = {
    "toy without seed": (lambda c: dataclasses.replace(c, seed=None), "seed"),
    "toy with detail": (lambda c: dataclasses.replace(c, detail=c.buildings), "detail"),
    "toy with hexes": (lambda c: dataclasses.replace(c, hexes=a(1, 2)), "hexes"),
    "toy with travel": (lambda c: dataclasses.replace(c, travel=a(1)), "travel"),
    "toy with bands": (lambda c: dataclasses.replace(c, bands=[Band("a", 0)]), "bands"),
    "toy resident hex nonzero": (
        lambda c: rep(c, "residents_engine", hex=a(0, 0, 1, 0)),
        "hex",
    ),
    "toy store hex nonzero": (lambda c: rep(c, "stores", hex=a(1)), "hex"),
}


@pytest.mark.parametrize(("mutate", "column"), TOY_REFUSALS.values(), ids=TOY_REFUSALS.keys())
def test_toy_refusal_names_the_column(toy, mutate, column):
    with pytest.raises(PackError, match=re.compile(column, re.IGNORECASE)):
        encode_pack(mutate(toy))


def test_toy_shaped_pack_is_accepted(toy):
    out = encode_pack(toy)
    counts = json.loads(out.manifest)["counts"]
    assert counts["hexes"] == 0 and counts["bands"] == 0
    assert "hexes.bin" not in out.raw and "travel.bin" not in out.raw


def test_toy_river_ring_may_run_past_the_bbox(toy):
    out = encode_pack(toy)
    bbox = json.loads(out.manifest)["bbox"]
    assert min(toy.river.x) < bbox[0] and max(toy.river.x) > bbox[2]


ACCEPTED = {
    "tile_level 0": lambda c: dataclasses.replace(c, tile_level=0),
    "tile_level 8": lambda c: dataclasses.replace(c, tile_level=8),
    "255 districts": lambda c: dataclasses.replace(c, districts=districts(255)),
    "road of 65535 points": lambda c: rep(
        c, "roads", points=a(65535), road_class=a(0), x=np.zeros(65535), y=np.zeros(65535)
    ),
    "floor 255": lambda c: rep(c, "residents", floor=a(255, 255, 255)),
    "top segment": lambda c: rep(c, "residents", segment=a(1, 1, 1)),
    "persona_ref u32 max": lambda c: rep(
        c, "residents_engine", persona_ref=a(0xFFFFFFFF, 0xFFFFFFFF, 0)
    ),
    "tiny positive weight": lambda c: rep(c, "residents_engine", weight=a(1e-9, 1, 1)),
    "travel at both limits": lambda c: dataclasses.replace(
        c, travel=a(1, 65535, 7, 8, 9, 10, 11, 12)
    ),
    "band starts 0 and 1439": lambda c: dataclasses.replace(
        c, bands=[Band("a", 0), Band("b", 1439)]
    ),
    "one-point road": lambda c: rep(
        c, "roads", points=a(1, 3), x=a(0.0, 0, 1, 2), y=a(0.0, 0, 1, 2)
    ),
    "isolated residents in graph": lambda c: rep(
        c, "graph", offsets=a(0, 0, 0, 0), targets=a(), kind=a(), strength=a()
    ),
}


@pytest.mark.parametrize("mutate", ACCEPTED.values(), ids=ACCEPTED.keys())
def test_boundary_values_are_accepted(city, mutate):
    encode_pack(mutate(city))
