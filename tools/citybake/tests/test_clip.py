from pathlib import Path

import osmium

from twl_citybake.config import BBox
from twl_citybake.osm.clip import clip_extract

BBOX = BBox(72.0, 23.0, 72.1, 23.1)

OSM_XML = """<?xml version="1.0" encoding="UTF-8"?>
<osm version="0.6" generator="test">
  <node id="1" version="1" lat="23.05" lon="72.05"/>
  <node id="2" version="1" lat="23.06" lon="72.05"/>
  <node id="3" version="1" lat="23.06" lon="72.06"/>
  <node id="4" version="1" lat="23.05" lon="72.20"/>
  <node id="5" version="1" lat="23.50" lon="72.50"/>
  <node id="6" version="1" lat="23.51" lon="72.50"/>
  <node id="7" version="1" lat="23.05" lon="72.07"><tag k="shop" v="kiosk"/></node>
  <way id="10" version="1">
    <nd ref="1"/><nd ref="2"/><nd ref="3"/><nd ref="1"/>
    <tag k="building" v="yes"/>
  </way>
  <way id="11" version="1">
    <nd ref="3"/><nd ref="4"/>
    <tag k="highway" v="primary"/>
  </way>
  <way id="12" version="1">
    <nd ref="5"/><nd ref="6"/>
    <tag k="highway" v="primary"/>
  </way>
  <relation id="20" version="1">
    <member type="way" ref="11" role="outer"/>
    <member type="way" ref="12" role="outer"/>
    <tag k="type" v="multipolygon"/>
    <tag k="natural" v="water"/>
  </relation>
  <relation id="21" version="1">
    <member type="way" ref="11" role=""/>
    <tag k="type" v="route"/>
    <tag k="route" v="bus"/>
  </relation>
</osm>
"""


def ids(path: Path) -> dict[str, set[int]]:
    found: dict[str, set[int]] = {"n": set(), "w": set(), "r": set()}
    for obj in osmium.FileProcessor(str(path)):
        found[obj.type_str()].add(obj.id)
    return found


def test_clip_keeps_touching_objects_complete(tmp_path: Path) -> None:
    src = tmp_path / "region.osm"
    src.write_text(OSM_XML)
    dest = tmp_path / "out" / "city.osm.pbf"

    clip_extract(src, dest, BBOX)

    found = ids(dest)
    assert found["w"] == {10, 11, 12}
    assert found["n"] == {1, 2, 3, 4, 5, 6, 7}
    assert found["r"] == {20}


def test_clip_drops_untouched_objects(tmp_path: Path) -> None:
    src = tmp_path / "region.osm"
    src.write_text(OSM_XML.replace('<member type="way" ref="12" role="outer"/>', ""))
    dest = tmp_path / "city.osm.pbf"

    clip_extract(src, dest, BBOX)

    found = ids(dest)
    assert 12 not in found["w"]
    assert found["n"].isdisjoint({5, 6})
