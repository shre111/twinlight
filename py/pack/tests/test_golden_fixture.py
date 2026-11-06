import json
from pathlib import Path

import numpy as np
import pytest

from twl_pack.fixture import small_contents, write_fixture
from twl_pack.reader import PackReader
from twl_pack.writer import encode_pack

pytestmark = pytest.mark.golden

FIXTURE = Path(__file__).resolve().parents[3] / "fixtures" / "pack" / "v2" / "small"


@pytest.fixture(scope="module")
def reader() -> PackReader:
    return PackReader(FIXTURE)


@pytest.fixture(scope="module")
def probes() -> dict:
    return json.loads((FIXTURE / "probes.json").read_text(encoding="utf-8"))


def test_regenerating_gives_byte_identical_files(tmp_path):
    write_fixture(tmp_path)
    fresh = {p.name for p in tmp_path.iterdir()}
    checked_in = {p.name for p in FIXTURE.iterdir()}
    assert fresh == checked_in
    for name in sorted(fresh):
        assert (tmp_path / name).read_bytes() == (FIXTURE / name).read_bytes(), name


def test_fixture_module_and_checked_in_pack_agree_on_the_hash(reader):
    assert encode_pack(small_contents()).pack_hash == reader.pack_hash


def test_checked_in_pack_verifies_and_stays_small(reader):
    reader.verify()
    assert reader.counts["buildings"] + reader.counts["detailBuildings"] <= 300
    assert sum(p.stat().st_size for p in FIXTURE.iterdir()) < 200_000


def test_probes_file_is_sorted_indent_2_with_trailing_newline():
    text = (FIXTURE / "probes.json").read_text(encoding="utf-8")
    assert text == json.dumps(json.loads(text), sort_keys=True, indent=2) + "\n"


def test_fixture_is_a_city_with_the_required_shape(reader):
    m = reader.manifest
    assert m["kind"] == "city"
    assert "seed" not in m
    assert reader.counts["buildingTileLevel"] == 3
    assert reader.counts["detailBuildings"] > 0
    assert len(m["segments"]) >= 2
    assert len(m["bands"]) >= 2
    travel = reader.travel()
    assert travel is not None
    assert travel.shape == (reader.counts["stores"], reader.counts["hexes"], len(m["bands"]))


def test_fixture_has_a_polygon_with_a_hole_and_a_lake(reader):
    river = reader.river()
    assert (river.rings > 1).any()
    assert (river.kind == 1).any()
    assert (river.kind == 0).any()


def test_fixture_has_a_one_point_road(reader):
    assert (reader.roads().points == 1).any()


def test_fixture_road_first_point_needs_the_wrap(reader):
    roads = reader.roads()
    assert roads.qx[0] > 32767 or roads.qy[0] > 32767


def test_fixture_roads_jump_over_half_the_bbox_on_both_axes(reader):
    roads = reader.roads()
    inside = np.ones(len(roads.qx), dtype=bool)
    inside[roads.offsets[:-1]] = False
    for q in (roads.qx, roads.qy):
        step = np.abs(np.diff(q.astype(np.int64), prepend=0))
        assert (step[inside] > 32767).any()


def test_fixture_clips_each_size_column_and_has_morton_ties(reader):
    clipped = encode_pack(small_contents()).clipped
    assert clipped["width"] > 0 and clipped["depth"] > 0 and clipped["height"] > 0
    b = reader.buildings()
    assert len(np.unique(b.code)) < len(b.code)


def test_checked_in_pack_holds_what_the_fixture_module_asked_for(reader):
    contents = small_contents()
    order = encode_pack(contents).building_order
    half_x = (reader.bbox[2] - reader.bbox[0]) / 65535 / 2 + 1e-9
    half_y = (reader.bbox[3] - reader.bbox[1]) / 65535 / 2 + 1e-9

    b, given = reader.buildings(), contents.buildings
    assert np.abs(reader.x_m(b.qx) - given.x[order]).max() <= half_x
    assert np.abs(reader.y_m(b.qy) - given.y[order]).max() <= half_y
    assert (b.district == given.district[order]).all()
    unclipped = given.height[order] < 451
    assert np.abs(b.height_m - given.height[order])[unclipped].max() < 2.0

    r = reader.residents()
    assert (order[r.building] == contents.residents.building).all()
    assert (r.floor == contents.residents.floor).all()
    assert (r.segment == contents.residents.segment).all()
    assert np.abs(reader.x_m(r.qx) - contents.residents.x).max() <= half_x

    roads = reader.roads()
    assert (roads.points == contents.roads.points).all()
    assert (roads.road_class == contents.roads.road_class).all()
    assert np.abs(reader.x_m(roads.qx) - contents.roads.x).max() <= half_x
    assert np.abs(reader.y_m(roads.qy) - contents.roads.y).max() <= half_y

    engine = reader.residents_engine()
    assert (engine.persona_ref == contents.residents_engine.persona_ref).all()
    assert (engine.weight == contents.residents_engine.weight.astype("<f4")).all()
    graph = reader.graph()
    assert (graph.offsets == contents.graph.offsets).all()
    assert (graph.targets == contents.graph.targets).all()
    assert (reader.hexes() == contents.hexes).all()
    assert (reader.travel().ravel() == contents.travel).all()


def test_fixture_has_a_non_ascii_name_local(reader):
    names = [d.get("name_local", "") for d in reader.districts()]
    assert any(not n.isascii() for n in names)


def test_fixture_tile_table_spans_all_buildings(reader):
    ts = reader.buildings().tile_start
    assert len(ts) == 4**3 + 1
    assert ts[0] == 0
    assert ts[-1] == reader.counts["buildings"]
    assert (np.diff(ts.astype(np.int64)) >= 0).all()


def approx(a, b):
    return a == pytest.approx(b, rel=1e-9, abs=1e-9)


def check_building_row(reader, rows, row):
    i = row["index"]
    for key in ("qx", "qy", "width", "depth", "height", "rotation", "district"):
        assert getattr(rows, key)[i] == row[key], (i, key)
    assert approx(float(reader.x_m(rows.qx[i : i + 1])[0]), row["x_m"])
    assert approx(float(reader.y_m(rows.qy[i : i + 1])[0]), row["y_m"])
    assert approx(float(rows.width_m[i]), row["width_m"])
    assert approx(float(rows.depth_m[i]), row["depth_m"])
    assert approx(float(rows.height_m[i]), row["height_m"])
    assert approx(float(rows.rotation_rad[i]), row["rotation_rad"])


def test_probe_pack_hash(reader, probes):
    assert probes["pack_hash"] == reader.pack_hash


def test_probe_tile_starts(reader, probes):
    ts = reader.buildings().tile_start
    assert probes["tile_start"]
    for row in probes["tile_start"]:
        assert ts[row["tile"]] == row["start"]


def test_probe_buildings(reader, probes):
    b = reader.buildings()
    n = reader.counts["buildings"]
    assert probes["buildings"]
    for row in probes["buildings"]:
        check_building_row(reader, b, row)
    indices = {row["index"] for row in probes["buildings"]}
    assert {0, n - 1} <= indices
    for column in (b.width, b.depth, b.height):
        assert indices & set(np.flatnonzero(column == column.max()).tolist())


def test_probe_buildings_straddle_a_tile_boundary(reader, probes):
    ts = reader.buildings().tile_start
    indices = {row["index"] for row in probes["buildings"]}
    boundaries = {int(s) for s in ts if 0 < s < reader.counts["buildings"]}
    assert any(s - 1 in indices and s in indices for s in boundaries)


def test_probe_detail(reader, probes):
    d = reader.detail()
    assert d is not None
    assert probes["detail"]
    for row in probes["detail"]:
        check_building_row(reader, d, row)


def test_probe_roads(reader, probes):
    r = reader.roads()
    assert probes["roads"]
    for row in probes["roads"]:
        i = row["road"]
        assert r.road_class[i] == row["class"]
        lo, hi = r.offsets[i], r.offsets[i + 1]
        got = np.stack([r.qx[lo:hi], r.qy[lo:hi]], axis=1).astype(int).tolist()
        assert row["points"] == got
    roads = {row["road"] for row in probes["roads"]}
    assert {0, reader.counts["roads"] - 1} <= roads
    assert any(len(row["points"]) == 1 for row in probes["roads"])


def test_probe_residents(reader, probes):
    r = reader.residents()
    assert probes["residents"]
    for row in probes["residents"]:
        i = row["index"]
        for key in ("qx", "qy", "floor", "building", "segment"):
            assert getattr(r, key)[i] == row[key], (i, key)
        assert approx(float(r.z_m[i]), row["z_m"])
    indices = {row["index"] for row in probes["residents"]}
    assert {0, reader.counts["residents"] - 1} <= indices


def test_probe_river(reader, probes):
    river = reader.river()
    ring_starts = np.concatenate([[0], np.cumsum(river.rings)], dtype=np.int64)
    assert probes["river"]
    for row in probes["river"]:
        p = row["polygon"]
        assert river.kind[p] == row["kind"]
        first_ring, end_ring = ring_starts[p], ring_starts[p + 1]
        sizes = np.diff(river.offsets[first_ring : end_ring + 1])
        assert row["rings"] == sizes.tolist()
        start = river.offsets[first_ring]
        assert row["first_point"] == [float(river.x[start]), float(river.y[start])]
        stop = river.offsets[end_ring]
        flat = [point for ring in row["points"] for point in ring]
        assert flat == np.stack([river.x[start:stop], river.y[start:stop]], axis=1).tolist()
        assert [len(ring) for ring in row["points"]] == row["rings"]
    assert any(len(row["rings"]) > 1 for row in probes["river"])


def test_probe_stores(reader, probes):
    s = reader.stores()
    assert [row["index"] for row in probes["stores"]] == list(range(reader.counts["stores"]))
    for row in probes["stores"]:
        i = row["index"]
        assert s.hex[i] == row["hex"] and s.district[i] == row["district"]
        assert [float(s.x[i]), float(s.y[i]), float(s.radius_m[i])] == [
            row["x"],
            row["y"],
            row["radius_m"],
        ]
