import json
from pathlib import Path

import numpy as np
import pytest
from pack_fixture import (
    ATTRIBUTIONS,
    BANDS,
    BUILDINGS,
    DETAIL,
    DISTRICTS,
    DROPPED_IN_WATER,
    EXPECTED_RESIDENTS,
    FLOOR_KEPT,
    KEPT,
    LOWERED,
    MOVED,
    PACK_ROAD_CLASS,
    REPOINT_DISTANCES,
    REPOINTED,
    RESIDENTS,
    ROAD_POINTS,
    SEGMENTS,
    SLUG,
    TRAVEL_SHAPE,
    WALLED_FLOOR_KEPT,
    WARD_GROUP_NAMES,
    WARD_NAMES,
    WARD_SOURCE,
    PackInputs,
    build_pack_inputs,
    fixture_city,
    patch_budget,
    rewrite_meta,
    snapshot,
)

from twl_citybake.pack import rules
from twl_citybake.pack.rules import PackStageError
from twl_citybake.pack.stage import (
    PackBudgetError,
    PackStage,
    StaleInputError,
    run_pack,
    write_pack_stage,
)
from twl_citybake.projection import LocalProjection
from twl_citybake.store import read_table, table_digest, write_table
from twl_pack import PackReader

CENTRE_TOLERANCE_M = 0.1


def write_stage(stage: PackStage, root: Path) -> tuple[Path, Path]:
    pack_dir, out_dir = root / "pack", root / "stage"
    write_pack_stage(stage, pack_dir, out_dir)
    return pack_dir, out_dir


class Baked:
    def __init__(self, inputs: PackInputs, root: Path, wards: bool = False) -> None:
        self.inputs = inputs
        self.stage = run_pack(*inputs.args(wards))
        self.pack_dir, self.out_dir = write_stage(self.stage, root)
        self.reader = PackReader(self.pack_dir)
        self.report = json.loads((self.out_dir / "report.json").read_text(encoding="utf-8"))
        self.layer = json.loads((self.out_dir / "layer.json").read_text(encoding="utf-8"))

    def building_centres(self) -> np.ndarray:
        b = self.reader.buildings()
        return np.column_stack([self.reader.x_m(b.qx), self.reader.y_m(b.qy)])

    def resident_positions(self) -> np.ndarray:
        r = self.reader.residents()
        return np.column_stack([self.reader.x_m(r.qx), self.reader.y_m(r.qy)])

    def rows_near(self, x: float, y: float) -> list[int]:
        centres = self.building_centres()
        distance = np.hypot(centres[:, 0] - x, centres[:, 1] - y)
        return np.flatnonzero(distance < CENTRE_TOLERANCE_M).tolist()

    def row_of(self, x: float, y: float) -> int:
        rows = self.rows_near(x, y)
        assert len(rows) == 1, f"{len(rows)} pack buildings at ({x}, {y})"
        return rows[0]


@pytest.fixture(scope="module")
def baked(tmp_path_factory: pytest.TempPathFactory) -> Baked:
    root = tmp_path_factory.mktemp("pack-baked")
    return Baked(build_pack_inputs(root / "data"), root / "out")


@pytest.fixture(scope="module")
def baked_with_wards(tmp_path_factory: pytest.TempPathFactory) -> Baked:
    root = tmp_path_factory.mktemp("pack-baked-wards")
    return Baked(build_pack_inputs(root / "data"), root / "out", wards=True)


@pytest.fixture
def inputs(tmp_path: Path) -> PackInputs:
    return build_pack_inputs(tmp_path / "data")


def test_the_pack_verifies(baked: Baked) -> None:
    baked.reader.verify()


def test_the_pack_is_a_city_pack_named_for_the_slug_without_a_seed(baked: Baked) -> None:
    manifest = baked.reader.manifest
    assert manifest["kind"] == "city"
    assert manifest["name"] == SLUG
    assert "seed" not in manifest


def test_the_crs_is_the_local_projection_of_the_city_bbox(baked: Baked) -> None:
    assert baked.reader.manifest["crs"] == LocalProjection.for_bbox(baked.inputs.city.bbox).proj4


def test_segments_and_bands_come_from_the_persona_and_store_layers(baked: Baked) -> None:
    assert baked.reader.manifest["segments"] == SEGMENTS
    assert baked.reader.manifest["bands"] == BANDS


def test_attributions_are_the_unique_sorted_union_of_the_input_metas(baked: Baked) -> None:
    assert baked.reader.manifest["attributions"] == ATTRIBUTIONS


def test_counts_agree_with_the_rules(baked: Baked) -> None:
    counts = baked.reader.counts
    assert counts["buildings"] == len(KEPT)
    assert counts["detailBuildings"] == len(DETAIL)
    assert counts["residents"] == len(RESIDENTS)
    assert counts["ties"] == len(RESIDENTS)
    assert counts["stores"] == 2
    assert counts["roads"] == 6
    assert counts["hexes"] == 3
    assert counts["bands"] == 2
    assert counts["buildingTileLevel"] == 6
    assert (counts["riverPolygons"], counts["riverRings"], counts["riverPoints"]) == (2, 3, 12)


def test_first_load_buildings_are_exactly_the_kept_ones_found_by_position(baked: Baked) -> None:
    found = {tuple(np.round(c, 1)) for c in baked.building_centres()}
    expected = {(BUILDINGS[i][0], BUILDINGS[i][1]) for i in KEPT}
    assert found == expected


def test_buildings_in_water_are_gone_from_both_files(baked: Baked) -> None:
    detail = baked.reader.detail()
    assert detail is not None
    for i in DROPPED_IN_WATER:
        x, y = BUILDINGS[i][:2]
        assert baked.rows_near(x, y) == []
        near = np.hypot(baked.reader.x_m(detail.qx) - x, baked.reader.y_m(detail.qy) - y)
        assert (near > CENTRE_TOLERANCE_M).all()


def test_the_isolated_small_building_is_the_whole_detail_file(baked: Baked) -> None:
    detail = baked.reader.detail()
    assert detail is not None
    assert len(detail.qx) == 1
    x, y = baked.reader.x_m(detail.qx)[0], baked.reader.y_m(detail.qy)[0]
    assert (x, y) == pytest.approx((BUILDINGS[2][0], BUILDINGS[2][1]), abs=CENTRE_TOLERANCE_M)
    assert detail.width_m[0] == pytest.approx(5.0, abs=0.3)
    assert detail.height_m[0] == pytest.approx(3.2, abs=0.3)


def test_clustered_small_buildings_stay_in_the_first_load(baked: Baked) -> None:
    for i in (3, 4):
        baked.row_of(BUILDINGS[i][0], BUILDINGS[i][1])


def test_building_columns_carry_the_baked_sizes_and_districts(baked: Baked) -> None:
    pack = baked.reader.buildings()
    for i in KEPT:
        row = baked.row_of(BUILDINGS[i][0], BUILDINGS[i][1])
        assert pack.width_m[row] == pytest.approx(BUILDINGS[i][2], abs=0.6)
        assert pack.height_m[row] == pytest.approx(BUILDINGS[i][3], abs=0.6)
        assert int(pack.district[row]) == BUILDINGS[i][4]


def test_residents_point_at_the_right_building_by_position(baked: Baked) -> None:
    pack = baked.reader.residents()
    centres = baked.building_centres()
    for r, (building, *_rest) in enumerate(EXPECTED_RESIDENTS):
        row = int(pack.building[r])
        assert tuple(centres[row]) == pytest.approx(
            (BUILDINGS[building][0], BUILDINGS[building][1]), abs=CENTRE_TOLERANCE_M
        ), f"resident {r}"


def test_residents_keep_their_building_when_the_pack_reorders_buildings(tmp_path: Path) -> None:
    # The fixture's kept buildings are already in Morton order, so the remap would go unseen.
    order = list(range(len(BUILDINGS)))[::-1]
    reordered = Baked(build_pack_inputs(tmp_path / "data", building_order=order), tmp_path / "out")
    building_order = reordered.stage.encoded.building_order
    assert (building_order != np.arange(building_order.size)).any()
    pack = reordered.reader.residents()
    centres = reordered.building_centres()
    for r, (building, *_rest) in enumerate(EXPECTED_RESIDENTS):
        assert tuple(centres[int(pack.building[r])]) == pytest.approx(
            (BUILDINGS[building][0], BUILDINGS[building][1]), abs=CENTRE_TOLERANCE_M
        ), f"resident {r}"


def test_a_stores_layer_that_records_no_road_digests_is_stale(inputs: PackInputs) -> None:
    meta = json.loads((inputs.stores / "layer.json").read_text(encoding="utf-8"))
    del meta["osm_road_nodes_sha256"]
    (inputs.stores / "layer.json").write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(StaleInputError, match="stores"):
        run_pack(*inputs.args())


def test_resident_positions_and_floors_follow_the_rehoming_rules(baked: Baked) -> None:
    pack = baked.reader.residents()
    positions = baked.resident_positions()
    for r, (_, x, y, floor) in enumerate(EXPECTED_RESIDENTS):
        assert tuple(positions[r]) == pytest.approx((x, y), abs=CENTRE_TOLERANCE_M), f"resident {r}"
        assert int(pack.floor[r]) == floor, f"resident {r}"


def test_a_resident_in_a_lake_hole_is_not_moved(baked: Baked) -> None:
    assert tuple(baked.resident_positions()[6]) == pytest.approx((-300.0, -240.0), abs=0.1)


def test_residents_keep_their_count_and_order_through_the_engine_and_graph_files(
    baked: Baked,
) -> None:
    n = len(RESIDENTS)
    engine = baked.reader.residents_engine()
    assert engine.persona_ref.tolist() == [1000 + i for i in range(n)]
    assert engine.weight.tolist() == [1.0 + 0.25 * i for i in range(n)]
    assert engine.hex.tolist() == [i % 3 for i in range(n)]
    assert baked.reader.residents().segment.tolist() == [i % 3 for i in range(n)]
    graph = baked.reader.graph()
    assert graph.offsets.tolist() == list(range(n + 1))
    assert graph.targets.tolist() == [(i + 1) % n for i in range(n)]


def test_roads_keep_their_order_and_map_bridge_and_arterial_classes(baked: Baked) -> None:
    roads = baked.reader.roads()
    assert roads.points.tolist() == ROAD_POINTS
    assert roads.road_class.tolist() == PACK_ROAD_CLASS


def test_the_river_file_holds_the_open_rings_with_the_hole_kept_on_its_lake(baked: Baked) -> None:
    river = baked.reader.river()
    assert river.kind.tolist() == [0, 1]
    assert river.rings.tolist() == [1, 2]
    assert river.offsets.tolist() == [0, 4, 8, 12]


def test_districts_are_the_packed_districts(baked: Baked) -> None:
    expected = rules.pack_districts(DISTRICTS)
    assert baked.reader.districts() == json.loads(json.dumps(expected, ensure_ascii=False))
    assert [d["east"] for d in baked.reader.districts()] == [False, True, False]
    assert baked.reader.districts()[2]["name"] == "Ōtsu Kālupur"


def test_stores_bands_hexes_and_travel_are_carried_over(baked: Baked) -> None:
    stores = baked.reader.stores()
    assert stores.x.tolist() == [50.0, 500.0]
    assert stores.hex.tolist() == [0, 2]
    hexes, travel = baked.reader.hexes(), baked.reader.travel()
    assert hexes is not None and travel is not None
    assert hexes.shape == (3,)
    assert travel.shape == TRAVEL_SHAPE
    assert travel.ravel().tolist() == list(range(1, 13))


def test_the_report_counts_buildings_exactly(baked: Baked) -> None:
    assert baked.report["buildings"] == {
        "baked": 9,
        "first_load": 6,
        "detail": 1,
        "in_water": 2,
        "in_water_would_be_first_load": 1,
        "in_water_would_be_detail": 1,
        "isolated_small": 2,
        "small": 4,
    }


def test_the_report_counts_residents_and_distances(baked: Baked) -> None:
    residents = baked.report["residents"]
    assert residents["count"] == len(RESIDENTS)
    assert residents["moved_out_of_water"] == MOVED
    assert residents["repointed"] == REPOINTED
    assert residents["floors_lowered"] == LOWERED
    distance = residents["repoint_distance_m"]
    assert distance["p50"] == pytest.approx(float(np.percentile(REPOINT_DISTANCES, 50)), abs=0.01)
    assert distance["max"] == pytest.approx(max(REPOINT_DISTANCES), abs=0.01)


def test_the_report_counts_roads(baked: Baked) -> None:
    assert baked.report["roads"] == {"count": 6, "arterial": 2, "bridge": 2, "street": 2}


def test_the_report_floor_kept_is_the_share_of_baked_floor_area_in_kept_buildings(
    baked: Baked,
) -> None:
    assert baked.report["floor_kept"] == pytest.approx(FLOOR_KEPT, abs=1e-3)


def test_the_report_carries_the_hash_sizes_budget_flags_and_clipping(baked: Baked) -> None:
    report = baked.report
    assert report["pack_hash"] == baked.reader.pack_hash == baked.stage.encoded.pack_hash
    assert report["sizes"] == baked.reader.manifest["sizes"]
    assert report["within_budget"] == {"first_load": True, "engine": True}
    assert report["clipped"] == baked.stage.encoded.clipped
    assert baked.stage.report == report


def test_without_wards_the_report_has_no_ward_groups(baked: Baked) -> None:
    assert baked.report["ward_groups"] == {}


def test_ward_groups_report_building_counts_and_floor_shares(baked_with_wards: Baked) -> None:
    groups = baked_with_wards.report["ward_groups"]
    assert set(groups) == set(WARD_GROUP_NAMES)
    walled, east = groups["walled city"], groups["east edge"]
    assert (walled["buildings"], walled["buildings_kept"]) == (5, 2)
    assert walled["floor_kept"] == pytest.approx(WALLED_FLOOR_KEPT, abs=1e-3)
    assert (east["buildings"], east["buildings_kept"]) == (2, 2)
    assert east["floor_kept"] == pytest.approx(1.0)


def test_ward_data_never_reaches_the_pack_or_the_layer_file(baked_with_wards: Baked) -> None:
    manifest = (baked_with_wards.pack_dir / "manifest.json").read_bytes()
    layer = (baked_with_wards.out_dir / "layer.json").read_bytes()
    pack_json = json.dumps(baked_with_wards.reader.districts(), ensure_ascii=False).encode()
    for blob in (manifest, layer, pack_json):
        for text in (WARD_SOURCE, "TESTWARDSRC", *WARD_NAMES, *WARD_GROUP_NAMES):
            assert text.encode() not in blob, text


def test_the_pack_is_identical_with_and_without_wards(
    baked: Baked, baked_with_wards: Baked
) -> None:
    assert baked.stage.encoded.pack_hash == baked_with_wards.stage.encoded.pack_hash


def test_only_pack_objects_and_the_manifest_are_in_the_pack_folder(baked: Baked) -> None:
    objects = set(baked.stage.encoded.objects)
    assert {p.name for p in baked.pack_dir.iterdir()} == objects | {"manifest.json"}
    assert {p.name for p in baked.out_dir.iterdir()} == {"report.json", "layer.json"}


def test_the_meta_names_the_city_seed_and_hash(baked: Baked) -> None:
    assert baked.layer["city"] == SLUG
    assert baked.layer["seed"] == baked.inputs.city.seed
    assert baked.layer["pack_hash"] == baked.reader.pack_hash


def test_the_meta_records_the_input_digests(baked: Baked) -> None:
    text = json.dumps(baked.layer)
    inputs = baked.inputs
    for folder, name in (
        (inputs.population, "residents"),
        (inputs.buildings_layers, "buildings"),
    ):
        assert table_digest(folder / name) in text, name


def test_running_twice_gives_byte_identical_files(tmp_path: Path) -> None:
    inputs = build_pack_inputs(tmp_path / "data")
    first = Baked(inputs, tmp_path / "a", wards=True)
    second = Baked(inputs, tmp_path / "b", wards=True)
    assert snapshot(first.pack_dir) == snapshot(second.pack_dir)
    assert snapshot(first.out_dir) == snapshot(second.out_dir)
    assert first.stage.encoded.manifest == second.stage.encoded.manifest


def test_a_larger_isolation_gap_keeps_a_small_building_out_of_the_detail_file(
    tmp_path: Path,
) -> None:
    city = fixture_city(isolated_m=40.0)
    baked = Baked(build_pack_inputs(tmp_path / "data", city), tmp_path / "out")
    # building 2 sits 31.5 m from building 7
    assert baked.report["buildings"]["first_load"] == 7
    assert baked.report["buildings"]["detail"] == 0
    assert baked.report["buildings"]["in_water_would_be_detail"] == 0
    baked.row_of(BUILDINGS[2][0], BUILDINGS[2][1])


def test_residents_of_a_building_kept_by_a_larger_gap_are_not_repointed(tmp_path: Path) -> None:
    city = fixture_city(isolated_m=40.0)
    baked = Baked(build_pack_inputs(tmp_path / "data", city), tmp_path / "out")
    assert baked.report["residents"]["repointed"] == 2


def test_tile_level_comes_from_the_config(tmp_path: Path) -> None:
    baked = Baked(
        build_pack_inputs(tmp_path / "data", fixture_city(tile_level=3)), tmp_path / "out"
    )
    assert baked.reader.counts["buildingTileLevel"] == 3


def test_a_pack_without_detail_removes_a_stale_detail_file(tmp_path: Path) -> None:
    city = fixture_city(min_footprint_m2=1.0)
    inputs = build_pack_inputs(tmp_path / "data", city)
    stage = run_pack(*inputs.args())
    pack_dir = tmp_path / "pack"
    pack_dir.mkdir()
    (pack_dir / "buildings_detail.bin").write_bytes(b"stale")
    write_pack_stage(stage, pack_dir, tmp_path / "stage")
    assert not (pack_dir / "buildings_detail.bin").exists()
    reader = PackReader(pack_dir)
    assert reader.counts["detailBuildings"] == 0
    reader.verify()


def test_a_pack_with_detail_overwrites_a_stale_detail_file(
    tmp_path: Path, inputs: PackInputs
) -> None:
    pack_dir = tmp_path / "pack"
    pack_dir.mkdir()
    (pack_dir / "buildings_detail.bin").write_bytes(b"stale")
    write_pack_stage(run_pack(*inputs.args()), pack_dir, tmp_path / "stage")
    PackReader(pack_dir).verify()


def test_write_pack_stage_leaves_unrelated_files_in_the_pack_folder(
    tmp_path: Path, inputs: PackInputs
) -> None:
    pack_dir = tmp_path / "pack"
    pack_dir.mkdir()
    (pack_dir / "notes.txt").write_text("keep", encoding="utf-8")
    write_pack_stage(run_pack(*inputs.args()), pack_dir, tmp_path / "stage")
    assert (pack_dir / "notes.txt").read_text(encoding="utf-8") == "keep"


STALE_CASES = [
    ("personas", "population_residents_sha256"),
    ("graph", "population_residents_sha256"),
    ("graph", "persona_assignment_sha256"),
    ("graph", "buildings_sha256"),
    ("stores", "population_residents_sha256"),
    ("stores", "buildings_sha256"),
    ("stores", "osm_road_nodes_sha256"),
    ("stores", "osm_road_edges_sha256"),
]


@pytest.mark.parametrize(("stage_name", "key"), STALE_CASES)
def test_a_recorded_digest_that_differs_raises_stale_input_naming_the_stage(
    inputs: PackInputs, stage_name: str, key: str
) -> None:
    rewrite_meta(getattr(inputs, stage_name), **{key: "0" * 64})
    with pytest.raises(StaleInputError, match=stage_name):
        run_pack(*inputs.args())


def test_stale_input_is_a_pack_stage_error() -> None:
    assert issubclass(StaleInputError, PackStageError)
    assert issubclass(PackBudgetError, PackStageError)


def test_changed_residents_make_the_persona_layer_stale(inputs: PackInputs) -> None:
    residents = read_table(inputs.population / "residents")
    residents["weight"] = residents["weight"] * 2
    write_table(inputs.population / "residents", residents)
    with pytest.raises(StaleInputError):
        run_pack(*inputs.args())


def test_resident_hex_with_another_row_count_than_the_residents_is_refused(
    inputs: PackInputs,
) -> None:
    short = read_table(inputs.stores / "resident_hex")["hex"][:-1]
    write_table(inputs.stores / "resident_hex", {"hex": short})
    with pytest.raises(PackStageError):
        run_pack(*inputs.args())


def test_an_assignment_with_another_row_count_than_the_residents_is_refused(
    inputs: PackInputs,
) -> None:
    assignment = {k: v[:-1] for k, v in read_table(inputs.personas / "assignment").items()}
    write_table(inputs.personas / "assignment", assignment)
    rewrite_meta(
        inputs.graph, persona_assignment_sha256=table_digest(inputs.personas / "assignment")
    )
    with pytest.raises(PackStageError):
        run_pack(*inputs.args())


def test_a_first_load_over_budget_raises_naming_both_numbers(
    inputs: PackInputs, baked: Baked, monkeypatch: pytest.MonkeyPatch
) -> None:
    used = baked.reader.manifest["sizes"]["first_load"]["bytes"]
    budget = used - 1
    patch_budget(monkeypatch, "FIRST_LOAD_BUDGET", budget)
    with pytest.raises(PackBudgetError) as caught:
        run_pack(*inputs.args())
    assert caught.value.report["within_budget"]["first_load"] is False
    message = str(caught.value)
    assert str(used) in message or f"{used:,}" in message
    assert str(budget) in message or f"{budget:,}" in message


def test_an_engine_over_its_soft_budget_is_reported_not_raised(
    inputs: PackInputs, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    patch_budget(monkeypatch, "ENGINE_BUDGET", 1)
    stage = run_pack(*inputs.args())
    assert stage.report["within_budget"] == {"first_load": True, "engine": False}
    pack_dir, _ = write_stage(stage, tmp_path / "out")
    PackReader(pack_dir).verify()


def test_a_first_load_exactly_at_its_budget_does_not_raise(
    inputs: PackInputs, baked: Baked, monkeypatch: pytest.MonkeyPatch
) -> None:
    used = baked.reader.manifest["sizes"]["first_load"]["bytes"]
    patch_budget(monkeypatch, "FIRST_LOAD_BUDGET", used)
    stage = run_pack(*inputs.args())
    assert stage.report["within_budget"]["first_load"] is True


def test_the_stage_does_not_modify_its_inputs(inputs: PackInputs, tmp_path: Path) -> None:
    folders = [
        inputs.buildings_layers,
        inputs.osm_layers,
        inputs.population,
        inputs.personas,
        inputs.graph,
        inputs.stores,
    ]
    before = [snapshot(f) for f in folders]
    write_stage(run_pack(*inputs.args(wards=True)), tmp_path / "out")
    assert [snapshot(f) for f in folders] == before


def test_the_arterial_classes_come_from_the_config(tmp_path: Path) -> None:
    city = fixture_city(arterial_classes=("major",))
    baked = Baked(build_pack_inputs(tmp_path / "data", city), tmp_path / "out")
    assert baked.report["roads"] == {"count": 6, "arterial": 1, "bridge": 2, "street": 3}
    assert baked.reader.roads().road_class.tolist() == [1, 0, 2, 0, 0, 2]
