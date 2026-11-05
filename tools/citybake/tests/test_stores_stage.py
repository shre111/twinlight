import json
import math
from fractions import Fraction
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import h3
import numpy as np
import pytest
import zstandard
from scipy.spatial import KDTree
from stores_fixture import (
    BANDS_SHUFFLED,
    BANDS_SORTED,
    DISTRICT_NAMES,
    SPEEDS,
    Inputs,
    StubBackend,
    city_with,
    grid_inputs,
    heavy_inputs,
    line_inputs,
    projection,
    snapshot,
    travel_config,
)

import twl_citybake.stores.stage as stage_module
from twl_citybake.config import CityConfig, TravelBand
from twl_citybake.store import read_table, table_digest
from twl_citybake.stores.check_pairs import Places, draw_check_pairs
from twl_citybake.stores.stage import run_stores, write_store_layers

GRID_RADIUS = 700.0
GRID_SPEEDS = {**SPEEDS, "local": 17.0}
BAND_FACTORS = {b.name: b.factor for b in BANDS_SHUFFLED}
BAND_STARTS = {b.name: b.start_minute for b in BANDS_SHUFFLED}
ISLAND_AND_SPUR = {15, 16, 17}


def grid_city(**changes: Any) -> CityConfig:
    return city_with(GRID_RADIUS, travel=travel_config(speed_kmh=GRID_SPEEDS), **changes)


def bake(inputs: Inputs, out: Path, backend: Any = None) -> Path:
    extra = {} if backend is None else {"backend": backend}
    layers = run_stores(
        inputs.population, inputs.osm_layers, inputs.buildings_layers, inputs.city, **extra
    )
    write_store_layers(layers, out)
    return out


def report_of(out: Path) -> dict[str, Any]:
    return json.loads((out / "report.json").read_text())


def layer_of(out: Path) -> dict[str, Any]:
    return json.loads((out / "layer.json").read_text())


class Baked:
    def __init__(self, inputs: Inputs, out: Path) -> None:
        self.inputs, self.out = inputs, out
        self.stores = read_table(out / "stores")
        self.hexes = read_table(out / "hexes")
        self.resident_hex = read_table(out / "resident_hex")["hex"]
        self.minutes = read_table(out / "travel")["minutes"]
        self.report, self.layer = report_of(out), layer_of(out)
        self.residents = read_table(inputs.population / "residents")
        self.roads = read_table(inputs.osm_layers / "road_nodes")
        self.pairs = json.loads((out / "check_pairs.json").read_text())

    @property
    def cube(self) -> np.ndarray:
        s, h, b = self.layer["travel_shape"]
        return self.minutes.reshape(s, h, b)

    def pair_distance_m(self) -> np.ndarray:
        sx, sy = self.stores["x"].astype(np.float64), self.stores["y"].astype(np.float64)
        hx, hy = self.hexes["x"].astype(np.float64), self.hexes["y"].astype(np.float64)
        return np.hypot(sx[:, None] - hx[None, :], sy[:, None] - hy[None, :])


@pytest.fixture(scope="module")
def grid(tmp_path_factory: pytest.TempPathFactory) -> Baked:
    root = tmp_path_factory.mktemp("stores-grid")
    inputs = grid_inputs(root, grid_city())
    return Baked(inputs, bake(inputs, root / "out"))


def test_stage_writes_every_documented_artifact(grid: Baked) -> None:
    names = {p.name for p in grid.out.iterdir()}
    assert {
        "stores",
        "hexes",
        "resident_hex",
        "travel",
        "bands.json",
        "check_pairs.json",
        "report.json",
        "layer.json",
    } <= names


def test_table_columns_have_the_documented_dtypes(grid: Baked) -> None:
    assert {k: str(v.dtype) for k, v in grid.stores.items()} == {
        "node": "uint32",
        "x": "float32",
        "y": "float32",
        "lon": "float64",
        "lat": "float64",
        "hex": "uint32",
        "radius_m": "float32",
        "district": "uint16",
    }
    assert {k: str(v.dtype) for k, v in grid.hexes.items()} == {
        "h3": "uint64",
        "x": "float32",
        "y": "float32",
        "node": "uint32",
        "node_distance_m": "float32",
    }
    assert grid.resident_hex.dtype == np.uint32
    assert len(grid.resident_hex) == grid.inputs.residents
    assert grid.minutes.dtype == np.uint16 and grid.minutes.ndim == 1


def test_travel_table_is_complete_with_rows_equal_to_stores_times_hexes_times_bands(
    grid: Baked,
) -> None:
    stores, hexes = len(grid.stores["node"]), len(grid.hexes["h3"])
    assert len({stores, hexes, len(BANDS_SORTED)}) == 3 and stores >= 2
    assert grid.layer["travel_shape"] == [stores, hexes, len(BANDS_SORTED)]
    assert grid.layer["travel_order"] == ["store", "hex", "band"]
    assert len(grid.minutes) == stores * hexes * len(BANDS_SORTED)
    meta = json.loads((grid.out / "travel" / "_table.json").read_text())
    assert meta["rows"] == len(grid.minutes)
    assert grid.minutes.min() >= 1


def test_store_rows_sit_on_their_road_node_with_the_configured_radius(grid: Baked) -> None:
    node = grid.stores["node"].astype(np.int64)
    assert len(set(node.tolist())) == len(node)
    assert not set(node.tolist()) & ISLAND_AND_SPUR
    np.testing.assert_allclose(grid.stores["x"], grid.roads["x"][node], atol=0.01)
    np.testing.assert_allclose(grid.stores["y"], grid.roads["y"][node], atol=0.01)
    lon, lat = projection(grid.inputs.city).inverse(grid.roads["x"][node], grid.roads["y"][node])
    np.testing.assert_allclose(grid.stores["lon"], lon, atol=1e-6)
    np.testing.assert_allclose(grid.stores["lat"], lat, atol=1e-6)
    assert (grid.stores["radius_m"] == GRID_RADIUS).all()


def test_store_hex_is_the_dense_id_of_the_cell_holding_the_store(grid: Baked) -> None:
    for lon, lat, hex_id in zip(
        grid.stores["lon"], grid.stores["lat"], grid.stores["hex"], strict=True
    ):
        want = h3.str_to_int(h3.latlng_to_cell(float(lat), float(lon), 9))
        assert int(grid.hexes["h3"][hex_id]) == want


def test_store_district_is_the_district_of_the_nearest_building(grid: Baked) -> None:
    buildings = read_table(grid.inputs.osm_layers / "buildings")
    tree = KDTree(np.column_stack([buildings["x"], buildings["y"]]))
    _, nearest = tree.query(np.column_stack([grid.stores["x"], grid.stores["y"]]))
    np.testing.assert_array_equal(grid.stores["district"], buildings["district"][nearest])


def test_hexes_are_sorted_dense_and_cover_every_resident_cell(grid: Baked) -> None:
    ids = grid.hexes["h3"].astype(np.int64)
    assert (np.diff(ids) > 0).all()
    assert len(ids) == 15
    proj = projection(grid.inputs.city)
    lon, lat = proj.inverse(grid.residents["x"], grid.residents["y"])
    for i in range(0, len(lon), 3):
        cell = h3.str_to_int(h3.latlng_to_cell(float(lat[i]), float(lon[i]), 9))
        assert int(ids[grid.resident_hex[i]]) == cell
    assert set(grid.resident_hex.tolist()) == set(range(len(ids)))


def test_each_hex_snaps_to_its_own_grid_node_never_the_island_or_the_dead_end_spur(
    grid: Baked,
) -> None:
    first_resident = {int(h): i for i, h in reversed(list(enumerate(grid.resident_hex)))}
    nodes = grid.hexes["node"].astype(np.int64)
    assert not set(nodes.tolist()) & ISLAND_AND_SPUR
    for h, node in enumerate(nodes):
        i = first_resident[h]
        assert grid.roads["x"][node] == pytest.approx(float(grid.residents["x"][i]), abs=0.01)
        assert grid.roads["y"][node] == pytest.approx(float(grid.residents["y"][i]), abs=0.01)
    d = np.hypot(grid.hexes["x"] - grid.roads["x"][nodes], grid.hexes["y"] - grid.roads["y"][nodes])
    np.testing.assert_allclose(grid.hexes["node_distance_m"], d, atol=0.05)


def manhattan_minutes(grid: Baked) -> np.ndarray:
    sn = grid.stores["node"].astype(np.int64)
    hn = grid.hexes["node"].astype(np.int64)
    out = np.zeros((len(sn), len(hn), len(BANDS_SORTED)), np.int64)
    metres_per_minute = Fraction(17000, 60)
    for s, a in enumerate(sn):
        for h, b in enumerate(hn):
            walk_m = abs(grid.roads["x"][a] - grid.roads["x"][b]) + abs(
                grid.roads["y"][a] - grid.roads["y"][b]
            )
            blocks = round(float(walk_m) / 600)
            for k, name in enumerate(BANDS_SORTED):
                exact = Fraction(600 * blocks) / (metres_per_minute * Fraction(BAND_FACTORS[name]))
                out[s, h, k] = max(1, math.ceil(exact))
    return out


def test_travel_minutes_equal_hand_computed_grid_distances_rounded_up(grid: Baked) -> None:
    want = manhattan_minutes(grid)
    np.testing.assert_array_equal(grid.cube, want)
    assert (want == 1).any() and want.max() >= 10


def test_bands_json_is_sorted_by_start_whatever_the_config_order(grid: Baked) -> None:
    bands = json.loads((grid.out / "bands.json").read_text())
    assert bands == [
        {"name": n, "start_minute": BAND_STARTS[n], "factor": BAND_FACTORS[n]} for n in BANDS_SORTED
    ]


def layout_value(sources: np.ndarray, targets: np.ndarray, band: int) -> np.ndarray:
    return sources[:, None] * 1000 + targets[None, :] * 10 + band + 1.0


def test_flat_travel_table_is_in_store_then_hex_then_band_order(tmp_path: Path) -> None:
    inputs = grid_inputs(tmp_path, grid_city())
    backend = StubBackend(layout_value)
    baked = Baked(inputs, bake(inputs, tmp_path / "out", backend))
    sn = baked.stores["node"].astype(np.int64)
    hn = baked.hexes["node"].astype(np.int64)
    stores, hexes, bands = len(sn), len(hn), len(BANDS_SORTED)
    assert len({stores, hexes, bands}) == 3
    want = np.stack([layout_value(sn, hn, b) for b in range(bands)], axis=-1).astype(np.uint16)
    assert baked.layer["travel_shape"] == [stores, hexes, bands]
    np.testing.assert_array_equal(baked.minutes, want.reshape(-1))
    assert len(np.unique(baked.minutes)) == stores * hexes * bands
    assert baked.layer["backend"] == "stub"
    assert baked.report["travel"]["backend"] == "stub"
    assert baked.minutes[1] == want[0, 0, 1]
    assert baked.minutes[bands] == want[0, 1, 0]
    assert baked.minutes[hexes * bands] == want[1, 0, 0]


def rounding_value(sources: np.ndarray, targets: np.ndarray, band: int) -> np.ndarray:
    shape = (len(sources), len(targets))
    odd = (sources[:, None] + targets[None, :]) % 2 == 1
    if band == 0:
        return np.where(odd, 0.2, 0.0)
    if band == 1:
        return np.zeros(shape) + (1.0 + sources[:, None] % 5)
    if band == 2:
        return np.full(shape, 3.0000001)
    return np.full(shape, 2.2)


def test_minutes_round_up_to_whole_minutes_with_a_floor_of_one(tmp_path: Path) -> None:
    inputs = grid_inputs(tmp_path, grid_city())
    baked = Baked(inputs, bake(inputs, tmp_path / "out", StubBackend(rounding_value)))
    sn = baked.stores["node"].astype(np.int64)
    cube = baked.cube
    assert (cube[:, :, 0] == 1).all()
    np.testing.assert_array_equal(cube[:, :, 1], np.repeat((1 + sn % 5)[:, None], cube.shape[1], 1))
    assert (cube[:, :, 2] == 4).all()
    assert (cube[:, :, 3] == 3).all()


def test_float_noise_on_a_whole_minute_does_not_add_a_minute(tmp_path: Path) -> None:
    inputs = grid_inputs(tmp_path, grid_city())
    backend = StubBackend(lambda s, t, b: np.full((len(s), len(t)), 5.000000000000001))
    baked = Baked(inputs, bake(inputs, tmp_path / "out", backend))
    assert (baked.minutes == 5).all()


@pytest.mark.parametrize(("value", "stored"), [(65535.0, 65535), (65534.2, 65535)])
def test_a_value_that_fits_in_u16_is_stored_as_is(
    tmp_path: Path, value: float, stored: int
) -> None:
    inputs = grid_inputs(tmp_path, grid_city())
    backend = StubBackend(lambda s, t, b: np.full((len(s), len(t)), value))
    baked = Baked(inputs, bake(inputs, tmp_path / "out", backend))
    assert (baked.minutes == stored).all()


@pytest.mark.parametrize("value", [65535.5, 65536.0, 70000.0, 1e9])
def test_a_value_that_does_not_fit_in_u16_raises_instead_of_wrapping(
    tmp_path: Path, value: float
) -> None:
    inputs = grid_inputs(tmp_path, grid_city())
    backend = StubBackend(lambda s, t, b: np.full((len(s), len(t)), value))
    with pytest.raises((ValueError, OverflowError)):
        bake(inputs, tmp_path / "out", backend)


@pytest.mark.parametrize("bad", [math.inf, math.nan])
def test_one_unreachable_pair_raises_the_named_failure_naming_the_pair(
    tmp_path: Path, bad: float
) -> None:
    inputs = grid_inputs(tmp_path, grid_city())

    def one_bad(s: np.ndarray, t: np.ndarray, band: int) -> np.ndarray:
        out = layout_value(s, t, band)
        if band == 2:
            out[-1, -1] = bad
        return out

    with pytest.raises(stage_module.UnreachablePairError) as caught:
        bake(inputs, tmp_path / "out", StubBackend(one_bad))
    message = str(caught.value).lower()
    assert "store" in message and "hex" in message


def test_hexes_report_counts_and_distance_quantiles(grid: Baked) -> None:
    hexes = grid.report["hexes"]
    per_hex = np.bincount(grid.resident_hex)
    assert hexes["count"] == len(grid.hexes["h3"]) == 15
    assert hexes["residents_per_hex_p50"] == pytest.approx(float(np.median(per_hex)), abs=1e-3)
    d = grid.hexes["node_distance_m"].astype(np.float64)
    quant = hexes["node_distance_m"]
    assert quant["p50"] == pytest.approx(float(np.median(d)), abs=2e-3)
    assert quant["p90"] == pytest.approx(float(np.percentile(d, 90)), abs=2e-3)
    assert quant["max"] == pytest.approx(float(d.max()), abs=2e-3)


def cumulative_coverage(grid: Baked) -> np.ndarray:
    res = grid.residents
    covered = np.zeros(len(res["x"]), bool)
    weight = res["weight"].astype(np.float64)
    out = []
    for sx, sy in zip(grid.stores["x"], grid.stores["y"], strict=True):
        covered |= np.hypot(res["x"] - sx, res["y"] - sy) <= GRID_RADIUS
        out.append(weight[covered].sum() / weight.sum())
    return np.array(out)


def test_stores_report_counts_coverage_by_weight_and_district_and_bank(grid: Baked) -> None:
    stores = grid.report["stores"]
    assert stores["count"] == len(grid.stores["node"])
    assert stores["radius_m"] == GRID_RADIUS
    want = cumulative_coverage(grid)
    np.testing.assert_allclose(stores["coverage_by_store"], want, atol=1e-3)
    assert stores["coverage"] == pytest.approx(float(want[-1]), abs=1e-3)
    assert stores["coverage"] == stores["coverage_by_store"][-1]
    assert (np.diff(stores["coverage_by_store"]) > 0).all()
    buildings = read_table(grid.inputs.osm_layers / "buildings")
    tree = KDTree(np.column_stack([buildings["x"], buildings["y"]]))
    _, nearest = tree.query(np.column_stack([grid.stores["x"], grid.stores["y"]]))
    districts = buildings["district"][nearest]
    assert stores["by_district"] == {
        name: int((districts == i).sum()) for i, name in enumerate(DISTRICT_NAMES)
    }
    banks = buildings["bank"][nearest]
    assert stores["by_bank"] == {"west": int((banks == 0).sum()), "east": int((banks == 1).sum())}
    assert sum(stores["by_bank"].values()) == stores["count"]


def test_graph_section_reports_node_count_and_strong_component_share(grid: Baked) -> None:
    assert grid.report["graph"]["nodes"] == 18
    assert grid.report["graph"]["main_component_share"] == pytest.approx(15 / 18, abs=1e-3)


def inside_radius(grid: Baked) -> np.ndarray:
    distance = grid.pair_distance_m()
    assert np.abs(distance - GRID_RADIUS).min() > 2.0
    return distance <= GRID_RADIUS


def test_travel_report_bands_hold_quantiles_overall_and_within_the_radius(grid: Baked) -> None:
    travel = grid.report["travel"]
    assert travel["backend"] == "graph-dijkstra"
    assert travel["complete"] is True
    assert set(travel["bands"]) == set(BANDS_SORTED)
    inside = inside_radius(grid)
    assert 0 < inside.sum() < inside.size
    for k, name in enumerate(BANDS_SORTED):
        band = travel["bands"][name]
        assert set(band) == {"p50", "p90", "max", "within_radius_p50", "within_radius_p90"}
        every, near = grid.cube[:, :, k].reshape(-1), grid.cube[:, :, k][inside]
        assert band["max"] == pytest.approx(float(every.max()), abs=1e-3)
        assert band["p50"] == pytest.approx(float(np.percentile(every, 50)), abs=1e-3)
        assert band["p90"] == pytest.approx(float(np.percentile(every, 90)), abs=1e-3)
        assert band["within_radius_p50"] == pytest.approx(float(np.percentile(near, 50)), abs=1e-3)
        assert band["within_radius_p90"] == pytest.approx(float(np.percentile(near, 90)), abs=1e-3)
        assert band["within_radius_p90"] < band["max"]
        assert band["within_radius_p50"] < band["p90"]


def test_in_radius_share_and_table_sizes_are_reported(grid: Baked) -> None:
    travel = grid.report["travel"]
    inside = inside_radius(grid)
    assert travel["table_in_radius_share"] == pytest.approx(float(inside.mean()), abs=1e-3)
    assert travel["table_bytes"] == len(grid.minutes) * 2
    packed = zstandard.ZstdCompressor(level=19).compress(grid.minutes.astype("<u2").tobytes())
    assert travel["table_zstd_bytes"] == len(packed)
    assert travel["speed_kmh"] == GRID_SPEEDS


def test_within_targets_are_all_true_when_the_target_is_met(grid: Baked) -> None:
    assert grid.report["stores"]["coverage"] == 1.0
    assert grid.report["within_targets"] == {"coverage": True, "travel_complete": True, "all": True}


def floats_in(value: Any) -> list[float]:
    if isinstance(value, dict):
        return [f for v in value.values() for f in floats_in(v)]
    if isinstance(value, list):
        return [f for v in value for f in floats_in(v)]
    return [value] if isinstance(value, float) else []


def test_every_float_in_the_report_is_rounded_to_three_decimals(grid: Baked) -> None:
    values = floats_in(grid.report)
    assert len(values) > 20
    assert all(v == round(v, 3) for v in values)


def test_a_coverage_miss_is_reported_not_raised_and_coverage_is_by_resident_weight(
    tmp_path: Path,
) -> None:
    inputs = heavy_inputs(tmp_path, city_with(1000.0, coverage_target=0.92, max_stores=1))
    baked = Baked(inputs, bake(inputs, tmp_path / "out"))
    assert baked.stores["x"].tolist() == [3000.0]
    assert baked.report["stores"]["coverage"] == 0.75
    assert baked.report["stores"]["coverage_by_store"] == [0.75]
    assert baked.report["within_targets"] == {
        "coverage": False,
        "travel_complete": True,
        "all": False,
    }


def test_the_same_world_reaches_the_target_once_a_second_store_is_allowed(tmp_path: Path) -> None:
    inputs = heavy_inputs(tmp_path, city_with(1000.0, coverage_target=0.92, max_stores=2))
    baked = Baked(inputs, bake(inputs, tmp_path / "out"))
    assert baked.stores["x"].tolist() == [3000.0, 0.0]
    assert baked.report["stores"]["coverage_by_store"] == [0.75, 1.0]
    assert baked.report["within_targets"] == {
        "coverage": True,
        "travel_complete": True,
        "all": True,
    }


@pytest.fixture(scope="module")
def line(tmp_path_factory: pytest.TempPathFactory) -> Baked:
    root = tmp_path_factory.mktemp("stores-line")
    inputs = line_inputs(root, city_with(1500.0))
    return Baked(inputs, bake(inputs, root / "out"))


def test_line_world_minutes_are_rounded_up_per_band_from_a_hand_computed_route(
    line: Baked,
) -> None:
    far, near = int(line.resident_hex[-1]), int(line.resident_hex[0])
    assert line.stores["node"].tolist() == [0]
    row = line.cube[0]
    assert row[far].tolist() == [6, 5, 7, 4]
    assert row[near].tolist() == [1, 1, 1, 1]
    assert row[int(line.stores["hex"][0])].tolist() == [1, 1, 1, 1]


def test_a_store_in_a_cell_without_residents_adds_a_hex_but_no_check_pair(line: Baked) -> None:
    store_hex = int(line.stores["hex"][0])
    resident_cells = set(line.resident_hex.tolist())
    assert store_hex not in resident_cells
    assert len(line.hexes["h3"]) == 3
    assert line.report["hexes"]["count"] == 3
    assert {p["hex"] for p in line.pairs} == resident_cells
    assert len(line.pairs) == 2
    assert sorted(int(p["distance_km"]) for p in line.pairs) == [0, 1]


def test_within_radius_minutes_leave_out_a_hex_that_only_holds_a_store(line: Baked) -> None:
    riders_see = sorted(set(line.resident_hex.tolist()))
    assert int(line.stores["hex"][0]) not in riders_see
    assert (line.pair_distance_m()[0] <= 1500.0).all()
    assert sorted(line.cube[0, riders_see, BANDS_SORTED.index("midday")].tolist()) == [1, 5]
    midday = line.report["travel"]["bands"]["midday"]
    assert midday["within_radius_p50"] == 3.0
    assert midday["within_radius_p90"] == pytest.approx(4.6)


def test_residents_per_hex_leaves_out_a_hex_that_only_holds_a_store(line: Baked) -> None:
    assert sorted(np.bincount(line.resident_hex, minlength=3).tolist()) == [0, 1, 3]
    assert line.report["hexes"]["residents_per_hex_p50"] == 2.0


def test_in_radius_share_counts_every_pair_including_a_store_only_hex(line: Baked) -> None:
    assert (line.pair_distance_m()[0] <= 1500.0).all()
    assert line.report["travel"]["table_in_radius_share"] == 1.0


def test_store_coordinates_are_a_real_latitude_and_longitude(grid: Baked) -> None:
    bbox = grid.inputs.city.bbox
    assert ((grid.stores["lat"] > bbox.south) & (grid.stores["lat"] < bbox.north)).all()
    assert ((grid.stores["lon"] > bbox.west) & (grid.stores["lon"] < bbox.east)).all()
    x, y = projection(grid.inputs.city).forward(grid.stores["lon"], grid.stores["lat"])
    np.testing.assert_allclose(x, grid.stores["x"], atol=1e-2)
    np.testing.assert_allclose(y, grid.stores["y"], atol=1e-2)


def pair_strata(pairs: list[dict[str, Any]]) -> list[int]:
    return [int(min(p["distance_km"], 2.999)) for p in pairs]


def test_check_pairs_are_stratified_by_distance_ten_each_or_all_that_exist(grid: Baked) -> None:
    resident_hexes = sorted(set(grid.resident_hex.tolist()))
    distance = grid.pair_distance_m() / 1000
    assert (
        np.abs(distance[:, resident_hexes] - np.array([[1.0], [2.0], [3.0]])[:, None]).min() > 0.005
    )
    available = [
        int(((distance[:, resident_hexes] >= lo) & (distance[:, resident_hexes] < lo + 1)).sum())
        for lo in (0, 1, 2)
    ]
    assert max(available) > 10
    found = [pair_strata(grid.pairs).count(k) for k in range(3)]
    assert found == [min(10, n) for n in available]
    assert len(grid.pairs) == sum(found)


def far_stratum_pairs(near_hexes: int) -> list[dict[str, Any]]:
    distance = np.concatenate(
        [np.full(50, 2500.0), np.full(50, 1500.0), np.full(near_hexes, 500.0)]
    )[None, :]
    zeros = np.zeros(distance.shape[1])
    pairs = draw_check_pairs(
        distance,
        np.ones_like(distance, bool),
        np.ones((*distance.shape, 1), np.uint16),
        [TravelBand(name="midday", start_minute=0, factor=1.0)],
        Places(np.zeros(1), np.zeros(1), zeros, zeros),
        np.random.default_rng(3),
    )
    return [p for p in pairs if p["distance_km"] >= 1]


def test_the_size_of_one_stratum_does_not_move_the_draws_of_another() -> None:
    assert far_stratum_pairs(3) == far_stratum_pairs(40)


def test_check_pairs_have_working_fields_and_use_only_hexes_holding_residents(grid: Baked) -> None:
    resident_cells = set(grid.resident_hex.tolist())
    seen = set()
    assert len(grid.pairs) > 0
    for p in grid.pairs:
        assert set(p) >= {
            "store",
            "hex",
            "store_lat",
            "store_lon",
            "hex_lat",
            "hex_lon",
            "distance_km",
            "minutes",
            "maps_url",
        }
        s, h = p["store"], p["hex"]
        seen.add((s, h))
        assert h in resident_cells
        assert p["store_lat"] == pytest.approx(float(grid.stores["lat"][s]), abs=1e-5)
        assert p["store_lon"] == pytest.approx(float(grid.stores["lon"][s]), abs=1e-5)
        hlat, hlon = h3.cell_to_latlng(h3.int_to_str(int(grid.hexes["h3"][h])))
        assert p["hex_lat"] == pytest.approx(hlat, abs=1e-5)
        assert p["hex_lon"] == pytest.approx(hlon, abs=1e-5)
        assert p["distance_km"] == pytest.approx(grid.pair_distance_m()[s, h] / 1000, rel=0.01)
        assert 0 <= p["distance_km"] <= 3
        assert p["minutes"] == {
            name: int(grid.cube[s, h, k]) for k, name in enumerate(BANDS_SORTED)
        }
    assert len(seen) == len(grid.pairs)


def test_check_pair_links_open_a_two_wheeler_route_from_the_store_to_the_hex(grid: Baked) -> None:
    for p in grid.pairs:
        url = urlparse(p["maps_url"])
        query = parse_qs(url.query)
        assert (url.scheme, url.netloc, url.path) == ("https", "www.google.com", "/maps/dir/")
        assert set(query) == {"api", "origin", "destination", "travelmode"}
        assert query["api"] == ["1"] and query["travelmode"] == ["two-wheeler"]
        origin = [float(v) for v in query["origin"][0].split(",")]
        destination = [float(v) for v in query["destination"][0].split(",")]
        assert origin == pytest.approx([p["store_lat"], p["store_lon"]], abs=1e-5)
        assert destination == pytest.approx([p["hex_lat"], p["hex_lon"]], abs=1e-5)


def test_check_pairs_depend_on_the_seed_alone_not_on_the_run(grid: Baked, tmp_path: Path) -> None:
    again = bake(grid.inputs, tmp_path / "again")
    assert (again / "check_pairs.json").read_bytes() == (grid.out / "check_pairs.json").read_bytes()
    other_inputs = grid_inputs(tmp_path / "other", grid_city(seed=99))
    other = Baked(other_inputs, bake(other_inputs, tmp_path / "other" / "out"))
    keys = {(p["store"], p["hex"]) for p in grid.pairs}
    assert keys != {(p["store"], p["hex"]) for p in other.pairs}
    assert other.layer["seed"] == 99
    assert other.report["stores"] == grid.report["stores"]
    np.testing.assert_array_equal(other.stores["node"], grid.stores["node"])
    np.testing.assert_array_equal(other.minutes, grid.minutes)


def test_reruns_are_byte_identical(grid: Baked, tmp_path: Path) -> None:
    again = bake(grid.inputs, tmp_path / "again")
    assert snapshot(again) == snapshot(grid.out)


def test_layer_json_records_provenance_and_a_digest_of_every_input(grid: Baked) -> None:
    layer, inputs = grid.layer, grid.inputs
    assert layer["city"] == "ahmedabad"
    assert layer["seed"] == inputs.city.seed
    assert layer["backend"] == "graph-dijkstra"
    assert layer["h3_resolution"] == 9
    assert layer["population_residents_sha256"] == table_digest(inputs.population / "residents")
    assert layer["osm_road_nodes_sha256"] == table_digest(inputs.osm_layers / "road_nodes")
    assert layer["osm_road_edges_sha256"] == table_digest(inputs.osm_layers / "road_edges")
    assert layer["buildings_sha256"] == table_digest(inputs.buildings_layers / "buildings")
    assert layer["buildings_stage"] == "osm"
    assert layer["attribution"] == "Road network © OpenStreetMap contributors, ODbL 1.0"
    assert layer["stores_source"] == "greedy max coverage, synthetic"
    digests = [layer[k] for k in layer if k.endswith("_sha256")]
    assert len(set(digests)) == len(digests) == 4


def test_layer_json_records_the_config_that_shaped_the_stores_and_the_table(grid: Baked) -> None:
    stores, travel = grid.inputs.city.stores, grid.inputs.city.travel
    assert grid.layer["stores"] == {
        "radius_m": stores.radius_m,
        "coverage_target": stores.coverage_target,
        "max_stores": stores.max_stores,
        "candidate_cell_m": stores.candidate_cell_m,
    }
    assert grid.layer["travel"]["speed_kmh"] == GRID_SPEEDS
    assert grid.layer["travel"]["h3_resolution"] == travel.h3_resolution
    assert {b["name"] for b in grid.layer["travel"]["bands"]} == set(BANDS_SORTED)
