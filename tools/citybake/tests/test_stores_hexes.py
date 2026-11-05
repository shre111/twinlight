import h3
import numpy as np
import pytest
from stores_fixture import hex_centre_xy

from twl_citybake.config import load_city
from twl_citybake.projection import LocalProjection
from twl_citybake.stores.hexes import hex_nodes, resident_hexes

PROJ = LocalProjection.for_bbox(load_city("ahmedabad").bbox)
RES = 9


def cell_at(x: float, y: float, resolution: int = RES) -> str:
    lon, lat = PROJ.inverse(x, y)
    return h3.latlng_to_cell(float(lat), float(lon), resolution)


def centre(cell: str) -> tuple[float, float]:
    lat, lon = h3.cell_to_latlng(cell)
    x, y = PROJ.forward(lon, lat)
    return float(x), float(y)


def test_residents_in_one_cell_share_an_id_and_other_cells_get_other_ids() -> None:
    cx, cy = centre(cell_at(0.0, 0.0))
    x = np.array([cx, cx + 20.0, cx - 15.0, cx + 2000.0])
    y = np.array([cy, cy - 10.0, cy + 25.0, cy])
    index, ids = resident_hexes(x, y, PROJ, RES)
    assert index[0] == index[1] == index[2]
    assert index[3] != index[0]
    assert len(ids) == 2


def test_ids_are_the_h3_cells_of_the_residents_sorted_ascending_as_uint64() -> None:
    rng = np.random.default_rng(3)
    x = rng.uniform(-3000, 3000, 200)
    y = rng.uniform(-3000, 3000, 200)
    index, ids = resident_hexes(x, y, PROJ, RES)
    want = sorted({h3.str_to_int(cell_at(a, b)) for a, b in zip(x, y, strict=True)})
    assert ids.dtype == np.uint64 and index.dtype == np.uint32
    assert ids.tolist() == want
    assert (np.diff(ids.astype(np.int64)) > 0).all()
    assert len(index) == 200


def test_hex_index_is_dense_and_points_at_the_residents_own_cell() -> None:
    rng = np.random.default_rng(5)
    x = rng.uniform(-3000, 3000, 150)
    y = rng.uniform(-3000, 3000, 150)
    index, ids = resident_hexes(x, y, PROJ, RES)
    assert sorted(set(index.tolist())) == list(range(len(ids)))
    for i in range(0, 150, 7):
        assert int(ids[index[i]]) == h3.str_to_int(cell_at(x[i], y[i]))


def test_dense_numbers_follow_sorted_id_order_not_first_appearance() -> None:
    cells = sorted(
        {cell_at(a, 0.0) for a in (-4000.0, -1000.0, 0.0, 1500.0, 3500.0)}, key=h3.str_to_int
    )
    low, mid, high = (centre(c) for c in cells[:3])
    x = np.array([high[0], low[0], mid[0], high[0], low[0]])
    y = np.array([high[1], low[1], mid[1], high[1], low[1]])
    index, ids = resident_hexes(x, y, PROJ, RES)
    assert index.tolist() == [2, 0, 1, 2, 0]
    assert ids.tolist() == [h3.str_to_int(c) for c in cells[:3]]


def test_resolution_is_honoured_by_merging_children_of_one_parent() -> None:
    parent = cell_at(0.0, 0.0, 8)
    kids = sorted(h3.cell_to_children(parent, 9))[:2]
    x, y = zip(*(centre(k) for k in kids), strict=True)
    fine, fine_ids = resident_hexes(np.array(x), np.array(y), PROJ, 9)
    coarse, coarse_ids = resident_hexes(np.array(x), np.array(y), PROJ, 8)
    assert len(fine_ids) == 2 and fine[0] != fine[1]
    assert coarse_ids.tolist() == [h3.str_to_int(parent)]
    assert coarse.tolist() == [0, 0]


def test_the_projection_origin_lands_in_the_cell_holding_the_bbox_centre() -> None:
    index, ids = resident_hexes(np.array([0.0]), np.array([0.0]), PROJ, RES)
    lon0, lat0 = load_city("ahmedabad").bbox.centre
    assert int(ids[0]) == h3.str_to_int(h3.latlng_to_cell(lat0, lon0, RES))
    assert index.tolist() == [0]


def test_a_node_on_a_hex_centre_is_chosen_at_distance_zero() -> None:
    cx, cy = hex_centre_xy(PROJ, 0.0, 0.0)
    _, ids = resident_hexes(np.array([cx]), np.array([cy]), PROJ, RES)
    nodes, dist = hex_nodes(
        ids, PROJ, np.array([cx + 300.0, cx, cx - 80.0]), np.array([cy, cy, cy]), np.ones(3, bool)
    )
    assert nodes.dtype == np.uint32
    assert nodes.tolist() == [1]
    assert dist[0] == pytest.approx(0.0, abs=0.01)


def test_hex_nodes_pick_the_nearest_node_and_report_its_distance_in_metres() -> None:
    cx, cy = hex_centre_xy(PROJ, 0.0, 0.0)
    nx = np.array([cx + 300.0, cx + 30.0, cx - 60.0, cx])
    ny = np.array([cy, cy, cy - 5.0, cy + 90.0])
    _, ids = resident_hexes(np.array([cx]), np.array([cy]), PROJ, RES)
    nodes, dist = hex_nodes(ids, PROJ, nx, ny, np.ones(4, bool))
    assert nodes.tolist() == [1]
    assert dist[0] == pytest.approx(30.0, abs=0.01)


def test_nodes_outside_the_main_component_are_never_chosen_even_when_nearer() -> None:
    cx, cy = hex_centre_xy(PROJ, 0.0, 0.0)
    nx = np.array([cx + 200.0, cx + 5.0, cx - 40.0])
    ny = np.array([cy, cy, cy])
    _, ids = resident_hexes(np.array([cx]), np.array([cy]), PROJ, RES)
    nodes, dist = hex_nodes(ids, PROJ, nx, ny, np.array([True, False, True]))
    assert nodes.tolist() == [2]
    assert dist[0] == pytest.approx(40.0, abs=0.01)


def test_each_hex_gets_its_own_nearest_node_in_id_order() -> None:
    a = hex_centre_xy(PROJ, -2000.0, 0.0)
    b = hex_centre_xy(PROJ, 2000.0, 500.0)
    x = np.array([a[0], b[0]])
    y = np.array([a[1], b[1]])
    index, ids = resident_hexes(x, y, PROJ, RES)
    nx = np.array([a[0] + 50.0, b[0] - 70.0, 0.0])
    ny = np.array([a[1], b[1], 0.0])
    nodes, dist = hex_nodes(ids, PROJ, nx, ny, np.ones(3, bool))
    first, second = index.tolist()
    assert (nodes[first], nodes[second]) == (0, 1)
    assert dist[first] == pytest.approx(50.0, abs=0.01)
    assert dist[second] == pytest.approx(70.0, abs=0.01)
