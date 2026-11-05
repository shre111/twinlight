import numpy as np
from h3.api import basic_int as h3
from numpy.typing import NDArray
from scipy.spatial import KDTree

from twl_citybake.projection import LocalProjection


def cells_at(
    x: NDArray[np.floating], y: NDArray[np.floating], proj: LocalProjection, resolution: int
) -> NDArray[np.uint64]:
    """H3 cell of each point in the local projection."""
    lon, lat = proj.inverse(x, y)
    cells = (
        h3.latlng_to_cell(a, o, resolution) for a, o in zip(lat.tolist(), lon.tolist(), strict=True)
    )
    return np.fromiter(cells, np.uint64, count=len(lat))


def resident_hexes(
    res_x: NDArray[np.floating],
    res_y: NDArray[np.floating],
    proj: LocalProjection,
    resolution: int,
) -> tuple[NDArray[np.uint32], NDArray[np.uint64]]:
    """Dense hex index of each resident, and the H3 ids those index into, sorted ascending."""
    h3_ids, index = np.unique(cells_at(res_x, res_y, proj, resolution), return_inverse=True)
    return index.astype(np.uint32), h3_ids


def hex_centres(
    h3_ids: NDArray[np.uint64], proj: LocalProjection
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    centres = [h3.cell_to_latlng(cell) for cell in h3_ids.tolist()]
    lat = np.fromiter((c[0] for c in centres), np.float64, count=len(centres))
    lon = np.fromiter((c[1] for c in centres), np.float64, count=len(centres))
    return proj.forward(lon, lat)


def hex_nodes(
    h3_ids: NDArray[np.uint64],
    proj: LocalProjection,
    node_x: NDArray[np.floating],
    node_y: NDArray[np.floating],
    in_component: NDArray[np.bool_],
) -> tuple[NDArray[np.uint32], NDArray[np.float64]]:
    """The main-component road node nearest each hex centre, and its distance in metres."""
    nodes = np.flatnonzero(in_component)
    xs, ys = np.asarray(node_x, np.float64)[nodes], np.asarray(node_y, np.float64)[nodes]
    x, y = hex_centres(h3_ids, proj)
    distance, nearest = KDTree(np.column_stack([xs, ys])).query(np.column_stack([x, y]))
    return nodes[nearest].astype(np.uint32), np.asarray(distance, np.float64)
