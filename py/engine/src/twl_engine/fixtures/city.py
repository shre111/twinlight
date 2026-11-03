import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from twl_engine.fixtures.geometry import (
    DISTRICT_EAST,
    DISTRICT_TABLE,
    OLD_CITY,
    FloatArray,
    arterial_lines,
    city_f,
    district_at,
    in_river,
    peak_at,
    river_half_width,
    river_x,
    smoothstep,
)
from twl_engine.fixtures.types import (
    ROAD_ARTERIAL,
    ROAD_BRIDGE,
    ROAD_STREET,
    Buildings,
    River,
    Roads,
)

IntArray = NDArray[np.int64]

MASK_CELL_M = 20.0
MASK_ORIGIN_M = 4800.0
MASK_CELLS = 480
ARTERIAL_MAX_CITY_F = 1.08
BLOCK_LIMIT_M = 4700.0
BRIDGE_Z = (-2700.0, -2050.0, -1250.0, -400.0, 350.0, 950.0, 1600.0, 2350.0)
RIVER_STEP_M = 30.0
RIVER_HALF_SAMPLES = 200


@dataclass(frozen=True, eq=False)
class CityLayout:
    buildings: Buildings
    roads: Roads
    river: River
    corners_x: FloatArray
    corners_y: FloatArray


@dataclass(frozen=True, eq=False)
class _Polylines:
    lengths: IntArray
    x: FloatArray
    z: FloatArray
    kind: int


@dataclass(frozen=True, eq=False)
class _Blocks:
    quad_x: FloatArray
    quad_z: FloatArray
    size: FloatArray
    city_f: FloatArray
    district: IntArray


@dataclass(frozen=True, eq=False)
class _DistrictGrid:
    streets: _Polylines
    corner_x: FloatArray
    corner_z: FloatArray
    blocks: _Blocks


def build_layout(rng: np.random.Generator) -> CityLayout:
    arterials, mask = _arterials()
    grids = [_district_grid(district, rng) for district in range(len(DISTRICT_TABLE))]
    buildings = _fill_blocks(_concat_blocks([g.blocks for g in grids]), mask, rng)
    roads = _roads([*(g.streets for g in grids), arterials, _bridges()])
    corners_x = np.concatenate([g.corner_x for g in grids])
    corners_z = np.concatenate([g.corner_z for g in grids])
    return CityLayout(buildings, roads, _river(), corners_x, -corners_z)


def _mask_cell(x: FloatArray, z: FloatArray) -> IntArray:
    gx = np.floor((x + MASK_ORIGIN_M) / MASK_CELL_M).astype(np.int64)
    gz = np.floor((z + MASK_ORIGIN_M) / MASK_CELL_M).astype(np.int64)
    inside = (gx >= 0) & (gz >= 0) & (gx < MASK_CELLS) & (gz < MASK_CELLS)
    return np.where(inside, gz * MASK_CELLS + gx, -1)


def _on_mask(mask: NDArray[np.bool_], x: FloatArray, z: FloatArray) -> NDArray[np.bool_]:
    cell = _mask_cell(x, z)
    return (cell >= 0) & mask[np.maximum(cell, 0)]


def _arterials() -> tuple[_Polylines, NDArray[np.bool_]]:
    """Arterials clipped to the city, as polylines, plus the 20 m cells within one cell of them."""
    runs_x: list[FloatArray] = []
    runs_z: list[FloatArray] = []
    for x, z in arterial_lines():
        keep = city_f((x[:-1] + x[1:]) / 2, (z[:-1] + z[1:]) / 2) <= ARTERIAL_MAX_CITY_F
        bounds = np.flatnonzero(np.diff(np.concatenate(([0], keep.astype(np.int8), [0]))))
        for start, stop in zip(bounds[::2], bounds[1::2], strict=True):
            runs_x.append(x[start : stop + 1])
            runs_z.append(z[start : stop + 1])
    lengths = np.array([len(run) for run in runs_x], dtype=np.int64)
    arterials = _Polylines(lengths, np.concatenate(runs_x), np.concatenate(runs_z), ROAD_ARTERIAL)
    return arterials, _arterial_mask(arterials)


def _arterial_mask(lines: _Polylines) -> NDArray[np.bool_]:
    has_next = np.ones(lines.x.size, dtype=bool)
    has_next[np.cumsum(lines.lengths) - 1] = False
    start = np.flatnonzero(has_next)
    x1, z1, x2, z2 = lines.x[start], lines.z[start], lines.x[start + 1], lines.z[start + 1]
    steps = np.maximum(np.ceil(np.hypot(x2 - x1, z2 - z1) / 10).astype(np.int64), 1)
    segment = np.repeat(np.arange(steps.size), steps + 1)
    step = np.arange(segment.size) - np.repeat(np.cumsum(steps + 1) - (steps + 1), steps + 1)
    t = step / steps[segment]
    px = x1[segment] + (x2 - x1)[segment] * t
    pz = z1[segment] + (z2 - z1)[segment] * t
    mask = np.zeros(MASK_CELLS * MASK_CELLS, dtype=bool)
    for a in (-1, 0, 1):
        for b in (-1, 0, 1):
            cell = _mask_cell(px + a * MASK_CELL_M, pz + b * MASK_CELL_M)
            mask[cell[cell >= 0]] = True
    return mask


def _district_grid(district: int, rng: np.random.Generator) -> _DistrictGrid:
    """A jittered, rotated street grid over the district's Voronoi cell (prototype `buildCity`)."""
    _, origin_x, origin_z = DISTRICT_TABLE[district]
    size = 98 + rng.random() * 8 if DISTRICT_EAST[district] else 110 + rng.random() * 14
    turn = (rng.random() - 0.5) * 0.55
    cos_t, sin_t = math.cos(turn), math.sin(turn)
    span = math.ceil(5000 / size)
    lattice = np.arange(-span, span + 1, dtype=np.float64)
    jitter = (rng.random((2, lattice.size, lattice.size)) - 0.5) * 0.22
    local_x = (lattice[:, None] + jitter[0]) * size
    local_z = (lattice[None, :] + jitter[1]) * size
    corner_x = origin_x + local_x * cos_t - local_z * sin_t
    corner_z = origin_z + local_x * sin_t + local_z * cos_t

    i, j = (g.ravel() for g in np.meshgrid(np.arange(2 * span), np.arange(2 * span), indexing="ij"))
    centre_lx = (i - span + 0.5) * size
    centre_lz = (j - span + 0.5) * size
    centre_x = origin_x + centre_lx * cos_t - centre_lz * sin_t
    centre_z = origin_z + centre_lx * sin_t + centre_lz * cos_t
    near = (np.abs(centre_x) <= BLOCK_LIMIT_M) & (np.abs(centre_z) <= BLOCK_LIMIT_M)
    i, j, centre_x, centre_z = i[near], j[near], centre_x[near], centre_z[near]
    f = city_f(centre_x, centre_z)
    own = (f <= 1) & (district_at(centre_x, centre_z) == district)
    i, j, f = i[own], j[own], f[own]

    quad_i = np.column_stack((i, i + 1, i + 1, i))
    quad_j = np.column_stack((j, j, j + 1, j + 1))
    quad_x, quad_z = corner_x[quad_i, quad_j], corner_z[quad_i, quad_j]
    dry = ~in_river(quad_x, quad_z, 18).any(axis=1)
    i, j, f, quad_x, quad_z = i[dry], j[dry], f[dry], quad_x[dry], quad_z[dry]

    built = rng.random(f.size) >= 0.045 + 0.3 * smoothstep(0.82, 1.0, f)
    blocks = _Blocks(
        quad_x[built],
        quad_z[built],
        np.full(int(built.sum()), size),
        f[built],
        np.full(int(built.sum()), district, dtype=np.int64),
    )
    streets = _block_edges(i, j, corner_x, corner_z)
    return _DistrictGrid(streets, quad_x[:, 0], quad_z[:, 0], blocks)


def _block_edges(
    i: IntArray, j: IntArray, corner_x: FloatArray, corner_z: FloatArray
) -> _Polylines:
    """The four lattice edges around every block, each shared edge once, in key order."""
    width = corner_x.shape[1]
    keys = np.unique(
        np.concatenate(
            (
                (i * width + j) * 2,
                (i * width + j) * 2 + 1,
                (i * width + j + 1) * 2,
                ((i + 1) * width + j) * 2 + 1,
            )
        )
    )
    vertical = keys % 2
    a, b = np.divmod(keys // 2, width)
    end_a, end_b = a + 1 - vertical, b + vertical
    return _Polylines(
        np.full(keys.size, 2, dtype=np.int64),
        np.column_stack((corner_x[a, b], corner_x[end_a, end_b])).ravel(),
        np.column_stack((corner_z[a, b], corner_z[end_a, end_b])).ravel(),
        ROAD_STREET,
    )


def _concat_blocks(parts: list[_Blocks]) -> _Blocks:
    return _Blocks(
        np.concatenate([b.quad_x for b in parts]),
        np.concatenate([b.quad_z for b in parts]),
        np.concatenate([b.size for b in parts]),
        np.concatenate([b.city_f for b in parts]),
        np.concatenate([b.district for b in parts]),
    )


def _fill_blocks(blocks: _Blocks, mask: NDArray[np.bool_], rng: np.random.Generator) -> Buildings:
    """Split each block into a grid of lots and put one box on each lot (prototype `fillBlock`)."""
    c00, c10, c11, c01 = ((blocks.quad_x[:, k], blocks.quad_z[:, k]) for k in range(4))
    ux = (c10[0] - c00[0] + c11[0] - c01[0]) / 2
    uz = (c10[1] - c00[1] + c11[1] - c01[1]) / 2
    vx = (c01[0] - c00[0] + c11[0] - c10[0]) / 2
    vz = (c01[1] - c00[1] + c11[1] - c10[1]) / 2
    len_u, len_v = np.hypot(ux, uz), np.hypot(vx, vz)
    angle = np.arctan2(uz, ux)
    mid_x, mid_z = _bilinear(blocks, 0.5, 0.5)
    old = np.hypot(mid_x - OLD_CITY[0], mid_z - OLD_CITY[1]) < 950
    cols, rows = _lot_grid(rng.random(old.size), old)
    margin = 8 / blocks.size
    tall = peak_at(mid_x, mid_z) * (1 - smoothstep(0.6, 1.0, blocks.city_f))

    lots = cols * rows
    block = np.repeat(np.arange(lots.size), lots)
    lot = np.arange(block.size) - np.repeat(np.cumsum(lots) - lots, lots)
    col, row = np.divmod(lot, rows[block])
    span = 1 - 2 * margin[block]
    u0 = margin[block] + span * col / cols[block]
    u1 = margin[block] + span * (col + 1) / cols[block]
    v0 = margin[block] + span * row / rows[block]
    v1 = margin[block] + span * (row + 1) / rows[block]
    x, z = _bilinear(blocks, (u0 + u1) / 2, (v0 + v1) / 2, block)

    draw = rng.random((block.size, 7))
    lot_tall = tall[block]
    width = (u1 - u0) * len_u[block] * (0.58 + draw[:, 1] * 0.34)
    depth = (v1 - v0) * len_v[block] * (0.58 + draw[:, 2] * 0.34)
    height = 6 + draw[:, 3] * np.where(old[block], 12, 9) + 104 * lot_tall * draw[:, 4] ** 1.7
    tower = (lot_tall > 0.3) & (draw[:, 5] < 0.11 * lot_tall)
    height = np.where(tower, 62 + draw[:, 6] * 58 * np.minimum(1, lot_tall + 0.2), height)
    width = np.where(tower, width * 0.78, width)
    depth = np.where(tower, depth * 0.78, depth)
    height = np.clip(height, 6, 120)

    turn = angle[block]
    clear = (draw[:, 0] >= 0.05) & _lot_is_clear(x, z, width, depth, turn, mask)
    return Buildings(
        x=x[clear].astype(np.float32),
        y=(-z[clear]).astype(np.float32),
        width=width[clear].astype(np.float32),
        depth=depth[clear].astype(np.float32),
        height=height[clear].astype(np.float32),
        rotation=(-turn[clear]).astype(np.float32),
        district=blocks.district[block[clear]].astype(np.uint16),
    )


def _bilinear(
    blocks: _Blocks, u: FloatArray | float, v: FloatArray | float, block: IntArray | None = None
) -> tuple[FloatArray, FloatArray]:
    qx = blocks.quad_x if block is None else blocks.quad_x[block]
    qz = blocks.quad_z if block is None else blocks.quad_z[block]
    w00, w10, w11, w01 = (1 - u) * (1 - v), u * (1 - v), u * v, (1 - u) * v
    x = w00 * qx[:, 0] + w10 * qx[:, 1] + w11 * qx[:, 2] + w01 * qx[:, 3]
    z = w00 * qz[:, 0] + w10 * qz[:, 1] + w11 * qz[:, 2] + w01 * qz[:, 3]
    return x, z


def _lot_grid(r: FloatArray, old: NDArray[np.bool_]) -> tuple[IntArray, IntArray]:
    cols = np.select([r < 0.13, r < 0.33, r < 0.5, r < 0.86], [1, 2, 1, 2], default=3)
    rows = np.select([r < 0.13, r < 0.33, r < 0.5, r < 0.86], [1, 1, 2, 2], default=2)
    cols = np.where(old, np.where(r < 0.5, 3, 4), cols)
    rows = np.where(old, 3, rows)
    return cols.astype(np.int64), rows.astype(np.int64)


def _lot_is_clear(
    x: FloatArray,
    z: FloatArray,
    width: FloatArray,
    depth: FloatArray,
    turn: FloatArray,
    mask: NDArray[np.bool_],
) -> NDArray[np.bool_]:
    """The four corners and the centre stay off arterials and at least 22 m from the river."""
    cos_t, sin_t = np.cos(turn), np.sin(turn)
    clear = np.ones(x.size, dtype=bool)
    for sx, sz in ((-1, -1), (1, -1), (1, 1), (-1, 1), (0, 0)):
        px = x + cos_t * sx * width / 2 - sin_t * sz * depth / 2
        pz = z + sin_t * sx * width / 2 + cos_t * sz * depth / 2
        clear &= ~(_on_mask(mask, px, pz) | in_river(px, pz, 22))
    return clear


def _bridges() -> _Polylines:
    z = np.array(BRIDGE_Z)
    reach = river_half_width(z) + 30
    x = np.column_stack((river_x(z) - reach, river_x(z) + reach)).ravel()
    return _Polylines(
        np.full(z.size, 2, dtype=np.int64),
        x,
        np.column_stack((z, z + 8)).ravel(),
        ROAD_BRIDGE,
    )


def _roads(groups: list[_Polylines]) -> Roads:
    lengths = np.concatenate([g.lengths for g in groups])
    return Roads(
        offsets=np.concatenate(([0], np.cumsum(lengths))).astype(np.uint32),
        x=np.concatenate([g.x for g in groups]).astype(np.float32),
        y=(-np.concatenate([g.z for g in groups])).astype(np.float32),
        kind=np.concatenate([np.full(g.lengths.size, g.kind) for g in groups]).astype(np.uint8),
    )


def _river() -> River:
    """One polygon: the west bank north to south, then the east bank back north."""
    z = np.arange(-RIVER_HALF_SAMPLES, RIVER_HALF_SAMPLES + 1) * RIVER_STEP_M
    centre, half = river_x(z), river_half_width(z)
    x = np.concatenate((centre - half, (centre + half)[::-1]))
    zs = np.concatenate((z, z[::-1]))
    return River(
        offsets=np.array([0, x.size], dtype=np.uint32),
        x=x.astype(np.float32),
        y=(-zs).astype(np.float32),
    )
