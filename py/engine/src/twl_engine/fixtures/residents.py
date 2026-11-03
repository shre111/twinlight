import numpy as np

from twl_engine.fixtures.geometry import DISTRICT_EAST
from twl_engine.fixtures.types import NO_PERSONA, Buildings, Residents

AGE_SHARE_WEST = np.array([0.2, 0.27, 0.27, 0.16, 0.1])
AGE_SHARE_EAST = np.array([0.16, 0.22, 0.28, 0.21, 0.13])
FIRST_FLOOR_M = 2.4
FLOOR_M = 3.3
WALL_GAP_M = 0.8


def place_residents(buildings: Buildings, count: int, rng: np.random.Generator) -> Residents:
    """Residents by floor area, on a floor just outside a wall (prototype `placeResidents`)."""
    width = buildings.width.astype(np.float64)
    depth = buildings.depth.astype(np.float64)
    height = buildings.height.astype(np.float64)
    capacity = np.cumsum(width * depth * height**0.85)
    draw = rng.random((count, 6))
    home = np.minimum(
        np.searchsorted(capacity, draw[:, 0] * capacity[-1], side="left"), capacity.size - 1
    )
    w, d, h = width[home], depth[home], height[home]

    floors = np.maximum(1, np.floor((h - 2.5) / FLOOR_M))
    floor_z = np.minimum(h - 1.2, FIRST_FLOOR_M + np.floor(draw[:, 1] * floors) * FLOOR_M)
    side = np.where(draw[:, 2] < 0.5, -1.0, 1.0)
    on_depth_wall = draw[:, 3] < d / (w + d)
    along = draw[:, 4] - 0.5
    local_x = np.where(on_depth_wall, side * (w / 2 + WALL_GAP_M), along * 0.84 * w)
    local_y = np.where(on_depth_wall, along * 0.84 * d, side * (d / 2 + WALL_GAP_M))
    turn = buildings.rotation[home].astype(np.float64)
    cos_t, sin_t = np.cos(turn), np.sin(turn)
    x = buildings.x[home] + local_x * cos_t - local_y * sin_t
    y = buildings.y[home] + local_x * sin_t + local_y * cos_t

    east = DISTRICT_EAST[buildings.district[home]]
    band = np.where(
        east,
        np.searchsorted(np.cumsum(AGE_SHARE_EAST)[:-1], draw[:, 5], side="left"),
        np.searchsorted(np.cumsum(AGE_SHARE_WEST)[:-1], draw[:, 5], side="left"),
    )
    return Residents(
        x=x.astype(np.float32),
        y=y.astype(np.float32),
        z=floor_z.astype(np.float32),
        building=home.astype(np.uint32),
        hex=np.zeros(count, dtype=np.uint32),
        segment=band.astype(np.uint16),
        persona_ref=np.full(count, NO_PERSONA, dtype=np.uint32),
        weight=np.ones(count, dtype=np.float32),
    )
