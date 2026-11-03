import numpy as np

from twl_engine.fixtures.geometry import FloatArray, district_at
from twl_engine.fixtures.types import Residents, Stores

KMEANS_SAMPLE = 9000
KMEANS_ITERATIONS = 14
DELIVERY_RADIUS_M = 900.0


def place_stores(
    residents: Residents,
    corners_x: FloatArray,
    corners_y: FloatArray,
    count: int,
    rng: np.random.Generator,
) -> Stores:
    """k-means over a resident sample, centres snapped to block corners, sorted west to east."""
    sample = rng.integers(0, residents.x.size, size=KMEANS_SAMPLE)
    centre_x, centre_y = _kmeans(
        residents.x[sample].astype(np.float64), residents.y[sample].astype(np.float64), count
    )
    corner = np.argmin(
        (corners_x[None, :] - centre_x[:, None]) ** 2
        + (corners_y[None, :] - centre_y[:, None]) ** 2,
        axis=1,
    )
    x, y = corners_x[corner], corners_y[corner]
    west_to_east = np.argsort(x, kind="stable")
    x, y = x[west_to_east], y[west_to_east]
    return Stores(
        x=x.astype(np.float32),
        y=y.astype(np.float32),
        hex=np.zeros(count, dtype=np.uint32),
        radius_m=np.full(count, DELIVERY_RADIUS_M, dtype=np.float32),
        district=district_at(x, -y).astype(np.uint16),
    )


def _kmeans(x: FloatArray, y: FloatArray, count: int) -> tuple[FloatArray, FloatArray]:
    seeds = np.arange(count) * x.size // count
    centre_x, centre_y = x[seeds], y[seeds]
    for _ in range(KMEANS_ITERATIONS):
        nearest = np.argmin(
            (x[:, None] - centre_x[None, :]) ** 2 + (y[:, None] - centre_y[None, :]) ** 2, axis=1
        )
        members = np.bincount(nearest, minlength=count)
        filled = members > 0
        sum_x = np.bincount(nearest, weights=x, minlength=count)
        sum_y = np.bincount(nearest, weights=y, minlength=count)
        centre_x = np.where(filled, sum_x / np.maximum(members, 1), centre_x)
        centre_y = np.where(filled, sum_y / np.maximum(members, 1), centre_y)
    return centre_x, centre_y
