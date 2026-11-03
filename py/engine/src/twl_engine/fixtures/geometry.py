"""Toy city shape from design/prototype/toy-sim.mjs, in the prototype's three.js frame.

Here x is east and z is south (negative z is north); pack coordinates are x east, y = -z north.
"""

import math
from typing import Any

import numpy as np
from numpy.typing import NDArray

from twl_engine.fixtures.types import District

FloatArray = NDArray[np.floating[Any]]

CITY_CENTRE = (-250.0, 0.0)
CITY_RADII = (3500.0, 3200.0)
OLD_CITY = (1400.0, 150.0)
CBD = (-1500.0, -350.0)

DISTRICT_TABLE: tuple[tuple[str, float, float], ...] = (
    ("Ellisbridge", -250.0, 300.0),
    ("Paldi", -350.0, 1450.0),
    ("Navrangpura", -950.0, -550.0),
    ("Naranpura", -850.0, -1650.0),
    ("Chandkheda", -250.0, -2650.0),
    ("Ghatlodia", -1750.0, -2250.0),
    ("Vastrapur", -2000.0, 350.0),
    ("Bodakdev", -2350.0, -900.0),
    ("Thaltej", -3050.0, -1500.0),
    ("Satellite", -1750.0, 1500.0),
    ("Shahibaug", 1650.0, -1350.0),
    ("Maninagar", 1850.0, 1350.0),
)


def river_x(z: FloatArray) -> FloatArray:
    return 520 + 230 * np.sin(z / 1250 + 0.6) + 110 * np.sin(z / 470 + 2.1)


def river_half_width(z: FloatArray) -> FloatArray:
    return 88 + 18 * np.sin(z / 800 + 1.0)


def in_river(x: FloatArray, z: FloatArray, pad: float) -> NDArray[np.bool_]:
    return np.abs(x - river_x(z)) < river_half_width(z) + pad


def smoothstep(a: float, b: float, x: FloatArray) -> FloatArray:
    t = np.clip((x - a) / (b - a), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def city_f(x: FloatArray, z: FloatArray) -> FloatArray:
    """Normalised distance from the centre of the wobbly city ellipse; 1 is the edge."""
    dx = (x - CITY_CENTRE[0]) / CITY_RADII[0]
    dz = (z - CITY_CENTRE[1]) / CITY_RADII[1]
    th = np.arctan2(dz, dx)
    r = 1 + 0.07 * np.sin(3 * th + 1) + 0.05 * np.sin(5 * th + 2) + 0.03 * np.sin(9 * th)
    return np.hypot(dx, dz) / r


def peak_at(x: FloatArray, z: FloatArray) -> FloatArray:
    def bump(cx: float, cz: float, sigma: float) -> FloatArray:
        return np.exp(-((x - cx) ** 2 + (z - cz) ** 2) / (2 * sigma * sigma))

    return np.maximum.reduce(
        [
            bump(CBD[0], CBD[1], 850),
            0.62 * bump(OLD_CITY[0], OLD_CITY[1], 650),
            0.5 * bump(-2650, -1350, 520),
            0.36 * bump(-1750, 1500, 520),
        ]
    )


DISTRICT_X = np.array([x for _, x, _ in DISTRICT_TABLE])
DISTRICT_Z = np.array([z for _, _, z in DISTRICT_TABLE])
DISTRICT_EAST = np.greater(DISTRICT_X, river_x(DISTRICT_Z))


def district_at(x: FloatArray, z: FloatArray) -> NDArray[np.intp]:
    """Nearest district centre on the same bank of the river."""
    east = x > river_x(z)
    d2 = (x[:, None] - DISTRICT_X) ** 2 + (z[:, None] - DISTRICT_Z) ** 2
    d2[east[:, None] != DISTRICT_EAST] = np.inf
    return np.argmin(d2, axis=1)


def toy_districts() -> tuple[District, ...]:
    return tuple(
        District(id=i, name=name, x=x, y=-z, east=bool(DISTRICT_EAST[i]))
        for i, (name, x, z) in enumerate(DISTRICT_TABLE)
    )


def _sample(a: float, b: float, n: int) -> FloatArray:
    return a + (b - a) * np.arange(n + 1) / n


def _diagonal(p: tuple[float, float], q: tuple[float, float]) -> tuple[FloatArray, FloatArray]:
    t = _sample(0.0, 1.0, 70)
    return p[0] + (q[0] - p[0]) * t + 70 * np.sin(t * 5.5), p[1] + (q[1] - p[1]) * t


def arterial_lines() -> list[tuple[FloatArray, FloatArray]]:
    a = _sample(0.0, 2 * math.pi, 220)
    wobble = 1 + 0.03 * np.sin(a * 5 + 0.7)
    ring = (-250 + 2700 * wobble * np.cos(a), -50 + 2450 * wobble * np.sin(a))
    west_z = _sample(-3900, 3700, 130)
    river_z = _sample(-3400, 3000, 130)
    north_x = _sample(-4000, 3400, 130)
    south_x = _sample(-3800, 3200, 130)
    far_north_x = _sample(-3700, 3100, 130)
    return [
        ring,
        (-2750 + 180 * np.sin(west_z / 1600), west_z),
        (river_x(river_z) - river_half_width(river_z) - 150, river_z),
        (north_x, -400 + 120 * np.sin(north_x / 900)),
        (south_x, 1350 + 100 * np.sin(south_x / 1100 + 1)),
        (far_north_x, -1750 + 90 * np.sin(far_north_x / 700)),
        _diagonal(OLD_CITY, (3300, -2600)),
        _diagonal(OLD_CITY, (3200, 2800)),
        _diagonal(OLD_CITY, (1000, -3400)),
        _diagonal(CBD, (-3600, -3100)),
        _diagonal(CBD, (-3500, 2700)),
    ]
