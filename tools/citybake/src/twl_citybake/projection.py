from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import numpy.typing as npt
from pyproj import Transformer

from twl_citybake.config import BBox

FloatArray = npt.NDArray[np.float64]


@lru_cache(maxsize=8)
def _transformer(src: str, dst: str) -> Transformer:
    return Transformer.from_crs(src, dst, always_xy=True)


@dataclass(frozen=True)
class LocalProjection:
    """Transverse Mercator centred on the bbox: x east and y north in metres, origin at centre."""

    lon0: float
    lat0: float

    @classmethod
    def for_bbox(cls, bbox: BBox) -> "LocalProjection":
        return cls(*bbox.centre)

    @property
    def proj4(self) -> str:
        return (
            f"+proj=tmerc +lat_0={self.lat0!r} +lon_0={self.lon0!r} +k=1 +x_0=0 +y_0=0 "
            "+ellps=WGS84 +units=m +no_defs"
        )

    def forward(self, lon: npt.ArrayLike, lat: npt.ArrayLike) -> tuple[FloatArray, FloatArray]:
        x, y = _transformer("EPSG:4326", self.proj4).transform(
            np.asarray(lon, np.float64), np.asarray(lat, np.float64)
        )
        return np.asarray(x), np.asarray(y)

    def inverse(self, x: npt.ArrayLike, y: npt.ArrayLike) -> tuple[FloatArray, FloatArray]:
        lon, lat = _transformer(self.proj4, "EPSG:4326").transform(
            np.asarray(x, np.float64), np.asarray(y, np.float64)
        )
        return np.asarray(lon), np.asarray(lat)
