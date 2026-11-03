from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from pyproj import Transformer

from twl_citybake.config import BBox

FloatArray = npt.NDArray[np.float64]


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

    def _transformer(self, inverse: bool) -> Transformer:
        src, dst = ("EPSG:4326", self.proj4)
        return Transformer.from_crs(*((dst, src) if inverse else (src, dst)), always_xy=True)

    def forward(self, lon: npt.ArrayLike, lat: npt.ArrayLike) -> tuple[FloatArray, FloatArray]:
        x, y = self._transformer(inverse=False).transform(
            np.asarray(lon, np.float64), np.asarray(lat, np.float64)
        )
        return np.asarray(x), np.asarray(y)

    def inverse(self, x: npt.ArrayLike, y: npt.ArrayLike) -> tuple[FloatArray, FloatArray]:
        lon, lat = self._transformer(inverse=True).transform(
            np.asarray(x, np.float64), np.asarray(y, np.float64)
        )
        return np.asarray(lon), np.asarray(lat)
