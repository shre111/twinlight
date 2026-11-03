from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.optimize import isotonic_regression

FloatArray = npt.NDArray[np.float64]
BANDS = ("1-2", "3-4", "5-7", "8-12", "13+")
BAND_STARTS = (3.0, 5.0, 8.0, 13.0)


@dataclass(frozen=True)
class HeightCorrection:
    """A monotone map from raw raster height to storey height (D-024): linear between bin
    centres, and proportional to the raw height past either end, keeping that end bin's ratio."""

    centres: FloatArray
    fitted: FloatArray

    def __post_init__(self) -> None:
        if len(self.centres) < 2 or len(self.centres) != len(self.fitted):
            raise ValueError("a height correction needs at least two centres, one fit per centre")
        if not (np.diff(self.centres) > 0).all():
            raise ValueError("height correction centres must increase")

    def apply(self, raw: npt.ArrayLike) -> FloatArray:
        raw = np.asarray(raw, np.float64)
        first, last = self.centres[0], self.centres[-1]
        inside = np.interp(raw, self.centres, self.fitted)
        below = raw * self.fitted[0] / first
        above = raw * self.fitted[-1] / last
        return np.where(raw < first, below, np.where(raw > last, above, inside))

    def knots(self) -> dict[str, list[float]]:
        return {"centres": _rounded(self.centres), "fitted": _rounded(self.fitted)}


def can_fit(raw: npt.ArrayLike, target: npt.ArrayLike, bin_m: float = 1.0) -> bool:
    """Whether the finite sample fills the two bins fit_correction needs."""
    sample_raw, _ = _sample(raw, target)
    return len(np.unique(_bins(sample_raw, bin_m))) >= 2


def fit_correction(
    raw: npt.ArrayLike, target: npt.ArrayLike, bin_m: float = 1.0
) -> HeightCorrection:
    """Isotonic fit of the per-bin median target, weighted by bin count, at `bin_m` bins of the
    raw height. Rows with a non-finite raw height or target are dropped."""
    sample_raw, sample_target = _sample(raw, target)
    index, medians, counts = _bin_medians(_bins(sample_raw, bin_m), sample_target)
    if len(index) < 2:
        raise ValueError(f"need at least 2 non-empty {bin_m} m bins, got {len(index)}")
    fitted = isotonic_regression(medians, weights=counts.astype(np.float64), increasing=True).x
    centres = (index + 0.5).astype(np.float64) * bin_m
    return HeightCorrection(centres, np.asarray(fitted, np.float64))


def validate_by_district(
    raw: npt.ArrayLike,
    target: npt.ArrayLike,
    district: npt.ArrayLike,
    levels: npt.ArrayLike,
    folds: int = 5,
) -> dict[str, Any]:
    """Median absolute percentage error per storey band of the raw heights and of the correction
    fitted without the row's district (fold = district id mod folds). A fold whose training rows
    cannot be fitted keeps the raw heights, as the stage does with too small a sample."""
    if folds < 2:
        raise ValueError(f"need at least 2 folds, got {folds}")
    raw_h, target_h, levels_f = (np.asarray(a, np.float64) for a in (raw, target, levels))
    keep = np.isfinite(raw_h) & np.isfinite(target_h) & (target_h > 0) & np.isfinite(levels_f)
    raw_h, target_h, levels_f = raw_h[keep], target_h[keep], levels_f[keep]
    fold = np.asarray(district, np.int64)[keep] % folds
    predicted = raw_h.copy()
    for k in range(folds):
        held, train = fold == k, fold != k
        if held.any() and can_fit(raw_h[train], target_h[train]):
            predicted[held] = fit_correction(raw_h[train], target_h[train]).apply(raw_h[held])
    band = np.searchsorted(BAND_STARTS, levels_f, side="right")
    return {
        "folds": folds,
        "bands": {
            name: _errors(raw_h, predicted, target_h, band == i) for i, name in enumerate(BANDS)
        },
        "overall": _errors(raw_h, predicted, target_h, np.ones(len(raw_h), np.bool_)),
    }


def _sample(raw: npt.ArrayLike, target: npt.ArrayLike) -> tuple[FloatArray, FloatArray]:
    raw_h, target_h = np.asarray(raw, np.float64), np.asarray(target, np.float64)
    if raw_h.shape != target_h.shape or raw_h.ndim != 1:
        raise ValueError(f"raw {raw_h.shape} and target {target_h.shape} must be equal 1-d")
    seen = np.isfinite(raw_h) & np.isfinite(target_h)
    return raw_h[seen], target_h[seen]


def _bins(raw: FloatArray, bin_m: float) -> npt.NDArray[np.int64]:
    if not bin_m > 0:
        raise ValueError(f"bin_m must be positive, got {bin_m}")
    return np.floor(raw / bin_m).astype(np.int64)


def _bin_medians(
    bins: npt.NDArray[np.int64], values: FloatArray
) -> tuple[npt.NDArray[np.int64], FloatArray, npt.NDArray[np.int64]]:
    """Per-bin np.median by sorting: the mean of the two middle values (one when odd)."""
    order = np.lexsort((values, bins))
    bins, values = bins[order], values[order]
    index, starts, counts = np.unique(bins, return_index=True, return_counts=True)
    medians = (values[starts + (counts - 1) // 2] + values[starts + counts // 2]) / 2
    return index, medians, counts


def _errors(
    raw: FloatArray, predicted: FloatArray, target: FloatArray, rows: npt.NDArray[np.bool_]
) -> dict[str, Any]:
    n = int(rows.sum())
    if n == 0:
        return {"n": 0, "before": None, "after": None}
    return {
        "n": n,
        "before": _median_ape(raw[rows], target[rows]),
        "after": _median_ape(predicted[rows], target[rows]),
    }


def _median_ape(height: FloatArray, target: FloatArray) -> float:
    return round(float(np.median(np.abs(height - target) / target)), 3)


def _rounded(values: FloatArray) -> list[float]:
    return [round(float(v), 3) for v in values]
