from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.optimize import isotonic_regression

FloatArray = npt.NDArray[np.float64]
BANDS = ("1-2", "3-4", "5-7", "8-12", "13+")
BAND_STARTS = (3.0, 5.0, 8.0, 13.0)
RASTER_BINS = ("0-3", "3-6", "6-9", "9-12", "12-15", "15-18", "18-24", "24-30", "30+")
RASTER_BIN_STARTS = (3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 24.0, 30.0)
MIN_TOP_COUNT = 30


@dataclass(frozen=True)
class HeightCorrection:
    """A monotone map from raw raster height to storey height (D-024): linear between bin
    centres; below the first, the first bin's ratio; above the last, the last bin's ratio or 1
    if that is higher, so a tall reading is never shrunk. When that floor binds, the map jumps
    upward at the last centre: still monotone, not continuous there."""

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
        above = raw * max(float(self.fitted[-1] / last), 1.0)
        return np.where(raw < first, below, np.where(raw > last, above, inside))

    def knots(self) -> dict[str, list[float]]:
        return {"centres": _rounded(self.centres), "fitted": _rounded(self.fitted)}


def can_fit(raw: npt.ArrayLike, target: npt.ArrayLike, bin_m: float = 1.0) -> bool:
    """Whether the finite sample fills the two bins fit_correction needs."""
    sample_raw, _ = _sample(raw, target)
    return len(np.unique(_bins(sample_raw, bin_m))) >= 2


def fit_correction(
    raw: npt.ArrayLike,
    target: npt.ArrayLike,
    bin_m: float = 1.0,
    min_top_count: int = MIN_TOP_COUNT,
) -> HeightCorrection:
    """Isotonic fit of the per-bin mean target, weighted by bin count, at `bin_m` bins of the raw
    height. The mean because storey targets are discrete: a bin median snaps to a multiple of
    3.2 m and turns the fit into a staircase. Sparse top bins pool until each group holds
    `min_top_count` rows, centred on their mean raw height. Rows with a non-finite raw height or
    target are dropped."""
    sample_raw, sample_target = _sample(raw, target)
    index, row_bin, counts = np.unique(
        _bins(sample_raw, bin_m), return_inverse=True, return_counts=True
    )
    if len(index) < 2:
        raise ValueError(f"need at least 2 non-empty {bin_m} m bins, got {len(index)}")
    group = _top_pooled_groups(counts, min_top_count)
    row_group = group[row_bin]
    n = np.bincount(row_group).astype(np.float64)
    means = np.bincount(row_group, weights=sample_target) / n
    fitted = isotonic_regression(means, weights=n, increasing=True).x
    pooled = np.bincount(group) > 1
    centres = np.where(
        pooled,
        np.bincount(row_group, weights=sample_raw) / n,
        np.bincount(group, weights=(index + 0.5).astype(np.float64) * bin_m),
    )
    return HeightCorrection(centres, np.asarray(fitted, np.float64))


def _top_pooled_groups(counts: npt.NDArray[np.int64], min_count: int) -> npt.NDArray[np.int64]:
    """Ascending group id per ascending bin. Walking down from the top, bins join a group until it
    holds `min_count` rows; an under-filled lowest group joins the one above. A sample too small
    to make two groups stays unpooled."""
    descending = np.zeros(len(counts), np.int64)
    group = held = 0
    for i in range(len(counts) - 1, -1, -1):
        descending[i] = group
        held += int(counts[i])
        if held >= min_count:
            group, held = group + 1, 0
    if held and group:
        descending[descending == group] = group - 1
    if descending.max() == 0:
        return np.arange(len(counts), dtype=np.int64)
    return descending.max() - descending


def validate_by_district(
    raw: npt.ArrayLike,
    target: npt.ArrayLike,
    district: npt.ArrayLike,
    levels: npt.ArrayLike,
    folds: int = 5,
) -> dict[str, Any]:
    """Median absolute and signed percentage error, per storey band and per raster height bin, of
    the raw heights and of the correction fitted without the row's district (fold = district id
    mod folds), plus the aggregate error (sum predicted over sum true, minus 1). A mean-calibrated
    curve keeps the aggregate near zero by raster bin; its median can sit off zero where storeys
    are skewed, and it tilts low to high by storey band. A fold whose training rows cannot be
    fitted keeps the raw heights, as the stage does with too small a sample."""
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
    raster_bin = np.searchsorted(RASTER_BIN_STARTS, raw_h, side="right")
    return {
        "folds": folds,
        "bands": {
            name: _errors(raw_h, predicted, target_h, band == i) for i, name in enumerate(BANDS)
        },
        "raster_bins": {
            name: _errors(raw_h, predicted, target_h, raster_bin == i)
            for i, name in enumerate(RASTER_BINS)
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


def _errors(
    raw: FloatArray, predicted: FloatArray, target: FloatArray, rows: npt.NDArray[np.bool_]
) -> dict[str, Any]:
    n = int(rows.sum())
    if n == 0:
        return dict.fromkeys(
            (
                "before",
                "after",
                "signed_before",
                "signed_after",
                "aggregate_before",
                "aggregate_after",
            )
        ) | {"n": 0}
    return {
        "n": n,
        "before": _median_error(np.abs(raw[rows] - target[rows]), target[rows]),
        "after": _median_error(np.abs(predicted[rows] - target[rows]), target[rows]),
        "signed_before": _median_error(raw[rows] - target[rows], target[rows]),
        "signed_after": _median_error(predicted[rows] - target[rows], target[rows]),
        "aggregate_before": round(float(raw[rows].sum() / target[rows].sum() - 1), 3),
        "aggregate_after": round(float(predicted[rows].sum() / target[rows].sum() - 1), 3),
    }


def _median_error(error: FloatArray, target: FloatArray) -> float:
    return round(float(np.median(error / target)), 3)


def _rounded(values: FloatArray) -> list[float]:
    return [round(float(v), 3) for v in values]
