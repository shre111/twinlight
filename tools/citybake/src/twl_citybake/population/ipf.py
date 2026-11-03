from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
TOTALS_RTOL = 1e-9


@dataclass(frozen=True)
class IpfResult:
    matrix: FloatArray
    iterations: int
    max_rel_error: float
    converged: bool


def ipf(
    seed: npt.ArrayLike,
    row_targets: npt.ArrayLike,
    col_targets: npt.ArrayLike,
    tol: float = 1e-9,
    max_iter: int = 1000,
) -> IpfResult:
    """Biproportional fitting of `seed` to row and column sums. Zero seed cells are structural
    zeros and stay zero; rows and columns with a zero target are zeroed."""
    matrix = np.array(seed, np.float64)
    rows = np.asarray(row_targets, np.float64)
    cols = np.asarray(col_targets, np.float64)
    _check_inputs(matrix, rows, cols)
    matrix[rows == 0] = 0.0
    matrix[:, cols == 0] = 0.0
    _check_support(matrix, rows, cols)
    iterations = 0
    error = max_rel_error(matrix, rows, cols)
    while error > tol and iterations < max_iter:
        _scale(matrix, rows, axis=1)
        _scale(matrix, cols, axis=0)
        iterations += 1
        error = max_rel_error(matrix, rows, cols)
    return IpfResult(matrix, iterations, error, error <= tol)


def max_rel_error(matrix: FloatArray, rows: FloatArray, cols: FloatArray) -> float:
    row_err = np.abs(matrix.sum(1) - rows) / np.maximum(rows, 1.0)
    col_err = np.abs(matrix.sum(0) - cols) / np.maximum(cols, 1.0)
    return float(max(row_err.max(initial=0.0), col_err.max(initial=0.0)))


def integerise(
    expected: npt.ArrayLike,
    sizes: npt.ArrayLike,
    capacity: npt.ArrayLike,
    col_totals: npt.ArrayLike,
) -> IntArray:
    """Whole households per (building, size class) from expected counts: floors, then one more
    household at a time by largest remainder (ties by lower row) wherever it still fits, largest
    sizes first, until each column reaches its total or nothing fits."""
    wanted = np.maximum(np.asarray(expected, np.float64), 0.0)
    size = np.asarray(sizes, np.int64)
    room_for = np.maximum(np.asarray(capacity, np.int64), 0)
    totals = np.asarray(col_totals, np.int64)
    counts = np.floor(wanted).astype(np.int64)
    remainder = wanted - counts
    _trim_to_capacity(counts, size, room_for)
    persons = counts @ size
    for k in np.argsort(-size, kind="stable"):
        need = int(totals[k] - counts[:, k].sum())
        order = np.argsort(-remainder[:, k], kind="stable")
        while need > 0:
            fits = order[room_for[order] - persons[order] >= size[k]][:need]
            if not len(fits):
                break
            counts[fits, k] += 1
            persons[fits] += size[k]
            need -= len(fits)
    return counts


def _trim_to_capacity(counts: IntArray, size: IntArray, capacity: IntArray) -> None:
    """Floors can only overflow when the fit behind `expected` did not converge; drop the
    largest households of overflowing rows until they fit."""
    over = counts @ size - capacity
    for k in np.argsort(-size, kind="stable"):
        if not (over > 0).any():
            return
        drop = np.minimum(counts[:, k], -(-np.maximum(over, 0) // size[k]))
        counts[:, k] -= drop
        over -= drop * size[k]


def _scale(matrix: FloatArray, targets: FloatArray, axis: int) -> None:
    sums = matrix.sum(axis)
    factor = np.divide(targets, sums, out=np.zeros_like(targets), where=sums > 0)
    matrix *= factor[:, None] if axis == 1 else factor[None, :]


def _check_inputs(matrix: FloatArray, rows: FloatArray, cols: FloatArray) -> None:
    if matrix.ndim != 2 or matrix.shape != (len(rows), len(cols)):
        raise ValueError(f"seed {matrix.shape} does not match targets ({len(rows)}, {len(cols)})")
    for name, values in (("seed", matrix), ("row targets", rows), ("column targets", cols)):
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError(f"{name} must be finite and non-negative")
    row_total, col_total = rows.sum(), cols.sum()
    if abs(row_total - col_total) > TOTALS_RTOL * max(row_total, col_total):
        raise ValueError(f"row targets sum to {row_total}, column targets to {col_total}")


def _check_support(matrix: FloatArray, rows: FloatArray, cols: FloatArray) -> None:
    if ((rows > 0) & (matrix.sum(1) == 0)).any():
        raise ValueError("a row with a positive target has no positive seed cell")
    if ((cols > 0) & (matrix.sum(0) == 0)).any():
        raise ValueError("a column with a positive target has no positive seed cell")
