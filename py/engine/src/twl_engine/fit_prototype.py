"""Fit the prototype-parity fixture's three free parameters (D-027).

`python -m twl_engine.fit_prototype [--write]` fits `paid_scale`, `q` and `alpha_0` to the
prototype's heard and ordered shares on days 6 and 13 only (mean of seeds 1 to 8 on the
48,000-resident toy city), then reports every day of the prototype's summary. Day 30 is out of
sample: it is printed, never fitted. `--write` rewrites the fitted values, which close the
scenario file's `[params]` table below `PARAMS_HEADER`; the pinned values above it are kept.
"""

import argparse
import dataclasses
import math
import os
from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from twl_engine.goldens import PROTOTYPE_SCENARIO_PATH, prototype_inputs
from twl_engine.montecarlo import run_seeds
from twl_engine.run import RunInputs
from twl_engine.scenario import MINUTES_PER_DAY
from twl_tape import read_tape

SEEDS = tuple(range(1, 9))
FIT_DAYS = (6, 13)
REPORT_DAYS = (1, 6, 10, 13, 20, 30)
PROTOTYPE_SHARES = {
    1: (13.4, 0.5),
    6: (23.0, 4.5),
    10: (36.5, 8.0),
    13: (41.7, 10.8),
    20: (51.9, 16.1),
    30: (63.1, 23.0),
}
START = (1.0, 0.02, -3.0)
STEPS = (0.15, 0.15, 0.25)
MAX_ITERATIONS = 12
DAMPING = 0.05
SIGNIFICANT_DIGITS = 4
PARAMS_HEADER = (
    "# Fitted by `python -m twl_engine.fit_prototype` on days 6 and 13 only; never edit by hand."
)

Shares = NDArray[np.float64]


@dataclass(frozen=True)
class Fit:
    paid_scale: float
    q: float
    alpha_0: float

    def coordinates(self) -> NDArray[np.float64]:
        return np.array([math.log(self.paid_scale), math.log(self.q), self.alpha_0])

    @staticmethod
    def at(coordinates: NDArray[np.float64]) -> "Fit":
        return Fit(
            paid_scale=_rounded(math.exp(coordinates[0])),
            q=_rounded(math.exp(coordinates[1])),
            alpha_0=_rounded(float(coordinates[2])),
        )


def fit_inputs(fit: Fit, days: int, seed: int) -> RunInputs:
    base = prototype_inputs(seed)
    return dataclasses.replace(
        base,
        scenario=dataclasses.replace(base.scenario, days=days),
        params=dataclasses.replace(base.params, **dataclasses.asdict(fit)),
    )


def mean_shares(fit: Fit, days: Sequence[int], workers: int) -> Shares:
    """Percent of residents who heard and who ordered by the end of each day, mean over SEEDS."""
    tapes = run_seeds(partial(fit_inputs, fit, max(days)), SEEDS, workers=workers)
    return np.mean([_shares(data, days) for data in tapes], axis=0)


def _shares(data: bytes, days: Sequence[int]) -> Shares:
    residents = read_tape(data).residents
    ends = np.array(days) * MINUTES_PER_DAY
    heard = (residents.aware_at[None, :] < ends[:, None]).mean(axis=1)
    ordered = (residents.first_order_at[None, :] < ends[:, None]).mean(axis=1)
    return 100.0 * np.stack([heard, ordered], axis=1)


def residuals(fit: Fit, workers: int) -> NDArray[np.float64]:
    targets = np.array([PROTOTYPE_SHARES[day] for day in FIT_DAYS])
    return (mean_shares(fit, FIT_DAYS, workers) - targets).ravel()


def fit_parameters(workers: int) -> Fit:
    """Damped Gauss-Newton on (ln paid_scale, ln q, alpha_0) with forward differences.

    The steps are wide because the residuals carry seed noise; the search keeps the best point
    seen and stops when a step no longer lowers the squared error.
    """
    x = Fit(*START).coordinates()
    best = Fit.at(x)
    r = residuals(best, workers)
    for iteration in range(MAX_ITERATIONS):
        print(f"iteration {iteration}: {best}  squared error {r @ r:.3f}", flush=True)
        jacobian = np.empty((r.shape[0], x.shape[0]))
        for k, step in enumerate(STEPS):
            shifted = x.copy()
            shifted[k] += step
            jacobian[:, k] = (residuals(Fit.at(shifted), workers) - r) / step
        normal = jacobian.T @ jacobian
        delta = np.linalg.solve(normal + DAMPING * np.diag(np.diag(normal)), -jacobian.T @ r)
        while True:
            candidate = Fit.at(x + delta)
            r_candidate = residuals(candidate, workers)
            if r_candidate @ r_candidate < r @ r:
                break
            delta = delta / 2.0
            if np.abs(delta).max() < 0.005:
                return best
        x, best, r = candidate.coordinates(), candidate, r_candidate
    return best


def rewrite_params(path: Path, fit: Fit) -> None:
    text = path.read_text()
    head = text[: text.index(PARAMS_HEADER)]
    if "\n[params]\n" not in head:
        raise ValueError(f"{path}: the fitted values must close the [params] table")
    values = "".join(f"{key} = {value!r}\n" for key, value in dataclasses.asdict(fit).items())
    path.write_text(f"{head}{PARAMS_HEADER}\n{values}")


def report(fit: Fit, workers: int) -> None:
    shares = mean_shares(fit, REPORT_DAYS, workers)
    print(f"{fit}")
    print("day   heard  (prototype)  ordered  (prototype)")
    for day, (heard, ordered) in zip(REPORT_DAYS, shares, strict=True):
        want_heard, want_ordered = PROTOTYPE_SHARES[day]
        role = "fit" if day in FIT_DAYS else "out of sample"
        print(
            f"{day:3d}  {heard:6.2f}  ({want_heard:5.1f})      {ordered:6.2f}  "
            f"({want_ordered:5.1f})     {role}"
        )


def _rounded(value: float) -> float:
    return float(f"{value:.{SIGNIFICANT_DIGITS}g}")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m twl_engine.fit_prototype")
    parser.add_argument("--write", action="store_true", help=f"rewrite {PROTOTYPE_SCENARIO_PATH}")
    parser.add_argument("--workers", type=int, default=min(len(SEEDS), os.cpu_count() or 1))
    args = parser.parse_args(argv)
    fit = fit_parameters(args.workers)
    if args.write:
        rewrite_params(PROTOTYPE_SCENARIO_PATH, fit)
        print(f"wrote {PROTOTYPE_SCENARIO_PATH}")
    report(fit, args.workers)


if __name__ == "__main__":
    main()
