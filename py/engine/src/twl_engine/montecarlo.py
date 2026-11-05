from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context

from twl_engine.run import RunInputs, run_seed_bytes

InputsFor = Callable[[int], RunInputs]


def run_seeds(inputs_for: InputsFor, seeds: Sequence[int], workers: int = 1) -> list[bytes]:
    """One tape per seed in the order of `seeds`, whatever the number of worker processes.

    `inputs_for` must be picklable when `workers > 1`; each worker builds its own inputs so no
    city arrays travel between processes.
    """
    if workers < 1:
        raise ValueError("workers must be at least 1")
    if workers == 1:
        return [run_seed_bytes(inputs_for(seed)) for seed in seeds]
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as pool:
        futures = [pool.submit(_run_one, inputs_for, seed) for seed in seeds]
        return [future.result() for future in futures]


def _run_one(inputs_for: InputsFor, seed: int) -> bytes:
    return run_seed_bytes(inputs_for(seed))
