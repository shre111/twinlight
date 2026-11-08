import numpy as np
import pytest
from numpy.typing import NDArray

from twl_engine.fixtures import build_uniform_city
from twl_engine.goldens import prototype_inputs
from twl_engine.montecarlo import run_seeds
from twl_engine.run import RunInputs, run_seed
from twl_engine.scenario import Campaign, Params, Scenario
from twl_tape.format import NEVER, Tape
from twl_tape.reader import read_tape

pytestmark = pytest.mark.anchor

N = 50_000
DAYS = 30
TICKS_PER_DAY = 96
SEEDS = range(1, 9)
BASS_P = 0.008
BASS_Q = 0.30
PAID_ONLY_P = 0.06
PARITY_DAYS = (6, 13, 20, 30)
# Days 6 and 13 are fitted, day 30 informed the model choice (D-029), day 20 is held out.
PROTOTYPE_TARGETS = {6: (23.0, 4.5), 13: (41.7, 10.8), 20: (51.9, 16.1), 30: (63.1, 23.0)}
PARITY_TOLERANCE = 3.0
WEST_EAST_TARGETS = (33.5, 12.0)


def reduced_params(q_per_day: float) -> Params:
    return Params(
        impression_gain=1.0,
        kappa=(1.0,),
        talk_gain=1.0,
        hearsay_factor=0.0,
        talk_decay_hours=None,
        awareness_half_life_days=None,
        q=q_per_day / 24,
        wom_mixing="uniform",
        order_on_awareness=True,
        affinity_sd=0.0,
        rho_sd=0.0,
        alpha_segment=(0.0,),
        occasions_per_day=0.0,
        occasion_profile=(1.0,) * 24,
    )


def adoption_curve(p_per_day: float, q_per_day: float) -> NDArray[np.float64]:
    city = build_uniform_city(N)
    campaign = Campaign(
        name="flat",
        start_minute=0,
        end_minute=DAYS * 1440,
        budget_per_day=p_per_day * N,
        cpm=1000.0,
        frequency_cap=None,
        channel="social",
    )
    scenario = Scenario(days=DAYS, campaigns=(campaign,))
    params = reduced_params(q_per_day)
    ends = [TICKS_PER_DAY * d - 1 for d in range(1, DAYS + 1)]
    curves = []
    for seed in SEEDS:
        inputs = RunInputs(
            city=city, city_pack_hash=bytes(32), scenario=scenario, params=params, seed=seed
        )
        adopted = run_seed(inputs).aggregates.adopted
        curves.append(adopted[ends].astype(np.float64) / N)
    return np.mean(curves, axis=0)


def bass(p: float, q: float, d: NDArray[np.float64]) -> NDArray[np.float64]:
    e = np.exp(-(p + q) * d)
    return (1 - e) / (1 + (q / p) * e)


def worst_day(mean: NDArray[np.float64], want: NDArray[np.float64]) -> str:
    gap = np.abs(mean - want)
    d = int(np.argmax(gap))
    return f"worst gap {gap[d]:.4f} on day {d + 1} (got {mean[d]:.4f}, want {want[d]:.4f})"


def test_adoption_follows_the_bass_curve_within_one_point() -> None:
    mean = adoption_curve(BASS_P, BASS_Q)
    days = np.arange(1, DAYS + 1, dtype=np.float64)
    want = bass(BASS_P, BASS_Q, days)
    assert np.all(np.abs(mean - want) <= 0.01), worst_day(mean, want)
    half = int(np.argmax(mean >= 0.5)) + 1
    assert mean.max() >= 0.5
    assert 10 <= half <= 15, f"50% point on day {half}"


def test_paid_only_adoption_follows_one_minus_exp_within_half_a_point() -> None:
    mean = adoption_curve(PAID_ONLY_P, 0.0)
    days = np.arange(1, DAYS + 1, dtype=np.float64)
    want = 1 - np.exp(-PAID_ONLY_P * days)
    assert np.all(np.abs(mean - want) <= 0.005), worst_day(mean, want)


@pytest.fixture(scope="module")
def prototype_tapes() -> list[Tape]:
    return [read_tape(data) for data in run_seeds(prototype_inputs, SEEDS, workers=8)]


def share_table(tapes: list[Tape]) -> tuple[dict[int, tuple[float, float]], str]:
    n = len(tapes[0].residents.aware_at)
    got: dict[int, tuple[float, float]] = {}
    rows = ["day  heard (got/want)  ordered (got/want)"]
    for day in PARITY_DAYS:
        tick = TICKS_PER_DAY * day - 1
        heard = 100 * float(np.mean([t.aggregates.aware[tick] for t in tapes])) / n
        ordered = 100 * float(np.mean([t.aggregates.adopted[tick] for t in tapes])) / n
        got[day] = (heard, ordered)
        want = PROTOTYPE_TARGETS[day]
        rows.append(f"{day:>3}  {heard:6.1f}/{want[0]:<6.1f}  {ordered:6.1f}/{want[1]:<6.1f}")
    return got, "\n".join(rows)


@pytest.mark.parametrize("day", PARITY_DAYS)
def test_prototype_launch_shares_are_within_three_points(
    day: int, prototype_tapes: list[Tape]
) -> None:
    got, table = share_table(prototype_tapes)
    heard, ordered = got[day]
    want_heard, want_ordered = PROTOTYPE_TARGETS[day]
    assert abs(heard - want_heard) <= PARITY_TOLERANCE, f"heard off on day {day}\n{table}"
    assert abs(ordered - want_ordered) <= PARITY_TOLERANCE, f"ordered off on day {day}\n{table}"


@pytest.mark.xfail(
    strict=False, reason="tier effects are neutral until P-062 calibrates them (D-037)"
)
def test_prototype_launch_orders_split_west_and_east(prototype_tapes: list[Tape]) -> None:
    city = prototype_inputs(1).city
    east_ids = {d.id for d in city.districts if d.east}
    district = city.buildings.district[city.residents.building]
    east = np.isin(district, list(east_ids))
    west_share = np.mean(
        [100 * np.mean((t.residents.first_order_at != NEVER)[~east]) for t in prototype_tapes]
    )
    east_share = np.mean(
        [100 * np.mean((t.residents.first_order_at != NEVER)[east]) for t in prototype_tapes]
    )
    want_west, want_east = WEST_EAST_TARGETS
    assert abs(west_share - want_west) <= 5.0, f"west {west_share:.1f} vs {want_west}"
    assert abs(east_share - want_east) <= 5.0, f"east {east_share:.1f} vs {want_east}"
