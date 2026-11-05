"""What never changes after a run starts: tables derived from the city, scenario and params."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from twl_engine.clock import Clock
from twl_engine.fixtures import ToyCity
from twl_engine.scenario import HOURS_PER_DAY, MINUTES_PER_DAY, Campaign, Params, Scenario

NO_CAP = -1


@dataclass(frozen=True, eq=False)
class PaidPlan:
    """Per campaign: expected impressions per tick in agent units and who they can land on."""

    per_tick: NDArray[np.float64]
    cum_target: NDArray[np.float64]
    last_target: NDArray[np.int64]
    cap: NDArray[np.int64]
    gain: NDArray[np.float64]


@dataclass(frozen=True, eq=False)
class World:
    scenario: Scenario
    params: Params
    clock: Clock
    residents: int
    paid: PaidPlan
    decay_per_tick: float
    affinity: NDArray[np.float64]
    repeat_taste: NDArray[np.float64]


def build_world(
    city: ToyCity, scenario: Scenario, params: Params, clock: Clock, rng: np.random.Generator
) -> World:
    """Derive the run's static tables; `rng` is the `params` stream (a_r, then rho_r)."""
    n = city.residents.x.shape[0]
    _require_segment_tables(params, int(city.residents.segment.max()))
    return World(
        scenario=scenario,
        params=params,
        clock=clock,
        residents=n,
        paid=build_paid_plan(city, scenario, params, clock),
        decay_per_tick=awareness_decay_per_tick(params, clock),
        affinity=rng.standard_normal(n) * params.affinity_sd,
        repeat_taste=rng.standard_normal(n) * params.rho_sd,
    )


def awareness_decay_per_tick(params: Params, clock: Clock) -> float:
    """2^(-dt / tau), MODEL.md section 3; 1 when awareness does not decay."""
    if params.awareness_half_life_days is None:
        return 1.0
    return 2.0 ** (-clock.tick_minutes / MINUTES_PER_DAY / params.awareness_half_life_days)


def build_paid_plan(city: ToyCity, scenario: Scenario, params: Params, clock: Clock) -> PaidPlan:
    campaigns = scenario.campaigns
    n = city.residents.x.shape[0]
    segment = city.residents.segment.astype(np.int64)
    district = city.buildings.district[city.residents.building].astype(np.int64)
    agents_per_impression = params.paid_scale / float(city.residents.weight.mean())
    per_tick = np.zeros((len(campaigns), clock.ticks))
    cum_target = np.zeros((len(campaigns), n))
    last_target = np.zeros(len(campaigns), dtype=np.int64)
    for c, campaign in enumerate(campaigns):
        target = _lookup(campaign, "segment_weights", segment) * _lookup(
            campaign, "district_weights", district
        )
        if not target.any():
            continue
        cum_target[c] = np.cumsum(target)
        last_target[c] = np.flatnonzero(target)[-1]
        per_tick[c] = tick_shares(campaign, clock) * (
            campaign.impressions_per_day * agents_per_impression
        )
    return PaidPlan(
        per_tick=per_tick,
        cum_target=cum_target,
        last_target=last_target,
        cap=np.array(
            [NO_CAP if c.frequency_cap is None else c.frequency_cap for c in campaigns],
            dtype=np.int64,
        ),
        gain=params.impression_gain * np.asarray(params.kappa)[segment],
    )


def tick_shares(campaign: Campaign, clock: Clock) -> NDArray[np.float64]:
    """Share of an active day's impressions in each tick: presence over that day's active ticks."""
    starts = np.arange(clock.ticks, dtype=np.int64) * clock.tick_minutes
    active = (starts >= campaign.start_minute) & (starts < campaign.end_minute)
    hour = (starts % MINUTES_PER_DAY) // 60
    presence = np.asarray(campaign.presence) if campaign.presence else np.ones(HOURS_PER_DAY)
    weight = np.where(active, presence[hour], 0.0)
    day = starts // MINUTES_PER_DAY
    day_total = np.bincount(day, weights=weight)[day]
    return np.divide(weight, day_total, out=np.zeros(clock.ticks), where=day_total > 0)


def _lookup(campaign: Campaign, field: str, ids: NDArray[np.int64]) -> NDArray[np.float64]:
    weights: tuple[float, ...] = getattr(campaign, field)
    if not weights:
        return np.ones(ids.shape[0])
    if int(ids.max()) >= len(weights):
        raise ValueError(
            f"campaign {campaign.name!r} {field} has {len(weights)} values; "
            f"the city needs {int(ids.max()) + 1}"
        )
    return np.asarray(weights)[ids]


def _require_segment_tables(params: Params, top_segment: int) -> None:
    for name in ("kappa", "alpha_segment"):
        if top_segment >= len(getattr(params, name)):
            raise ValueError(f"params.{name} has no value for segment {top_segment}")
