"""What never changes after a run starts: tables derived from the city, scenario and params."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from twl_engine.clock import Clock
from twl_engine.fixtures import ToyCity
from twl_engine.scenario import (
    HOURS_PER_DAY,
    MINUTES_PER_DAY,
    Campaign,
    Params,
    Scenario,
    Supply,
)

NO_CAP = -1
NO_STORE = -1


@dataclass(frozen=True, eq=False)
class PaidPlan:
    """Per campaign: expected impressions per tick in agent units and who they can land on."""

    per_tick: NDArray[np.float64]
    cum_target: NDArray[np.float64]
    last_target: NDArray[np.int64]
    cap: NDArray[np.int64]
    gain: NDArray[np.float64]


@dataclass(frozen=True, eq=False)
class TalkGraph:
    """Who a talker reaches. `pull` is a talker's summed tie weight `w_uv * talk_kind`, so a
    talker emits `q * pull * dt` expected talks per tick before decay and the hearsay factor.
    With uniform mixing every talker has pull 1 and listeners are drawn from all residents.
    """

    uniform: bool
    talks_per_pull: float
    hearsay: float
    floor: float
    decay_per_minute: float
    pull: NDArray[np.float64]
    offsets: NDArray[np.int64]
    targets: NDArray[np.int64]
    tie_weight: NDArray[np.float64]
    tie_gain: NDArray[np.float64]
    uniform_gain: float


@dataclass(frozen=True, eq=False)
class ChoiceModel:
    """Purchase decisions (MODEL.md section 4) with everything constant per resident folded in.

    `first_logit` is U without the awareness term, `repeat_logit` the whole repeat utility, both
    at `avail = 1` and `satisfaction = 0`; with supply on the tick takes `delta` off where
    `avail = 0` and adds `satisfaction_r` (section 12). `occasion_probability` is per tick.
    """

    occasion_probability: NDArray[np.float64]
    first_logit: NDArray[np.float64]
    repeat_logit: NDArray[np.float64]
    gamma: float
    delta: float


@dataclass(frozen=True, eq=False)
class SupplyPlan:
    """Who serves whom and what each store holds (MODEL.md section 12).

    `own_store` is the nearest store whose radius contains the resident, `spill_store` the
    nearest other store within `spill_radius_factor` times its own radius; `NO_STORE` where there
    is none. Distances are straight-line metres. Stock levels are whole agent orders: the
    scenario's rate times the store's catchment, rounded.
    """

    own_store: NDArray[np.int64]
    spill_store: NDArray[np.int64]
    own_distance: NDArray[np.float64]
    spill_distance: NDArray[np.float64]
    catchment: NDArray[np.int64]
    initial_stock: NDArray[np.int64]
    reorder_point: NDArray[np.int64]
    order_up_to: NDArray[np.int64]

    @property
    def stores(self) -> int:
        return self.catchment.shape[0]


@dataclass(frozen=True, eq=False)
class World:
    scenario: Scenario
    params: Params
    clock: Clock
    residents: int
    paid: PaidPlan
    talk: TalkGraph
    choice: ChoiceModel
    supply: SupplyPlan | None
    decay_per_tick: float
    affinity: NDArray[np.float64]
    repeat_taste: NDArray[np.float64]


def build_world(
    city: ToyCity, scenario: Scenario, params: Params, clock: Clock, rng: np.random.Generator
) -> World:
    """Derive the run's static tables; `rng` is the `params` stream (a_r, then rho_r)."""
    n = city.residents.x.shape[0]
    _require_segment_tables(params, int(city.residents.segment.max()))
    affinity = rng.standard_normal(n) * params.affinity_sd
    repeat_taste = rng.standard_normal(n) * params.rho_sd
    return World(
        scenario=scenario,
        params=params,
        clock=clock,
        residents=n,
        paid=build_paid_plan(city, scenario, params, clock),
        talk=build_talk_graph(city, params, clock),
        choice=build_choice_model(city, scenario, params, clock, affinity, repeat_taste),
        supply=None if scenario.supply is None else build_supply_plan(city, scenario.supply),
        decay_per_tick=awareness_decay_per_tick(params, clock),
        affinity=affinity,
        repeat_taste=repeat_taste,
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


def build_talk_graph(city: ToyCity, params: Params, clock: Clock) -> TalkGraph:
    n = city.residents.x.shape[0]
    ties = city.ties
    uniform = params.wom_mixing == "uniform"
    tie_weight = ties.strength.astype(np.float64) * np.asarray(params.talk_kind)[ties.kind]
    owner = np.repeat(np.arange(n), np.diff(ties.offsets.astype(np.int64)))
    graph_pull = np.bincount(owner, weights=tie_weight, minlength=n)
    decay_hours = params.talk_decay_hours
    return TalkGraph(
        uniform=uniform,
        talks_per_pull=params.q * clock.tick_minutes / 60.0,
        hearsay=params.hearsay_factor,
        floor=params.talk_floor,
        decay_per_minute=0.0 if decay_hours is None else 1.0 / (decay_hours * 60.0),
        pull=np.full(n, 1.0 if n > 1 else 0.0) if uniform else graph_pull,
        offsets=ties.offsets.astype(np.int64),
        targets=ties.targets.astype(np.int64),
        tie_weight=tie_weight,
        tie_gain=params.talk_gain * ties.strength.astype(np.float64),
        uniform_gain=params.talk_gain,
    )


def build_choice_model(
    city: ToyCity,
    scenario: Scenario,
    params: Params,
    clock: Clock,
    affinity: NDArray[np.float64],
    repeat_taste: NDArray[np.float64],
) -> ChoiceModel:
    price_term = params.beta * scenario.price / scenario.ref_price
    segment = city.residents.segment.astype(np.int64)
    alpha = params.alpha_0 + np.asarray(params.alpha_segment)[segment]
    return ChoiceModel(
        occasion_probability=occasion_probability_per_tick(params, clock),
        first_logit=alpha + affinity - price_term + params.delta,
        repeat_logit=params.alpha_rep + repeat_taste - price_term,
        gamma=params.gamma,
        delta=params.delta,
    )


def build_supply_plan(city: ToyCity, supply: Supply) -> SupplyPlan:
    stores = city.stores
    distance = np.hypot(
        city.residents.x.astype(np.float64)[:, None] - stores.x.astype(np.float64)[None, :],
        city.residents.y.astype(np.float64)[:, None] - stores.y.astype(np.float64)[None, :],
    )
    radius = stores.radius_m.astype(np.float64)[None, :]
    own_store, own_distance = nearest_allowed_store(distance, distance <= radius)
    other = np.arange(radius.shape[1])[None, :] != own_store[:, None]
    in_spill_reach = distance <= supply.spill_radius_factor * radius
    spill_store, spill_distance = nearest_allowed_store(
        distance, other & in_spill_reach & (own_store != NO_STORE)[:, None]
    )
    catchment = np.bincount(own_store[own_store != NO_STORE], minlength=radius.shape[1])
    return SupplyPlan(
        own_store=own_store,
        spill_store=spill_store,
        own_distance=own_distance,
        spill_distance=spill_distance,
        catchment=catchment.astype(np.int64),
        initial_stock=stock_level(supply.initial_stock, catchment),
        reorder_point=stock_level(supply.reorder_point, catchment),
        order_up_to=stock_level(supply.order_up_to, catchment),
    )


def nearest_allowed_store(
    distance: NDArray[np.float64], allowed: NDArray[np.bool_]
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    """Per resident the nearest allowed store (lowest index on a tie) and its distance."""
    residents, stores = distance.shape
    if stores == 0:
        return np.full(residents, NO_STORE, dtype=np.int64), np.zeros(residents)
    reach = np.where(allowed, distance, np.inf)
    nearest = np.argmin(reach, axis=1).astype(np.int64)
    best = reach[np.arange(residents), nearest]
    found = np.isfinite(best)
    return np.where(found, nearest, NO_STORE), np.where(found, best, 0.0)


def stock_level(rate: float, catchment: NDArray[np.int64]) -> NDArray[np.int64]:
    return np.rint(rate * catchment).astype(np.int64)


def occasion_probability_per_tick(params: Params, clock: Clock) -> NDArray[np.float64]:
    """1 - exp(-omega(slot) dt), with omega averaging `occasions_per_day` over the day."""
    if params.occasions_per_day <= 0.0:
        return np.zeros(clock.ticks)
    profile = np.asarray(params.occasion_profile)
    per_hour = params.occasions_per_day / HOURS_PER_DAY * profile / profile.mean()
    starts = np.arange(clock.ticks, dtype=np.int64) * clock.tick_minutes
    hour = (starts % MINUTES_PER_DAY) // 60
    return -np.expm1(-per_hour[hour] * clock.tick_minutes / 60.0)


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
