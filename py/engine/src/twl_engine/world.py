"""What never changes after a run starts: tables derived from the city, scenario and params."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from twl_engine.calendar import (
    DAY_TYPES,
    WEEKEND,
    WORKING_DAY,
    Calendar,
    build_calendar,
    plain_calendar,
)
from twl_engine.clock import Clock
from twl_engine.fixtures import ToyCity
from twl_engine.scenario import (
    CHANNELS,
    HOURS_PER_DAY,
    IN_APP,
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
    """Per campaign: expected impressions per tick in agent units, who they can land on and
    what each impression adds.

    A campaign whose presence differs by segment lands in two stages: `segment_share[c, day type,
    hour]` splits a tick's impressions over segments, then `segment_cum` (residents in
    `by_segment` order, cumulative targeting restarting at each `segment_start`) picks within a
    segment.
    """

    per_tick: NDArray[np.float64]
    cum_target: NDArray[np.float64]
    last_target: NDArray[np.int64]
    cap: NDArray[np.int64]
    gain: NDArray[np.float64]
    two_stage: NDArray[np.bool_]
    segment_share: NDArray[np.float64]
    by_segment: NDArray[np.int64]
    segment_start: NDArray[np.int64]
    segment_cum: NDArray[np.float64]
    segment_last: NDArray[np.int64]


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
    `avail = 0` and adds `satisfaction_r` (section 12). `occasion_probability` is [tick, segment]
    and `occasion_shared[tick]` says every segment has the same one; `segment` is per resident.
    """

    occasion_probability: NDArray[np.float64]
    occasion_shared: NDArray[np.bool_]
    segment: NDArray[np.int64]
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

    serviceable: NDArray[np.bool_]
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
    calendar: Calendar
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
    calendar = city_calendar(city, scenario, clock)
    affinity = rng.standard_normal(n) * params.affinity_sd
    repeat_taste = rng.standard_normal(n) * params.rho_sd
    return World(
        scenario=scenario,
        params=params,
        clock=clock,
        calendar=calendar,
        residents=n,
        paid=build_paid_plan(city, scenario, params, clock, calendar),
        talk=build_talk_graph(city, params, clock),
        choice=build_choice_model(city, scenario, params, clock, calendar, affinity, repeat_taste),
        supply=None if scenario.supply is None else build_supply_plan(city, scenario.supply),
        decay_per_tick=awareness_decay_per_tick(params, clock),
        affinity=affinity,
        repeat_taste=repeat_taste,
    )


def city_calendar(city: ToyCity, scenario: Scenario, clock: Clock) -> Calendar:
    return build_calendar(clock, scenario.start_date, city.timezone, city.weekend_days)


def awareness_decay_per_tick(params: Params, clock: Clock) -> float:
    """2^(-dt / tau), MODEL.md section 3; 1 when awareness does not decay."""
    if params.awareness_half_life_days is None:
        return 1.0
    return 2.0 ** (-clock.tick_minutes / MINUTES_PER_DAY / params.awareness_half_life_days)


def build_paid_plan(
    city: ToyCity,
    scenario: Scenario,
    params: Params,
    clock: Clock,
    calendar: Calendar | None = None,
) -> PaidPlan:
    """Campaign tables; `calendar` defaults to the city's calendar for the scenario."""
    calendar = city_calendar(city, scenario, clock) if calendar is None else calendar
    campaigns = scenario.campaigns
    n, count = city.residents.x.shape[0], len(campaigns)
    segments = len(params.kappa)
    segment = city.residents.segment.astype(np.int64)
    district = city.buildings.district[city.residents.building].astype(np.int64)
    agents_per_impression = params.paid_scale / float(city.residents.weight.mean())
    by_segment = np.argsort(segment, kind="stable")
    segment_start = np.searchsorted(segment[by_segment], np.arange(segments + 1))
    base_gain = params.impression_gain * np.asarray(params.kappa)[segment]
    per_tick = np.zeros((count, clock.ticks))
    cum_target = np.zeros((count, n))
    last_target = np.zeros(count, dtype=np.int64)
    gain = np.zeros((count, n))
    two_stage = np.zeros(count, dtype=np.bool_)
    segment_share = np.zeros((count, DAY_TYPES, HOURS_PER_DAY, segments))
    segment_cum = np.zeros((count, n))
    segment_last = np.zeros((count, segments), dtype=np.int64)
    for c, campaign in enumerate(campaigns):
        gain[c] = base_gain * channel_kappa(campaign, params, segments)[segment]
        target = _lookup(campaign, "segment_weights", segment) * _lookup(
            campaign, "district_weights", district
        )
        if not target.any():
            continue
        cum_target[c] = np.cumsum(target)
        last_target[c] = np.flatnonzero(target)[-1]
        presence = np.stack(
            [campaign_presence(campaign, params, segments, weekend=w) for w in (False, True)]
        )
        if campaign.channel == IN_APP and not presence.any():
            raise ValueError(
                f"in-app campaign {campaign.name!r} would show nowhere: in-app ads are seen in "
                "app sessions, and no segment has occasions in this run"
            )
        hourly = presence[:, 0]
        if not (presence == presence[:, :1]).all():
            two_stage[c] = True
            audience = np.bincount(segment, weights=target, minlength=segments)
            hourly = np.empty((DAY_TYPES, HOURS_PER_DAY))
            for day_type in (WORKING_DAY, WEEKEND):
                hourly[day_type], segment_share[c, day_type] = _split_by_segment(
                    audience, presence[day_type]
                )
            segment_cum[c], segment_last[c] = _cumulative_by_segment(
                target[by_segment], segment_start
            )
        per_tick[c] = tick_shares(campaign, clock, hourly, calendar) * (
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
        gain=gain,
        two_stage=two_stage,
        segment_share=segment_share,
        by_segment=by_segment,
        segment_start=segment_start,
        segment_cum=segment_cum,
        segment_last=segment_last,
    )


def channel_kappa(campaign: Campaign, params: Params, segments: int) -> NDArray[np.float64]:
    """The channel's responsiveness multiplier per segment; 1 until calibrated (D-034)."""
    row = params.channel_kappa[CHANNELS.index(campaign.channel)] if params.channel_kappa else ()
    if not row:
        return np.ones(segments)
    if len(row) != segments:
        raise ValueError(f"params.channel_kappa for {campaign.channel} needs {segments} values")
    return np.asarray(row)


def campaign_presence(
    campaign: Campaign, params: Params, segments: int, weekend: bool = False
) -> NDArray[np.float64]:
    """Hourly presence per segment: the campaign's own weights times the channel's (D-034).

    The channel's presence is 1 until calibrated, and for in-app ads it is the segment's
    occasion rate per hour for the day type, so in-app ads are seen when people order (D-036).
    """
    own = np.asarray(campaign.presence) if campaign.presence else np.ones(HOURS_PER_DAY)
    if campaign.channel == IN_APP:
        channel = occasions_per_hour(params, segments)[WEEKEND if weekend else WORKING_DAY]
    else:
        c = CHANNELS.index(campaign.channel)
        row = params.channel_presence[c] if params.channel_presence else ()
        if not row:
            return np.broadcast_to(own, (segments, HOURS_PER_DAY))
        if len(row) != segments:
            raise ValueError(
                f"params.channel_presence for {campaign.channel} needs {segments} rows"
            )
        channel = np.asarray(row)
    return own[None, :] * channel


def _split_by_segment(
    audience: NDArray[np.float64], presence: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Per hour: the summed targeting times presence, and each segment's share of it."""
    reach = audience[:, None] * presence
    hourly = reach.sum(axis=0)
    share = np.divide(
        reach.T, hourly[:, None], out=np.zeros_like(reach.T), where=hourly[:, None] > 0
    )
    return hourly, share


def _cumulative_by_segment(
    target: NDArray[np.float64], segment_start: NDArray[np.int64]
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Cumulative targeting restarting at each segment, and each segment's last targeted slot."""
    cum = np.zeros_like(target)
    last = np.zeros(segment_start.size - 1, dtype=np.int64)
    for s in range(segment_start.size - 1):
        lo, hi = int(segment_start[s]), int(segment_start[s + 1])
        cum[lo:hi] = np.cumsum(target[lo:hi])
        targeted = np.flatnonzero(target[lo:hi])
        last[s] = lo + (int(targeted[-1]) if targeted.size else 0)
    return cum, last


def tick_shares(
    campaign: Campaign,
    clock: Clock,
    hourly: NDArray[np.float64] | None = None,
    calendar: Calendar | None = None,
) -> NDArray[np.float64]:
    """Share of an active day's impressions in each tick: the hourly weights over that local
    day's active ticks.

    `hourly` is 24 weights, or 24 per day type; the campaign's own presence when none are given.
    Without a calendar days start at midnight of sim minute 0 and are all working days.
    """
    calendar = plain_calendar(clock) if calendar is None else calendar
    starts = np.arange(clock.ticks, dtype=np.int64) * clock.tick_minutes
    active = (starts >= campaign.start_minute) & (starts < campaign.end_minute)
    if hourly is None:
        hourly = np.asarray(campaign.presence) if campaign.presence else np.ones(HOURS_PER_DAY)
    by_day_type = np.broadcast_to(hourly, (DAY_TYPES, HOURS_PER_DAY))
    weight = np.where(active, by_day_type[calendar.day_type, calendar.hour], 0.0)
    day = calendar.day
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
    calendar: Calendar,
    affinity: NDArray[np.float64],
    repeat_taste: NDArray[np.float64],
) -> ChoiceModel:
    tier = city.residents.tier.astype(np.int64)
    beta = params.beta * np.asarray(params.beta_tier)[tier]
    price_term = beta * scenario.price / scenario.ref_price
    segment = city.residents.segment.astype(np.int64)
    alpha = (
        params.alpha_0
        + np.asarray(params.alpha_segment)[segment]
        + np.asarray(params.alpha_tier)[tier]
    )
    probability = occasion_probability_per_tick(params, clock, calendar)
    return ChoiceModel(
        occasion_probability=probability,
        occasion_shared=(probability == probability[:, :1]).all(axis=1),
        segment=segment,
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
        serviceable=own_store != NO_STORE,
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


def occasion_tables(
    params: Params, segments: int
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Per day type and segment: the daily occasion rate and the 24 hourly weights (D-036)."""
    _require_occasion_tables(params, segments)
    working = params.segment_occasion_profile or ((),) * segments
    weekend = params.weekend_occasion_profile or ((),) * segments
    rates = np.empty((DAY_TYPES, segments))
    rates[WORKING_DAY] = params.segment_occasions_per_day or params.occasions_per_day
    rates[WEEKEND] = rates[WORKING_DAY] * np.asarray(params.weekend_occasion_factor or 1.0)
    profiles = np.empty((DAY_TYPES, segments, HOURS_PER_DAY))
    for s in range(segments):
        profiles[WORKING_DAY, s] = working[s] or params.occasion_profile
        profiles[WEEKEND, s] = weekend[s] or profiles[WORKING_DAY, s]
    silent = (profiles.sum(axis=2) <= 0.0) & (rates > 0.0)
    if silent.any():
        day_type, s = np.argwhere(silent)[0]
        day = "weekend" if day_type == WEEKEND else "working-day"
        raise ValueError(f"segment {s} has occasions but its {day} profile sums to zero")
    return rates, profiles


def occasions_per_hour(params: Params, segments: int) -> NDArray[np.float64]:
    """[day type, segment, hour]: omega_s, the day type's daily rate spread over the hours by the
    profile, `daily / 24 * profile[h] / mean(profile)` (MODEL.md section 4, D-036)."""
    rates, profiles = occasion_tables(params, segments)
    per_hour = np.zeros((DAY_TYPES, segments, HOURS_PER_DAY))
    for day_type in (WORKING_DAY, WEEKEND):
        for s in range(segments):
            if rates[day_type, s] > 0.0:
                profile = profiles[day_type, s]
                per_hour[day_type, s] = (
                    rates[day_type, s] / HOURS_PER_DAY * profile / profile.mean()
                )
    return per_hour


def occasion_probability_per_tick(
    params: Params, clock: Clock, calendar: Calendar | None = None
) -> NDArray[np.float64]:
    """[tick, segment]: 1 - exp(-omega_s(day type, hour) dt)."""
    calendar = plain_calendar(clock) if calendar is None else calendar
    segments = len(params.kappa)
    per_hour = occasions_per_hour(params, segments)
    by_tick = per_hour[calendar.day_type[:, None], np.arange(segments), calendar.hour[:, None]]
    return -np.expm1(-by_tick * clock.tick_minutes / 60.0)


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


def _require_occasion_tables(params: Params, segments: int) -> None:
    for name in (
        "segment_occasions_per_day",
        "segment_occasion_profile",
        "weekend_occasion_factor",
        "weekend_occasion_profile",
    ):
        table = getattr(params, name)
        if table and len(table) != segments:
            raise ValueError(
                f"params.{name} has {len(table)} entries; it needs one per segment "
                f"({segments}, the length of kappa) or none"
            )
