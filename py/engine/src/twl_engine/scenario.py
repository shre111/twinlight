import dataclasses
import json
import math
import operator
from dataclasses import dataclass

from blake3 import blake3

MINUTES_PER_DAY = 1440
HOURS_PER_DAY = 24
TIE_KINDS = 4
WOM_MIXINGS = ("graph", "uniform")
_TOD_AM = (0.25, 0.12, 0.06, 0.04, 0.04, 0.06, 0.15, 0.3, 0.4, 0.45, 0.5, 0.6)
_TOD_PM = (0.75, 0.8, 0.65, 0.6, 0.7, 0.85, 1.0, 1.35, 1.55, 1.45, 1.1, 0.6)
PROTOTYPE_TOD = _TOD_AM + _TOD_PM


@dataclass(frozen=True)
class Campaign:
    """A paid channel (MODEL.md sections 3 and 11), active for ticks starting in [start, end).

    Each active day buys `budget_per_day / cpm * 1000` impressions, spread over that day's active
    ticks by `presence` (24 hourly weights; empty is flat). `segment_weights` and
    `district_weights` are indexed by segment and district id; empty means 1 for every one.
    """

    name: str
    start_minute: int
    end_minute: int
    budget_per_day: float
    cpm: float
    segment_weights: tuple[float, ...] = ()
    district_weights: tuple[float, ...] = ()
    frequency_cap: int | None = 3
    presence: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        _coerce(
            self,
            floats=("budget_per_day", "cpm"),
            vectors=("segment_weights", "district_weights", "presence"),
        )
        _require_ints(self, ("start_minute", "end_minute"))
        if not 0 <= self.start_minute < self.end_minute:
            raise ValueError(f"campaign {self.name!r} needs 0 <= start_minute < end_minute")
        if self.budget_per_day < 0 or not math.isfinite(self.budget_per_day):
            raise ValueError(f"campaign {self.name!r} budget_per_day must be finite and >= 0")
        if self.cpm <= 0 or not math.isfinite(self.cpm):
            raise ValueError(f"campaign {self.name!r} cpm must be finite and positive")
        if self.frequency_cap is not None:
            _require_ints(self, ("frequency_cap",))
            if self.frequency_cap < 1:
                raise ValueError(f"campaign {self.name!r} frequency_cap must be at least 1")
        if self.presence and len(self.presence) != HOURS_PER_DAY:
            raise ValueError(f"campaign {self.name!r} presence needs 24 hourly weights")
        for field in ("segment_weights", "district_weights", "presence"):
            _require_weights(f"campaign {self.name!r} {field}", getattr(self, field))

    @property
    def impressions_per_day(self) -> float:
        return self.budget_per_day / self.cpm * 1000.0


@dataclass(frozen=True)
class Scenario:
    """What is launched and when: the price and the paid campaigns. P-013 adds the stock."""

    days: int = 30
    tick_minutes: int = 15
    price: float = 49.0
    ref_price: float = 49.0
    campaigns: tuple[Campaign, ...] = ()

    def __post_init__(self) -> None:
        _coerce(self, floats=("price", "ref_price"), vectors=())
        _require_ints(self, ("days", "tick_minutes"))
        object.__setattr__(self, "campaigns", tuple(self.campaigns))
        if self.days < 1:
            raise ValueError("a scenario runs for at least one day")
        if self.tick_minutes < 1:
            raise ValueError("tick_minutes must be at least 1")
        if self.price < 0 or not math.isfinite(self.price):
            raise ValueError("price must be finite and >= 0")
        if self.ref_price <= 0 or not math.isfinite(self.ref_price):
            raise ValueError("ref_price must be finite and positive")
        if not all(isinstance(campaign, Campaign) for campaign in self.campaigns):
            raise TypeError("campaigns must be Campaign instances")

    @property
    def sim_minutes(self) -> int:
        return self.days * MINUTES_PER_DAY

    def canonical_json(self) -> bytes:
        return _canonical(dataclasses.asdict(self))

    def hash(self) -> bytes:
        return blake3(self.canonical_json()).digest()


@dataclass(frozen=True)
class Params:
    """Per-run model parameters with the MODEL.md section 11 defaults; rates are per hour."""

    theta_aware: float = 1.0
    impression_gain: float = 0.6
    talk_gain: float = 0.9
    hearsay_factor: float = 0.15
    talk_decay_hours: float | None = 72.0
    awareness_half_life_days: float | None = 4.0
    kappa: tuple[float, ...] = (1.0, 0.9, 0.62, 0.38, 0.2)
    q: float = 0.02
    talk_kind: tuple[float, ...] = (1.0, 0.6, 1.0, 0.5)
    occasions_per_day: float = 3.0
    occasion_profile: tuple[float, ...] = PROTOTYPE_TOD
    alpha_0: float = -3.0
    alpha_segment: tuple[float, ...] = (
        math.log(1.35),
        math.log(1.2),
        math.log(0.9),
        math.log(0.6),
        math.log(0.38),
    )
    affinity_sd: float = 0.3
    gamma: float = 2.5
    beta: float = 1.0
    delta: float = 1.0
    alpha_rep: float = -2.3
    rho_sd: float = 0.3
    paid_scale: float = 1.0
    wom_mixing: str = "graph"
    order_on_awareness: bool = False

    def __post_init__(self) -> None:
        _coerce(
            self,
            floats=(
                "theta_aware",
                "impression_gain",
                "talk_gain",
                "hearsay_factor",
                "q",
                "occasions_per_day",
                "alpha_0",
                "affinity_sd",
                "gamma",
                "beta",
                "delta",
                "alpha_rep",
                "rho_sd",
                "paid_scale",
            ),
            vectors=("kappa", "talk_kind", "occasion_profile", "alpha_segment"),
            optional_floats=("talk_decay_hours", "awareness_half_life_days"),
        )
        object.__setattr__(self, "order_on_awareness", bool(self.order_on_awareness))
        if self.theta_aware <= 0:
            raise ValueError("theta_aware must be positive")
        for name in (
            "impression_gain",
            "talk_gain",
            "hearsay_factor",
            "q",
            "occasions_per_day",
            "affinity_sd",
            "rho_sd",
            "paid_scale",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0")
        for name in ("talk_decay_hours", "awareness_half_life_days"):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive, or None to switch it off")
        if not self.kappa or not self.alpha_segment:
            raise ValueError("kappa and alpha_segment need a value per segment")
        if len(self.talk_kind) != TIE_KINDS:
            raise ValueError(f"talk_kind needs {TIE_KINDS} values, one per tie kind")
        if len(self.occasion_profile) != HOURS_PER_DAY:
            raise ValueError("occasion_profile needs 24 hourly weights")
        for name in ("kappa", "talk_kind", "occasion_profile"):
            _require_weights(name, getattr(self, name))
        if self.occasions_per_day > 0 and sum(self.occasion_profile) <= 0:
            raise ValueError("occasion_profile must have a positive sum")
        if self.wom_mixing not in WOM_MIXINGS:
            raise ValueError(f"wom_mixing must be one of {WOM_MIXINGS}")

    def canonical_json(self) -> bytes:
        return _canonical(dataclasses.asdict(self))

    def hash(self) -> bytes:
        return blake3(self.canonical_json()).digest()


def _coerce(
    instance: object,
    floats: tuple[str, ...],
    vectors: tuple[str, ...],
    optional_floats: tuple[str, ...] = (),
) -> None:
    """Store floats as floats and vectors as float tuples, so equal values hash equally."""
    for name in floats:
        object.__setattr__(instance, name, float(getattr(instance, name)))
    for name in optional_floats:
        value = getattr(instance, name)
        object.__setattr__(instance, name, None if value is None else float(value))
    for name in vectors:
        object.__setattr__(instance, name, tuple(float(v) for v in getattr(instance, name)))


def _require_ints(instance: object, names: tuple[str, ...]) -> None:
    for name in names:
        value = getattr(instance, name)
        if isinstance(value, bool):
            raise TypeError(f"{name} must be an integer, got {value!r}")
        object.__setattr__(instance, name, operator.index(value))


def _require_weights(label: str, values: tuple[float, ...]) -> None:
    if any(v < 0 or not math.isfinite(v) for v in values):
        raise ValueError(f"{label} must be finite and >= 0")


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


EMPTY = Scenario()
