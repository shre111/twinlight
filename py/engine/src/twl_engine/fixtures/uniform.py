import numpy as np

from twl_engine.fixtures.types import (
    NO_PERSONA,
    TIER_MIDDLE,
    Buildings,
    District,
    Residents,
    River,
    Roads,
    Stores,
    Ties,
    ToyCity,
)


def build_uniform_city(residents: int = 50_000) -> ToyCity:
    """Identical residents in one building and district, no ties: the Bass-limit city (D-027)."""
    if residents < 1:
        raise ValueError("a uniform city needs at least one resident")
    one = np.ones(1, dtype=np.float32)
    zero = np.zeros(1, dtype=np.float32)
    no_lines = np.zeros(1, dtype=np.uint32)
    no_points = np.zeros(0, dtype=np.float32)
    no_kinds = np.zeros(0, dtype=np.uint8)
    return ToyCity(
        seed=0,
        buildings=Buildings(
            x=zero,
            y=zero,
            width=one,
            depth=one,
            height=one,
            rotation=zero,
            district=np.zeros(1, dtype=np.uint16),
        ),
        roads=Roads(offsets=no_lines, x=no_points, y=no_points, kind=no_kinds),
        river=River(offsets=no_lines, x=no_points, y=no_points, kind=no_kinds),
        districts=(District(id=0, name="Uniform", x=0.0, y=0.0, east=False),),
        residents=Residents(
            x=np.zeros(residents, dtype=np.float32),
            y=np.zeros(residents, dtype=np.float32),
            z=np.zeros(residents, dtype=np.float32),
            building=np.zeros(residents, dtype=np.uint32),
            hex=np.zeros(residents, dtype=np.uint32),
            segment=np.zeros(residents, dtype=np.uint16),
            persona_ref=np.full(residents, NO_PERSONA, dtype=np.uint32),
            weight=np.ones(residents, dtype=np.float32),
            tier=np.full(residents, TIER_MIDDLE, dtype=np.uint8),
        ),
        ties=Ties(
            offsets=np.zeros(residents + 1, dtype=np.uint32),
            targets=np.zeros(0, dtype=np.uint32),
            kind=np.zeros(0, dtype=np.uint8),
            strength=np.zeros(0, dtype=np.float32),
        ),
        stores=Stores(
            x=zero,
            y=zero,
            hex=np.zeros(1, dtype=np.uint32),
            radius_m=np.full(1, 900.0, dtype=np.float32),
            district=np.zeros(1, dtype=np.uint16),
        ),
    )
