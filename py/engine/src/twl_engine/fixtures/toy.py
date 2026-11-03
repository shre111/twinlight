from twl_engine.fixtures.city import build_layout
from twl_engine.fixtures.geometry import toy_districts
from twl_engine.fixtures.residents import place_residents
from twl_engine.fixtures.stores import place_stores
from twl_engine.fixtures.ties import build_ties
from twl_engine.fixtures.types import ToyCity
from twl_engine.rng import spawn_streams

STREAMS = ("city", "residents", "stores", "ties")


def build_toy_city(seed: int = 49, residents: int = 48_000, stores: int = 14) -> ToyCity:
    """The prototype's procedural city (design/prototype/toy-sim.mjs) in pack coordinates."""
    if residents < 1 or stores < 1:
        raise ValueError("a toy city needs at least one resident and one store")
    streams = spawn_streams(seed, STREAMS)
    layout = build_layout(streams["city"])
    people = place_residents(layout.buildings, residents, streams["residents"])
    return ToyCity(
        seed=seed,
        buildings=layout.buildings,
        roads=layout.roads,
        river=layout.river,
        districts=toy_districts(),
        residents=people,
        ties=build_ties(people, streams["ties"]),
        stores=place_stores(people, layout.corners_x, layout.corners_y, stores, streams["stores"]),
    )
