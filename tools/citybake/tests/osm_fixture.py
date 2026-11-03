import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from xml.sax.saxutils import quoteattr

import numpy as np
import numpy.typing as npt

from twl_citybake.config import BBox
from twl_citybake.projection import LocalProjection

BBOX = BBox(72.57 - 0.00975, 23.02 - 0.00905, 72.57 + 0.00975, 23.02 + 0.00905)

Tags = Mapping[str, str]

ROAD_NODES: dict[int, tuple[float, float]] = {
    5006: (-900.0, 0.0),
    5005: (-450.0, 50.0),
    5004: (-100.0, 0.0),
    5003: (100.0, 0.0),
    5002: (450.0, 0.0),
    5001: (900.0, 0.0),
    5010: (-450.0, -400.0),
    5011: (450.0, -600.0),
    5012: (450.0, 600.0),
    5013: (480.0, 300.0),
    5014: (520.0, 300.0),
    5015: (900.0, 300.0),
    5016: (900.0, -300.0),
    5017: (-800.0, -900.0),
    5018: (-700.0, -900.0),
    5019: (-1300.0, 0.0),
}

ROAD_WAYS: dict[int, tuple[list[int], dict[str, str]]] = {
    100: ([5006, 5005, 5004], {"highway": "primary"}),
    101: ([5004, 5003], {"highway": "primary", "bridge": "yes"}),
    102: ([5003, 5002, 5001], {"highway": "primary"}),
    110: ([5005, 5010], {"highway": "residential", "bridge": "no"}),
    120: ([5011, 5002, 5013, 5012], {"highway": "secondary"}),
    130: ([5001, 5015], {"highway": "residential", "oneway": "yes"}),
    131: ([5001, 5016], {"highway": "residential", "oneway": "-1"}),
    140: ([5017, 5018], {"highway": "service"}),
    150: ([5013, 5014], {"highway": "footway"}),
    160: ([5006, 5019], {"highway": "tertiary"}),
}

KEPT_ROAD_NODE_IDS = [5001, 5002, 5003, 5004, 5005, 5006, 5010, 5011, 5012, 5015, 5016, 5017, 5018]


class OsmBuilder:
    def __init__(self, projection: LocalProjection) -> None:
        self.projection = projection
        self.nodes: dict[int, tuple[float, float, Tags]] = {}
        self.ways: dict[int, tuple[list[int], Tags]] = {}
        self.relations: dict[int, tuple[list[tuple[str, int, str]], Tags]] = {}
        self._next_node = 100_000

    def node(self, x: float, y: float, tags: Tags | None = None, node_id: int | None = None) -> int:
        if node_id is None:
            node_id = self._next_node
            self._next_node += 1
        self.nodes[node_id] = (x, y, tags or {})
        return node_id

    def way(self, way_id: int, refs: Sequence[int], tags: Tags) -> None:
        self.ways[way_id] = (list(refs), tags)

    def polygon(self, way_id: int, points: Sequence[tuple[float, float]], tags: Tags) -> None:
        refs = [self.node(x, y) for x, y in points]
        self.way(way_id, [*refs, refs[0]], tags)

    def rect(
        self,
        way_id: int,
        cx: float,
        cy: float,
        width: float,
        height: float,
        tags: Tags,
        angle_deg: float = 0.0,
    ) -> None:
        cos, sin = math.cos(math.radians(angle_deg)), math.sin(math.radians(angle_deg))
        corners = [(-1, -1), (1, -1), (1, 1), (-1, 1)]
        points = [
            (
                cx + sx * width / 2 * cos - sy * height / 2 * sin,
                cy + sx * width / 2 * sin + sy * height / 2 * cos,
            )
            for sx, sy in corners
        ]
        self.polygon(way_id, points, tags)

    def relation(
        self, relation_id: int, members: Sequence[tuple[str, int, str]], tags: Tags
    ) -> None:
        self.relations[relation_id] = (list(members), tags)

    def write(self, path: Path) -> None:
        lines = ['<?xml version="1.0" encoding="UTF-8"?>', '<osm version="0.6" generator="test">']
        for node_id in sorted(self.nodes):
            x, y, tags = self.nodes[node_id]
            lon, lat = self.projection.inverse(x, y)
            head = f'<node id="{node_id}" version="1" lat="{float(lat):.7f}" lon="{float(lon):.7f}"'
            lines.append(head + (">" if tags else "/>"))
            lines.extend(_tag_lines(tags))
            if tags:
                lines.append("</node>")
        for way_id in sorted(self.ways):
            refs, tags = self.ways[way_id]
            lines.append(f'<way id="{way_id}" version="1">')
            lines.extend(f'<nd ref="{ref}"/>' for ref in refs)
            lines.extend(_tag_lines(tags))
            lines.append("</way>")
        for relation_id in sorted(self.relations):
            members, tags = self.relations[relation_id]
            lines.append(f'<relation id="{relation_id}" version="1">')
            lines.extend(
                f'<member type="{kind}" ref="{ref}" role="{role}"/>' for kind, ref, role in members
            )
            lines.extend(_tag_lines(tags))
            lines.append("</relation>")
        lines.append("</osm>")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _tag_lines(tags: Tags) -> list[str]:
    return [f"<tag k={quoteattr(k)} v={quoteattr(v)}/>" for k, v in tags.items()]


def populate_tiny_city(b: OsmBuilder) -> None:
    for i in range(5):
        for j in range(5):
            b.rect(
                20_000 + i * 5 + j,
                -780 + 40 * i,
                620 + 40 * j,
                20,
                20,
                {"building": "apartments", "building:levels": "3"},
            )
            b.rect(
                21_000 + i * 5 + j,
                620 + 40 * i,
                -780 + 40 * j,
                20,
                20,
                {"building": "apartments", "building:levels": "6"},
            )
    b.rect(1001, 300, 300, 20, 10, {"building": "office", "height": "15"})
    b.rect(1002, -500, 300, 20, 20, {"building": "yes", "building:levels": "4"})
    b.rect(1003, -700, 500, 20, 20, {"building": "yes"})
    b.rect(1004, -600, 500, 20, 20, {"building": "apartments"})
    b.rect(1005, -800, 500, 20, 20, {"building": "retail"})
    b.rect(1006, -300, -300, 20, 20, {"building": "no"})
    b.rect(1007, 1030, 100, 80, 20, {"building": "yes"})
    b.rect(1008, 200, -300, 30, 12, {"building": "house"}, angle_deg=120)
    b.polygon(
        1009,
        [(-820, -320), (-780, -280), (-780, -320), (-820, -280)],
        {"building": "yes"},
    )
    b.rect(1010, 60, -800, 20, 20, {"building": "yes"})
    b.rect(1011, -300, -700, 20, 20, {"building": "apartments"})
    b.rect(1012, 200, 600, 20, 20, {"building": "yes", "building:height": "8 m"})
    b.rect(1013, 150, 450, 20, 20, {"building": "yes", "height": "unknow", "building:height": "12"})

    b.rect(1100, 350, -450, 100, 100, {})
    b.rect(1101, 350, -450, 40, 40, {})
    b.relation(
        900,
        [("way", 1100, "outer"), ("way", 1101, "inner")],
        {"type": "multipolygon", "building": "yes"},
    )

    for node_id, (x, y) in ROAD_NODES.items():
        b.node(x, y, node_id=node_id)
    for way_id, (refs, tags) in ROAD_WAYS.items():
        b.way(way_id, refs, tags)

    r0 = b.node(0, -1200, node_id=6001)
    r1 = b.node(0, 0, node_id=6002)
    r2 = b.node(0, 1200, node_id=6003)
    d0 = b.node(-560, -1200, node_id=6004)
    d1 = b.node(-560, 1200, node_id=6005)
    b.way(300, [r0, r1], {"waterway": "river", "name": "Sabarmati River"})
    b.way(301, [r1, r2], {"waterway": "river", "name": "Sabarmati"})
    b.way(302, [d0, d1], {"waterway": "river", "name": "Other Creek"})

    b.polygon(
        310,
        [(-40, -1200), (40, -1200), (40, 1200), (-40, 1200)],
        {"natural": "water", "water": "river"},
    )
    b.rect(311, 700, 700, 200, 200, {"natural": "water", "water": "lake"})
    b.rect(312, -600, -600, 50, 50, {"natural": "water", "water": "pond"})
    b.rect(313, 1025, 250, 150, 100, {"natural": "water", "water": "lake"})
    b.rect(314, -800, -800, 120, 120, {"natural": "water", "water": "reservoir"})

    b.node(-700, 700, {"place": "suburb", "name": "Alpha"}, node_id=3003)
    b.node(700, -700, {"place": "neighbourhood", "name": "Beta"}, node_id=3004)
    b.node(-300, -800, {"place": "suburb", "name": "Gamma"}, node_id=3001)
    b.node(300, 800, {"place": "suburb", "name": "gamma "}, node_id=3002)
    b.node(-200, -200, {"place": "hamlet", "name": "Hamlet"}, node_id=3005)
    b.node(1500, 0, {"place": "suburb", "name": "Outside"}, node_id=3006)
    b.node(-100, 100, {"place": "suburb"}, node_id=3007)


@dataclass(frozen=True)
class TinyCity:
    path: Path
    projection: LocalProjection
    extent: tuple[float, float, float, float]

    @property
    def clip_height_m(self) -> float:
        return self.extent[3] - self.extent[1]


def clip_extent(projection: LocalProjection) -> tuple[float, float, float, float]:
    t = np.linspace(0.0, 1.0, 65)
    lon = BBOX.west + t * (BBOX.east - BBOX.west)
    lat = BBOX.south + t * (BBOX.north - BBOX.south)
    edge_lon = np.concatenate([lon, lon, np.full(65, BBOX.west), np.full(65, BBOX.east)])
    edge_lat = np.concatenate([np.full(65, BBOX.south), np.full(65, BBOX.north), lat, lat])
    x, y = projection.forward(edge_lon, edge_lat)
    return float(x.min()), float(y.min()), float(x.max()), float(y.max())


def ring_area(xy: npt.ArrayLike) -> float:
    ring = np.asarray(xy, np.float64)
    x, y = ring[:, 0], ring[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2)
