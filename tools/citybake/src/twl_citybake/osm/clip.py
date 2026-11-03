from pathlib import Path

import osmium
from osmium.osm import NODE, RELATION, WAY, Node, Relation, Way

from twl_citybake.config import BBox

AREA_RELATION_TYPES = ("multipolygon", "boundary")


def clip_extract(src: Path, dest: Path, bbox: BBox) -> None:
    """Keep what touches the bbox, completed with every node and member it references.

    Ways crossing the edge keep their outside nodes and area relations keep all their
    members, so geometry is clipped later in projected space rather than torn here.
    Route and other relation types are dropped: completing them would pull in ways
    from across the region.
    """
    tracker = osmium.IdTracker()
    dest.parent.mkdir(parents=True, exist_ok=True)
    with osmium.BackReferenceWriter(str(dest), ref_src=str(src), overwrite=True) as writer:
        for obj in osmium.FileProcessor(str(src), NODE | WAY | RELATION):
            if isinstance(obj, Node):
                loc = obj.location
                keep = loc.valid() and bbox.contains(loc.lon, loc.lat)
                if keep:
                    tracker.add_node(obj.id)
            elif isinstance(obj, Way):
                keep = tracker.contains_any_references(obj)
                if keep:
                    tracker.add_way(obj.id)
            else:
                assert isinstance(obj, Relation)
                keep = obj.tags.get("type") in AREA_RELATION_TYPES and (
                    tracker.contains_any_references(obj)
                )
                if keep:
                    tracker.add_relation(obj.id)
            if keep:
                writer.add(obj)
