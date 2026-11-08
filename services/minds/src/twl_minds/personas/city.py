import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from twl_minds.personas.sampling import PersonaSample, SampleConfig, sample_personas
from twl_pack import PackReader

_PACK_HASH = re.compile(r"[0-9a-f]{64}")
DEFAULT_CONFIG = SampleConfig()


def pack_entropy(pack_hash: str) -> tuple[int, ...]:
    """The pack's blake3 as eight 32-bit words, most significant first, so each city draws its
    own sample and the same pack always draws the same one."""
    if not _PACK_HASH.fullmatch(pack_hash):
        raise ValueError(f"pack_hash must be 64 lowercase hex digits, got {pack_hash!r}")
    return tuple(int(pack_hash[i : i + 8], 16) for i in range(0, 64, 8))


@dataclass(frozen=True, eq=False)
class CitySample:
    pack_hash: str
    segments: tuple[str, ...]
    config: SampleConfig
    sample: PersonaSample
    district: NDArray[np.uint8]


def sample_city(pack_dir: Path, config: SampleConfig = DEFAULT_CONFIG) -> CitySample:
    reader = PackReader(pack_dir)
    segments = tuple(reader.manifest["segments"])
    residents = reader.residents()
    engine = reader.residents_engine()
    sample = sample_personas(
        residents.segment,
        engine.persona_ref,
        engine.weight,
        engine.tier,
        len(segments),
        config,
        pack_entropy(reader.pack_hash),
    )
    district = reader.buildings().district[residents.building[sample.resident]]
    return CitySample(
        pack_hash=reader.pack_hash,
        segments=segments,
        config=config,
        sample=sample,
        district=district.astype(np.uint8),
    )


def sample_rows(city: CitySample) -> list[dict[str, Any]]:
    s = city.sample
    tier = s.tier
    if tier is None:
        raise ValueError("a city sample carries the pack's tier")
    return [
        {
            "resident": int(s.resident[i]),
            "persona_ref": int(s.persona_ref[i]),
            "segment": city.segments[int(s.segment[i])],
            "tier": int(tier[i]),
            "district": int(city.district[i]),
            "stratum": int(s.stratum[i]),
            "design_weight": float(s.design_weight[i]),
        }
        for i in range(len(s))
    ]
