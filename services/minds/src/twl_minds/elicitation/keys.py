import hashlib
import json
import re
from dataclasses import asdict, dataclass
from typing import Any

from twl_minds.elicitation.instrument import ElicitationBrief, PersonaProfile

_SCENARIO_HASH = re.compile(r"[0-9a-f]{1,128}")


@dataclass(frozen=True)
class CacheKey:
    persona_id: str
    scenario_hash: str
    prompt_version: str
    model: str

    def __post_init__(self) -> None:
        for name in ("persona_id", "scenario_hash", "prompt_version", "model"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must not be blank")


def brief_digest(brief: ElicitationBrief) -> str:
    return _sha256(
        _canonical_json(
            {
                "category": brief.category,
                "description": brief.description,
                "pack_size": brief.pack_size,
                "price_inr": int(brief.price_inr),
                "product_name": brief.product_name,
            }
        )
    )


def elicitation_scenario_hash(scenario_hash: str, brief: ElicitationBrief) -> str:
    if not isinstance(scenario_hash, str) or not _SCENARIO_HASH.fullmatch(scenario_hash):
        raise ValueError(
            f"scenario_hash must be 1 to 128 lowercase hex characters, got {scenario_hash!r}"
        )
    return _sha256(f"{scenario_hash}\n{brief_digest(brief)}")


def profile_digest(profile: PersonaProfile) -> str:
    return _sha256(_canonical_json(asdict(profile)))


def persona_id(persona_uuid: str, profile: PersonaProfile) -> str:
    if (
        not isinstance(persona_uuid, str)
        or not persona_uuid
        or ":" in persona_uuid
        or any(char.isspace() for char in persona_uuid)
    ):
        raise ValueError(
            f"persona_uuid must be non-blank with no ':' or whitespace, got {persona_uuid!r}"
        )
    return f"{persona_uuid}:{profile_digest(profile)[:16]}"


def _canonical_json(data: dict[str, Any]) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
