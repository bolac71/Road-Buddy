from __future__ import annotations

SUPPORTED_TYPES = {
    "sign_identification",
    "rule_compliance",
    "verification",
    "object_presence",
    "information_reading",
    "navigation",
    "counting",
    "other",
}


def resolve_dataset_type(sample_type: str | None) -> str:
    t = (sample_type or "other").strip()
    return t if t in SUPPORTED_TYPES else "other"
