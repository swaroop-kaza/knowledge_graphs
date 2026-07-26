"""Canonical structured key normalization and collision checks."""

from kg_construction.structured.structured_identity import (
    TargetKeyIndex,
    TargetKeyLookup,
    canonical_key,
    normalize_key_value,
    validate_unique_key,
)

__all__ = [
    "TargetKeyIndex",
    "TargetKeyLookup",
    "canonical_key",
    "normalize_key_value",
    "validate_unique_key",
]
