"""Exact and normalized-key entity resolution."""

from __future__ import annotations

from kg_construction.domain.models import (
    ConstructionBatch,
    EntityCandidate,
    normalize_text,
)
from kg_construction.infrastructure.resolution.common import (
    apply_merge_decisions,
    decisions_from_groups,
)


class ExactKeyEntityResolver:
    """Attach canonical IDs by an explicit identity key or normalized display name.

    Structured mappers can provide a source-scoped ``identity_key``.  It takes precedence over
    mutable/redacted display values; legacy and text candidates retain name-based behavior.
    """

    name = "exact_key_entity_resolver"
    technique = "exact_key_matching"

    def __init__(self, *, allow_cross_source_name_fallback: bool = False) -> None:
        self.allow_cross_source_name_fallback = allow_cross_source_name_fallback

    def resolve(self, batch: ConstructionBatch) -> ConstructionBatch:
        decisions = decisions_from_groups(
            batch.entities,
            key_fn=lambda entity: _exact_resolution_key(
                entity,
                allow_cross_source_name_fallback=self.allow_cross_source_name_fallback,
            ),
            resolver_name=self.name,
            confidence=1.0,
            reason=(
                "explicit_identity_key_or_global_exact_normalized_name"
                if self.allow_cross_source_name_fallback
                else "explicit_identity_key_or_source_scoped_exact_normalized_name"
            ),
        )
        return apply_merge_decisions(
            batch,
            decisions,
            resolver_name=self.name,
            resolver_technique=self.technique,
        )


def _exact_resolution_key(
    entity: EntityCandidate,
    *,
    allow_cross_source_name_fallback: bool,
) -> str:
    identity_key = entity.properties.get("identity_key")
    if isinstance(identity_key, str) and identity_key.strip():
        # Explicit identifiers are already governed values.  Treat them as
        # exact and case-sensitive unless an upstream namespace policy has
        # intentionally normalized them.
        return f"identity:{identity_key}"
    normalized_name = normalize_text(entity.name)
    if allow_cross_source_name_fallback:
        return f"name:{normalized_name}"
    source_scopes = sorted(
        {
            f"{evidence.source_type}:{evidence.source_name}"
            for evidence in entity.evidence
        }
    )
    scope = "|".join(source_scopes) if source_scopes else entity.candidate_id
    return f"source_name:{scope}:{normalized_name}"


__all__ = ["ExactKeyEntityResolver"]
