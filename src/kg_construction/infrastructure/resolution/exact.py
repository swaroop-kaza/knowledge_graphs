"""Exact and normalized-key entity resolution."""

from __future__ import annotations

from kg_construction.domain.models import (
    ConstructionBatch,
    normalize_text,
)
from kg_construction.infrastructure.resolution.common import (
    apply_merge_decisions,
    decisions_from_groups,
)


class ExactKeyEntityResolver:
    """Attach canonical IDs by exact normalized entity type + name key."""

    name = "exact_key_entity_resolver"
    technique = "exact_key_matching"

    def resolve(self, batch: ConstructionBatch) -> ConstructionBatch:
        decisions = decisions_from_groups(
            batch.entities,
            key_fn=lambda entity: normalize_text(entity.name),
            resolver_name=self.name,
            confidence=1.0,
            reason="exact_normalized_entity_type_name",
        )
        return apply_merge_decisions(
            batch,
            decisions,
            resolver_name=self.name,
            resolver_technique=self.technique,
        )


__all__ = ["ExactKeyEntityResolver"]
