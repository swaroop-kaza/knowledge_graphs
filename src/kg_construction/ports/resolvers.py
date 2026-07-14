"""Entity resolution plugin interface."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from kg_construction.domain.models import ConstructionBatch


@runtime_checkable
class EntityResolver(Protocol):
    """Resolve candidate entities without destructive merges."""

    name: str
    technique: str

    def resolve(self, batch: ConstructionBatch) -> ConstructionBatch:
        """Return a batch with canonicalized candidates and audit metadata."""
        ...
