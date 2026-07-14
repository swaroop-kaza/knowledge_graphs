"""Entity and relation extraction plugin interfaces."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from kg_construction.domain.models import ConstructionBatch, SourceRecord


@runtime_checkable
class EntityExtractor(Protocol):
    """Extract candidate entities from a source record."""

    name: str
    technique: str

    def extract(self, source: SourceRecord) -> ConstructionBatch:
        """Return extracted entity candidates in a construction batch."""
        ...


@runtime_checkable
class RelationExtractor(Protocol):
    """Extract candidate relations from a source record."""

    name: str
    technique: str

    def extract(self, source: SourceRecord) -> ConstructionBatch:
        """Return extracted relation candidates in a construction batch."""
        ...
