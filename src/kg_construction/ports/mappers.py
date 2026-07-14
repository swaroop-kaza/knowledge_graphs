"""Structured and semi-structured mapping plugin interfaces."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from kg_construction.domain.models import ConstructionBatch, SourceRecord


@runtime_checkable
class SourceMapper(Protocol):
    """Map structured or semi-structured data into graph candidates."""

    name: str
    technique: str

    def map_source(self, source: SourceRecord) -> ConstructionBatch:
        """Map one source record into construction candidates."""
        ...
