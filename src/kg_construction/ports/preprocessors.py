"""Preprocessing plugin interface."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from kg_construction.domain.models import SourceRecord


@runtime_checkable
class SourcePreprocessor(Protocol):
    """Normalize, segment, clean, or enrich source records before extraction."""

    name: str

    def preprocess(self, source: SourceRecord) -> list[SourceRecord]:
        """Return preprocessed records."""
        ...
