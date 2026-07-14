"""Source loader plugin interface."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from kg_construction.domain.models import SourceRecord


@runtime_checkable
class SourceLoader(Protocol):
    """Load raw data into normalized source records."""

    name: str
    source_type: str

    def load(self, path: str | Path) -> list[SourceRecord]:
        """Load source records from a path."""
        ...
