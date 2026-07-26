"""Validation plugin interface."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from kg_construction.domain.models import ConstructionBatch
from kg_construction.domain.validation import ValidationReport


@runtime_checkable
class GraphValidator(Protocol):
    """Validate candidate facts before graph materialization."""

    name: str

    def validate(self, batch: ConstructionBatch) -> ValidationReport:
        """Return validation findings and status."""
        ...
