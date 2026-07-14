"""Evaluation plugin interface."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from kg_construction.domain.models import ConstructionBatch


@runtime_checkable
class ConstructionEvaluator(Protocol):
    """Evaluate KG construction quality, cost, latency, and retrieval readiness."""

    name: str

    def evaluate(self, predicted: ConstructionBatch, expected: ConstructionBatch | None = None) -> dict[str, float]:
        """Return evaluation metrics."""
        ...
