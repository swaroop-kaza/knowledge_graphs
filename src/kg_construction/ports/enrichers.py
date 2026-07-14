"""Graph enrichment plugin interface."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from kg_construction.domain.graph_objects import PropertyGraph


@runtime_checkable
class GraphEnricher(Protocol):
    """Add embeddings, summaries, inferred links, or graph analytics."""

    name: str
    technique: str

    def enrich(self, graph: PropertyGraph) -> PropertyGraph:
        """Return an enriched graph."""
        ...
