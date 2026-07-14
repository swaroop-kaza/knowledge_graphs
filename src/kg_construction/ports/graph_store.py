"""Graph store and export plugin interfaces."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from kg_construction.domain.graph_objects import PropertyGraph, RDFGraph


@runtime_checkable
class GraphWriter(Protocol):
    """Serialize a graph to a target-specific representation."""

    name: str

    def export_property_graph(self, graph: PropertyGraph) -> object:
        """Export a property graph."""
        ...

    def export_rdf_graph(self, graph: RDFGraph) -> object:
        """Export an RDF graph."""
        ...


@runtime_checkable
class GraphStore(Protocol):
    """Persist validated graph objects to a graph database or RDF store."""

    name: str

    def write_property_graph(self, graph: PropertyGraph) -> None:
        """Write a property graph."""
        ...

    def write_rdf_graph(self, graph: RDFGraph) -> None:
        """Write an RDF graph."""
        ...
