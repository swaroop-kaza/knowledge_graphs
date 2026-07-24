"""Dependency-light graph exporters."""

from __future__ import annotations

import json
import re
from typing import Any

from kg_construction.domain.graph_objects import PropertyGraph, RDFGraph
from kg_construction.infrastructure.graph_stores.ingestion import (
    LadybugIngestionAdapter,
    Neo4jIngestionAdapter,
)

_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9_]")


def _safe_name(value: str) -> str:
    cleaned = _SAFE_NAME_RE.sub("_", value)
    return cleaned or "Unknown"


def _json_props(properties: dict[str, Any]) -> str:
    return json.dumps(properties, sort_keys=True)


class Neo4jExporter:
    """Export a property graph to idempotent Cypher statements."""

    name = "neo4j_exporter"

    def export_property_graph(self, graph: PropertyGraph) -> list[str]:
        statements: list[str] = []
        for node in graph.nodes:
            labels = ":".join(_safe_name(label) for label in node.labels)
            statements.append(
                f'MERGE (n:{labels} {{id: "{node.node_id}"}}) SET n += {_json_props(node.properties)};'
            )
        for edge in graph.edges:
            rel_type = _safe_name(edge.type).upper()
            statements.append(
                f'MATCH (s {{id: "{edge.source_id}"}}), (t {{id: "{edge.target_id}"}}) '
                f'MERGE (s)-[r:{rel_type} {{id: "{edge.edge_id}"}}]->(t) '
                f"SET r += {_json_props(edge.properties)};"
            )
        return statements

    def export_rdf_graph(self, graph: RDFGraph) -> list[str]:
        return graph.to_nquads_lines() or graph.to_turtle_lines()


class MemgraphExporter(Neo4jExporter):
    """Memgraph-compatible Cypher exporter."""

    name = "memgraph_exporter"


class LadybugExporter:
    """Export a property graph into a simple Ladybug ingestion manifest."""

    name = "ladybug_exporter"

    def export_property_graph(self, graph: PropertyGraph) -> dict[str, object]:
        return {
            "nodes": [node.model_dump(mode="json") for node in graph.nodes],
            "relationships": [edge.model_dump(mode="json") for edge in graph.edges],
            "notes": "Use this manifest to create node/relationship tables in Ladybug.",
        }

    def export_rdf_graph(self, graph: RDFGraph) -> dict[str, object]:
        return graph.to_dict()


class NetworkXExporter:
    """Export a property graph to a NetworkX-compatible node-link dictionary."""

    name = "networkx_exporter"

    def export_property_graph(self, graph: PropertyGraph) -> dict[str, object]:
        return {
            "directed": True,
            "multigraph": True,
            "graph": graph.metadata,
            "nodes": [
                {"id": node.node_id, "labels": node.labels, **node.properties}
                for node in graph.nodes
            ],
            "links": [
                {
                    "source": edge.source_id,
                    "target": edge.target_id,
                    "key": edge.edge_id,
                    "type": edge.type,
                    **edge.properties,
                }
                for edge in graph.edges
            ],
        }

    def export_rdf_graph(self, graph: RDFGraph) -> dict[str, object]:
        return graph.to_dict()


class RDFExporter:
    """Export RDF graphs as RDFLib-backed Turtle, N-Quads, and JSON payloads."""

    name = "rdf_exporter"

    def export_property_graph(self, graph: PropertyGraph) -> dict[str, object]:
        return graph.to_dict()

    def export_rdf_graph(self, graph: RDFGraph) -> dict[str, str]:
        return {
            "turtle": graph.serialize_turtle(),
            "nquads": graph.serialize_nquads(),
        }


class RDFLibExporter:
    """RDFLib exporter for data graphs and datasets."""

    name = "rdflib_exporter"

    def export_property_graph(self, graph: PropertyGraph) -> dict[str, object]:
        return graph.to_dict()

    def export_rdf_graph(self, graph: RDFGraph) -> object:
        return graph.to_rdflib_graph()

    def export_rdf_dataset(self, graph: RDFGraph) -> object:
        return graph.to_rdflib_dataset()


# Compatibility aliases for the database's former product name.
KuzuExporter = LadybugExporter
KuzuIngestionAdapter = LadybugIngestionAdapter


__all__ = [
    "LadybugExporter",
    "LadybugIngestionAdapter",
    "KuzuExporter",
    "KuzuIngestionAdapter",
    "MemgraphExporter",
    "Neo4jExporter",
    "Neo4jIngestionAdapter",
    "NetworkXExporter",
    "RDFExporter",
    "RDFLibExporter",
]
