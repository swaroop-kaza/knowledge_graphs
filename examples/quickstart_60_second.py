"""60-second local quickstart for data scientists.

This example uses only the core package install. It does not call live Neo4j,
Ollama, GLiNER, Kuzu, or graph ML backends.
"""

from __future__ import annotations

import json
from typing import Any

from kg_construction import construct_kg, evaluate_techniques, profile_dataset
from kg_construction.infrastructure.graph_stores import Neo4jExporter, NetworkXExporter, RDFExporter


def _counts(result: Any) -> dict[str, int]:
    return {
        "entities": len(result.batch.entities),
        "relations": len(result.batch.relations),
        "property_nodes": len(result.property_graph.nodes),
        "property_edges": len(result.property_graph.edges),
        "rdf_triples": len(result.rdf_graph.triples),
    }


def run_quickstart() -> dict[str, Any]:
    """Run CSV, text, JSON, benchmark, and export examples."""

    csv_path = "examples/data_to_kg/sample.csv"
    text_path = "examples/data_to_kg/sample.txt"
    json_path = "examples/data_to_kg/sample.json"

    csv_result = construct_kg(csv_path)
    text_result = construct_kg(text_path, source_type="text")
    json_result = construct_kg(json_path, source_type="json")

    benchmark = evaluate_techniques(
        text_path,
        source_type="text",
        techniques="regex_entity_extractor,hybrid_rules_gazetteer_regex",
    )

    rdf_payload = RDFExporter().export_rdf_graph(csv_result.rdf_graph)
    neo4j_cypher = Neo4jExporter().export_property_graph(csv_result.property_graph)
    networkx_payload = NetworkXExporter().export_property_graph(csv_result.property_graph)

    return {
        "csv_profile": profile_dataset(csv_path).model_dump(mode="json"),
        "csv_to_kg": _counts(csv_result),
        "text_to_kg": _counts(text_result),
        "json_to_kg": _counts(json_result),
        "best_benchmark_technique": benchmark.best_technique,
        "export_preview": {
            "rdf_turtle_lines": len(rdf_payload["turtle"].splitlines()),
            "neo4j_cypher_statements": len(neo4j_cypher),
            "networkx_nodes": len(networkx_payload["nodes"]),
            "networkx_links": len(networkx_payload["links"]),
        },
    }


def main() -> None:
    """Print the quickstart summary as JSON."""

    print(json.dumps(run_quickstart(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
