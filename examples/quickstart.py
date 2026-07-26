"""Complete construction quickstart for structured and unstructured data.

Run ``make setup`` first. The text path deliberately executes the locked,
platform-mandatory model ensemble; this example does not substitute a lighter
profile when a provider is unavailable.
"""

from __future__ import annotations

import json
from typing import Any

from kg_construction import construct_kg, profile_dataset
from kg_construction.infrastructure.graph_stores import (
    Neo4jExporter,
    NetworkXExporter,
    RDFExporter,
)


def _counts(result: Any) -> dict[str, Any]:
    return {
        "entities": len(result.batch.entities),
        "relations": len(result.batch.relations),
        "property_nodes": len(result.property_graph.nodes),
        "property_edges": len(result.property_graph.edges),
        "rdf_triples": len(result.rdf_graph.triples),
        "formation_quality_passed": result.quality_report.passed,
        "executed_quality_families": result.quality_report.executed_families,
        "missing_quality_families": result.quality_report.missing_families,
        "intended_use_passed": result.intended_use_report.passed,
        "missing_intended_uses": result.intended_use_report.missing_uses,
        "publication_passed": result.publication_gate.passed,
        "release_eligible": result.release_eligible,
    }


def run_quickstart() -> dict[str, Any]:
    """Build evaluated draft graphs and demonstrate the three export forms."""

    csv_path = "examples/data_to_kg/sample.csv"
    text_path = "examples/data_to_kg/sample.txt"
    config_path = "configs/data_to_kg_complete.example.yaml"

    csv_result = construct_kg(
        csv_path,
        source_type="csv",
        config=config_path,
        evaluation_suite_path="configs/graph_evaluation_suite.sample_csv.yaml",
    )
    text_result = construct_kg(
        text_path,
        source_type="text",
        config=config_path,
        evaluation_suite_path="configs/graph_evaluation_suite.sample_text.yaml",
    )

    rdf_payload = RDFExporter().export_rdf_graph(csv_result.rdf_graph)
    neo4j_cypher = Neo4jExporter().export_property_graph(csv_result.property_graph)
    networkx_payload = NetworkXExporter().export_property_graph(csv_result.property_graph)

    return {
        "csv_profile": profile_dataset(csv_path).model_dump(mode="json"),
        "csv_to_kg": _counts(csv_result),
        "text_to_kg": _counts(text_result),
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
