"""Command-line Data-to-KG construction runner."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from kg_construction.construction import construct_kg, load_sources
from kg_construction.construction.config import DataToKGConfig
from kg_construction.infrastructure.graph_stores.exporters import Neo4jExporter, RDFExporter


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Construct knowledge graphs from data sources.")
    parser.add_argument("--input", required=True, help="Input file path.")
    parser.add_argument(
        "--source-type",
        choices=[
            "api",
            "api_response",
            "chat_transcript",
            "csv",
            "email",
            "event_payload",
            "excel",
            "html",
            "json",
            "jsonld",
            "logs",
            "markdown",
            "pdf",
            "report",
            "scientific_paper",
            "sql",
            "structured_api",
            "text",
            "webpage",
            "xml",
        ],
        required=True,
    )
    parser.add_argument(
        "--config",
        help="YAML config path; the product defaults are used when omitted.",
    )
    parser.add_argument(
        "--technique",
        help=(
            "Conversion technique (for example semantic_table_mapper, "
            "direct_table_mapper, jsonpath_mapper, evidence_first_unstructured_pipeline). "
            "The default technique for the source type is used when omitted."
        ),
    )
    parser.add_argument(
        "--evaluation-suite",
        help=(
            "Sealed versioned YAML/JSON gold suite to execute against the draft "
            "graph. Its formation, extraction, and intended-use results are "
            "reported; only the separate certification command authorizes release."
        ),
    )
    parser.add_argument("--output", required=True, help="Property graph JSON output path.")
    parser.add_argument(
        "--rdf-output",
        help="RDF JSON output path override; always emitted beside --output by default.",
    )
    parser.add_argument(
        "--evidence-output",
        help="Evidence graph JSON path override; always emitted beside --output by default.",
    )
    parser.add_argument(
        "--neo4j-output",
        help="Neo4j Cypher path override; always emitted beside --output by default.",
    )
    parser.add_argument(
        "--quality-output",
        help="Quality-gate JSON path override; always emitted beside --output by default.",
    )
    parser.add_argument(
        "--intended-use-output",
        help=(
            "Intended-use benchmark JSON path override; always emitted beside "
            "--output by default."
        ),
    )
    parser.add_argument(
        "--publication-contract-output",
        help=(
            "Trusted release-contract evidence path override; always emitted "
            "beside --output by default."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    raw_args = list(sys.argv[1:] if argv is None else argv)
    if raw_args and raw_args[0] == "doctor":
        from kg_construction.interfaces.cli.doctor import main as doctor_main

        return doctor_main(raw_args[1:])
    if raw_args and raw_args[0] == "construct":
        raw_args = raw_args[1:]

    args = build_parser().parse_args(raw_args)
    config = DataToKGConfig.from_yaml(args.config) if args.config else DataToKGConfig()
    sources = load_sources(args.input, source_type=args.source_type)
    result = construct_kg(
        sources,
        source_type=args.source_type,
        config=config,
        technique=args.technique,
        evaluation_suite_path=args.evaluation_suite,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(result.property_graph.to_json(), encoding="utf-8")

    rdf_path = _derived_output_path(output_path, args.rdf_output, ".rdf.json")
    evidence_path = _derived_output_path(
        output_path,
        args.evidence_output,
        ".evidence.json",
    )
    neo4j_path = _derived_output_path(
        output_path,
        args.neo4j_output,
        ".neo4j.cypher",
    )
    quality_path = _derived_output_path(
        output_path,
        args.quality_output,
        ".quality.json",
    )
    intended_use_path = _derived_output_path(
        output_path,
        args.intended_use_output,
        ".intended-use.json",
    )
    publication_contract_path = _derived_output_path(
        output_path,
        args.publication_contract_output,
        ".publication-contract.json",
    )
    for path in (
        rdf_path,
        evidence_path,
        neo4j_path,
        quality_path,
        intended_use_path,
        publication_contract_path,
    ):
        path.parent.mkdir(parents=True, exist_ok=True)

    rdf_payload = RDFExporter().export_rdf_graph(result.rdf_graph)
    rdf_path.write_text(
        json.dumps(rdf_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    evidence_path.write_text(result.evidence_graph.to_json(), encoding="utf-8")
    neo4j_path.write_text(
        "\n".join(Neo4jExporter().export_property_graph(result.property_graph)),
        encoding="utf-8",
    )
    quality_path.write_text(
        result.quality_report.model_dump_json(indent=2),
        encoding="utf-8",
    )
    intended_use_path.write_text(
        result.intended_use_report.model_dump_json(indent=2),
        encoding="utf-8",
    )
    # Draft conversion has no release authorization. This branch is retained for
    # callers that supply a workflow result already bound by certification.
    if result.publication_contract is not None:
        publication_contract_path.write_text(
            result.publication_contract.model_dump_json(indent=2),
            encoding="utf-8",
        )

    summary = {
        "sources": len(sources),
        "entities": len(result.batch.entities),
        "relations": len(result.batch.relations),
        "property_nodes": len(result.property_graph.nodes),
        "property_edges": len(result.property_graph.edges),
        "validation_findings": len(result.validation_report.findings),
        "quality_passed": result.quality_report.passed,
        "quality_missing_families": result.quality_report.missing_families,
        "intended_use_passed": result.intended_use_report.passed,
        "missing_intended_uses": result.intended_use_report.missing_uses,
        "publication_passed": result.publication_gate.passed,
        "release_eligible": result.release_eligible,
        "publication_contract_id": (
            result.publication_contract.contract_id
            if result.publication_contract is not None
            else None
        ),
        "publication_contract_output": (
            str(publication_contract_path)
            if result.publication_contract is not None
            else None
        ),
        "output": str(output_path),
        "rdf_output": str(rdf_path),
        "evidence_output": str(evidence_path),
        "neo4j_output": str(neo4j_path),
        "quality_output": str(quality_path),
        "intended_use_output": str(intended_use_path),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _derived_output_path(
    property_output: Path,
    override: str | None,
    suffix: str,
) -> Path:
    if override:
        return Path(override)
    base_name = (
        property_output.name[: -len(property_output.suffix)]
        if property_output.suffix
        else property_output.name
    )
    return property_output.with_name(f"{base_name}{suffix}")


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_parser", "main"]
