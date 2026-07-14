"""Command-line Data-to-KG construction runner."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from kg_construction.application.config import DataToKGConfig
from kg_construction.application.production import build_production_services
from kg_construction.application.workflow import ConstructionWorkflow
from kg_construction.infrastructure.extractors.regex import RegexEntityExtractor
from kg_construction.infrastructure.graph_stores.exporters import Neo4jExporter, RDFExporter
from kg_construction.infrastructure.loaders import (
    APIResponseLoader,
    ChatTranscriptLoader,
    EmailLoader,
    ExcelLoader,
    MarkdownLoader,
    PDFLoader,
    ReportLoader,
    XMLLoader,
)
from kg_construction.infrastructure.loaders.csv_loader import CsvLoader
from kg_construction.infrastructure.loaders.json_loader import JsonLoader
from kg_construction.infrastructure.loaders.text_loader import TextLoader
from kg_construction.infrastructure.mapping.direct_table_mapper import DirectTableMapper
from kg_construction.infrastructure.mapping.semi_structured import JsonPathMapper
from kg_construction.infrastructure.ontology.validation import EvidenceRequiredValidator
from kg_construction.infrastructure.resolution.exact import ExactKeyEntityResolver


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Construct knowledge graphs from data sources.")
    parser.add_argument("--input", required=True, help="Input file path.")
    parser.add_argument(
        "--source-type",
        choices=[
            "api_response",
            "chat_transcript",
            "csv",
            "email",
            "excel",
            "json",
            "markdown",
            "pdf",
            "report",
            "text",
            "xml",
        ],
        required=True,
    )
    parser.add_argument("--config", help="Optional YAML config path.")
    parser.add_argument("--output", required=True, help="Property graph JSON output path.")
    parser.add_argument("--rdf-output", help="RDF JSON/Turtle/N-Quads output path.")
    parser.add_argument("--evidence-output", help="Evidence graph JSON output path.")
    parser.add_argument("--neo4j-output", help="Neo4j Cypher output path.")
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
    sources = _load_sources(args.input, args.source_type)
    production_services = build_production_services(config)
    workflow = _workflow_for_source(
        args.source_type,
        config,
        production_kwargs=production_services.workflow_kwargs(),
    )
    result = workflow.run(sources)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(result.property_graph.to_json(), encoding="utf-8")

    if args.rdf_output:
        rdf_payload = RDFExporter().export_rdf_graph(result.rdf_graph)
        Path(args.rdf_output).write_text(json.dumps(rdf_payload, indent=2, sort_keys=True), encoding="utf-8")

    if args.evidence_output:
        Path(args.evidence_output).write_text(result.evidence_graph.to_json(), encoding="utf-8")

    if args.neo4j_output:
        Path(args.neo4j_output).write_text(
            "\n".join(Neo4jExporter().export_property_graph(result.property_graph)),
            encoding="utf-8",
        )

    summary = {
        "sources": len(sources),
        "entities": len(result.batch.entities),
        "relations": len(result.batch.relations),
        "property_nodes": len(result.property_graph.nodes),
        "property_edges": len(result.property_graph.edges),
        "validation_findings": len(result.validation_report.findings),
        "output": str(output_path),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _load_sources(input_path: str, source_type: str):
    if source_type == "csv":
        return CsvLoader().load(input_path)
    if source_type == "excel":
        return ExcelLoader().load(input_path)
    if source_type == "json":
        return JsonLoader().load(input_path)
    if source_type == "api_response":
        return APIResponseLoader().load(input_path)
    if source_type == "xml":
        return XMLLoader().load(input_path)
    if source_type == "markdown":
        return MarkdownLoader().load(input_path)
    if source_type == "pdf":
        return PDFLoader().load(input_path)
    if source_type == "report":
        return ReportLoader().load(input_path)
    if source_type == "email":
        return EmailLoader().load(input_path)
    if source_type == "chat_transcript":
        return ChatTranscriptLoader().load(input_path)
    return TextLoader().load(input_path)


def _workflow_for_source(
    source_type: str,
    config: DataToKGConfig,
    *,
    production_kwargs: dict[str, Any] | None = None,
) -> ConstructionWorkflow:
    validators: list[Any] = [EvidenceRequiredValidator(config.pipeline.validation_policy)]
    resolvers: list[Any] = [ExactKeyEntityResolver()]
    common_kwargs: dict[str, Any] = {
        "pipeline_name": config.pipeline.name,
        "config_payload": config.model_dump(mode="json"),
        **(production_kwargs or {}),
    }
    if source_type in {"csv", "excel", "xml", "chat_transcript"}:
        return ConstructionWorkflow(
            mappers=[DirectTableMapper()],
            resolvers=resolvers,
            validators=validators,
            **common_kwargs,
        )
    if source_type in {"json", "api_response"}:
        return ConstructionWorkflow(
            mappers=[JsonPathMapper()],
            resolvers=resolvers,
            validators=validators,
            **common_kwargs,
        )
    return ConstructionWorkflow(
        extractors=[RegexEntityExtractor()],
        resolvers=resolvers,
        validators=validators,
        **common_kwargs,
    )


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_parser", "main"]
