"""Regenerate the schema and instance quality report.

Computes the quality profile on the policy-driven semantic graph of the bundled
sample and writes a gated, reproducible report.

    make schema-quality
"""

from __future__ import annotations

import csv
import json
from datetime import date
from pathlib import Path

from kg_construction.application.plan_resolution import resolve_modeling_plan
from kg_construction.application.semantic_profiler import SemanticColumnProfiler
from kg_construction.application.semantic_table_mapper import SemanticTableMapper
from kg_construction.application.workflow import ConstructionWorkflow
from kg_construction.domain.models import SourceRecord
from kg_construction.infrastructure.evaluation.schema_quality import (
    SchemaQualityEvaluator,
    SchemaQualityReport,
)
from kg_construction.infrastructure.ontology.structured_ontology import propose_structured_ontology

_SAMPLE = Path("examples/data_to_kg/sample.csv")
_REPORTS = Path("reports")


def _report() -> tuple[SchemaQualityReport, dict]:
    records = list(csv.DictReader(_SAMPLE.read_text(encoding="utf-8").splitlines()))
    source = SourceRecord.from_rows(records, source_type="csv", source_name="sample.csv")
    modeling_plan = SemanticColumnProfiler().profile(records, table_name="sample", record_type="Employee")
    policy = modeling_plan.to_modeling_policy()
    resolved = resolve_modeling_plan(modeling_plan, source_name="sample.csv")
    ontology = propose_structured_ontology(policy).ontology
    graph = ConstructionWorkflow(mappers=[SemanticTableMapper(resolved)]).run([source]).property_graph
    report = SchemaQualityEvaluator().evaluate(
        graph, policy=policy, source_records=records, ontology=ontology
    )
    return report, {"record_count": len(records)}


def _markdown(report: SchemaQualityReport, meta: dict) -> str:
    lines = [
        "# Schema & Instance Quality Report",
        "",
        f"Generated: {date.today().isoformat()}",
        "",
        "Computed on the policy-driven semantic graph of `examples/data_to_kg/sample.csv` "
        "(the semantic mapper). Categorical duplication is policy-aware.",
        "",
        "## Release gates",
        "",
        "| Gate | Metric | Operator | Threshold | Actual | Passed |",
        "| --- | --- | --- | ---: | ---: | :---: |",
    ]
    for gate in report.gates:
        lines.append(
            f"| {gate.gate} | {gate.metric} | {gate.operator} | {gate.threshold:g} | {gate.actual:g} | "
            f"{'PASS' if gate.passed else 'FAIL'} |"
        )
    lines.extend(
        [
            "",
            f"**Overall: {'PASS' if report.passed else 'FAIL'}**",
            "",
            "## Conciseness (freedom from redundancy)",
            f"- categorical_duplication_rate: {report.categorical_duplication_rate}",
            f"- literal_that_should_be_node_rate: {report.literal_that_should_be_node_rate}",
            f"- duplicate_node_rate: {report.duplicate_node_rate}",
            "",
            "## Completeness",
            f"- schema_completeness: {report.schema_completeness}",
            f"- property_completeness: {report.property_completeness:.3f}",
            f"- relationship_completeness: {report.relationship_completeness:.3f}",
            f"- class_population: {report.class_population}",
            "",
            "## Consistency",
            f"- domain_range_violation_rate: {report.domain_range_violation_rate}",
            f"- cardinality_violation_rate: {report.cardinality_violation_rate}",
            f"- consistency_violation_count: {report.consistency_violation_count}",
            "",
            "## Structure (OntoQA)",
            f"- node_count: {report.node_count} | edge_count: {report.edge_count}",
            f"- average_degree: {report.average_degree:.3f} | orphan_rate: {report.orphan_rate}",
            f"- connected_component_count: {report.connected_component_count}",
            f"- class_richness: {report.class_richness} | relationship_richness: {report.relationship_richness:.3f}",
            f"- attribute_richness: {report.attribute_richness:.3f}",
            "",
            "Regenerate with `make schema-quality`.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    report, meta = _report()
    _REPORTS.mkdir(exist_ok=True)
    (_REPORTS / "schema_quality.json").write_text(
        json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (_REPORTS / "schema_quality.md").write_text(_markdown(report, meta), encoding="utf-8")
    print(f"schema-quality-ok: overall={'PASS' if report.passed else 'FAIL'}; gates={len(report.gates)}")


if __name__ == "__main__":
    main()
