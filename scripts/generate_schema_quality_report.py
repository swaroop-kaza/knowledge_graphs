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

from kg_construction.construction.workflow import DraftConstructionWorkflow
from kg_construction.domain.models import SourceRecord
from kg_construction.evaluation.schema_quality import (
    SchemaQualityEvaluator,
    SchemaQualityReport,
)
from kg_construction.infrastructure.ontology.structured_ontology import (
    StructuredReleaseGate,
)
from kg_construction.structured.plan_resolution import resolve_modeling_plan
from kg_construction.structured.semantic_profiler import SemanticColumnProfiler
from kg_construction.structured.semantic_table_mapper import SemanticTableMapper

_SAMPLE = Path("examples/data_to_kg/sample.csv")
_REPORTS = Path("reports")


def _report() -> SchemaQualityReport:
    records = list(csv.DictReader(_SAMPLE.read_text(encoding="utf-8").splitlines()))
    source = SourceRecord.from_rows(records, source_type="csv", source_name="sample.csv")
    modeling_plan = SemanticColumnProfiler().profile(records, table_name="sample", record_type="Employee")
    resolved = resolve_modeling_plan(modeling_plan, source_name="sample.csv")
    ontology = resolved.to_ontology_schema()
    result = DraftConstructionWorkflow(
        mappers=[SemanticTableMapper(resolved)]
    ).run([source])
    shacl = StructuredReleaseGate().validate(
        result.rdf_graph,
        shacl_turtle=resolved.to_shacl_shapes(),
    )
    report = SchemaQualityEvaluator().evaluate(
        result.property_graph,
        policy=resolved,
        source_records=records,
        ontology=ontology,
        shacl_executed=True,
        shacl_violation_count=shacl.violation_count,
    )
    return report


def _markdown(report: SchemaQualityReport) -> str:
    lines = [
        "# Schema & Instance Quality Report",
        "",
        "[Reports index](README.md) · [Evaluation guide](../docs/evaluation_metrics.md) · [Testing guide](../docs/testing_guide.md)",
        "",
        "| Metadata | Value |",
        "| --- | --- |",
        f"| Generated | {date.today().isoformat()} |",
        f"| Thresholds | `{report.thresholds_version}` |",
        f"| Overall | **{'PASS' if report.passed else 'FAIL'}** |",
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
            "## Conciseness (freedom from redundancy)",
            "",
            "| Metric | Value |",
            "| --- | ---: |",
            f"| `categorical_duplication_rate` | {report.categorical_duplication_rate} |",
            f"| `literal_that_should_be_node_rate` | {report.literal_that_should_be_node_rate} |",
            f"| `duplicate_node_rate` | {report.duplicate_node_rate} |",
            "",
            "## Completeness",
            "",
            "| Metric | Value |",
            "| --- | ---: |",
            f"| `schema_completeness` | {report.schema_completeness} |",
            f"| `property_completeness` | {report.property_completeness:.3f} |",
            f"| `relationship_completeness` | {report.relationship_completeness:.3f} |",
            f"| `class_population` | {report.class_population} |",
            "",
            "## Consistency",
            "",
            "| Metric | Value |",
            "| --- | ---: |",
            f"| `domain_range_violation_rate` | {report.domain_range_violation_rate} |",
            f"| `cardinality_violation_rate` | {report.cardinality_violation_rate} |",
            f"| `shacl_executed` | {report.shacl_executed} |",
            f"| `shacl_violation_count` | {report.shacl_violation_count} |",
            f"| `consistency_violation_count` | {report.consistency_violation_count} |",
            "",
            "## Structure (OntoQA)",
            "",
            "| Metric | Value |",
            "| --- | ---: |",
            f"| `node_count` | {report.node_count} |",
            f"| `edge_count` | {report.edge_count} |",
            f"| `average_degree` | {report.average_degree:.3f} |",
            f"| `orphan_rate` | {report.orphan_rate} |",
            f"| `connected_component_count` | {report.connected_component_count} |",
            f"| `class_richness` | {report.class_richness} |",
            f"| `relationship_richness` | {report.relationship_richness:.3f} |",
            f"| `attribute_richness` | {report.attribute_richness:.3f} |",
            "",
            "## Reproduce",
            "",
            "```bash",
            "make schema-quality",
            "```",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    report = _report()
    _REPORTS.mkdir(exist_ok=True)
    (_REPORTS / "schema_quality.json").write_text(
        json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (_REPORTS / "schema_quality.md").write_text(_markdown(report), encoding="utf-8")
    print(f"schema-quality-ok: overall={'PASS' if report.passed else 'FAIL'}; gates={len(report.gates)}")
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
