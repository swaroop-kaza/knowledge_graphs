"""Verify construction output against gold modelling fixtures.

Runs the real construction pipeline on a small gold case and checks the output
graph conforms to the expected modelling (input + expected graph, executed and
compared). Shared by the tests and the published modelling-correctness report so
both enforce the same contract.

Structured case: deduplicated dimension nodes, typed foreign-key edges, and zero
categorical duplication. Unstructured case: entity coverage against gold salient
mentions meets a minimum recall.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from kg_construction.construction.workflow import DraftConstructionWorkflow
from kg_construction.domain.models import SourceRecord
from kg_construction.evaluation.claim_evaluations import ClaimEvaluationSuite
from kg_construction.evaluation.extraction_coverage import CoverageEvaluator
from kg_construction.evaluation.schema_quality import SchemaQualityEvaluator
from kg_construction.infrastructure.extractors.regex import RegexEntityExtractor
from kg_construction.infrastructure.resolution.canonicalization import UnstructuredCanonicalizer
from kg_construction.structured.plan_resolution import resolve_modeling_plan
from kg_construction.structured.semantic_profiler import SemanticColumnProfiler
from kg_construction.structured.semantic_table_mapper import SemanticTableMapper
from kg_construction.unstructured.pipeline import UnstructuredPipeline

JsonDict = dict[str, Any]


class ModellingCase(BaseModel):
    """Result of one modelling-correctness case."""

    model_config = ConfigDict(extra="forbid")

    case_id: str
    kind: str
    passed: bool
    checks: dict[str, bool] = Field(default_factory=dict)
    metrics: dict[str, float] = Field(default_factory=dict)


def evaluate_structured_case(
    *,
    case_id: str,
    records: list[JsonDict],
    table_name: str,
    record_type: str,
    gold_graph: JsonDict,
) -> ModellingCase:
    """Build the semantic graph and compare it to the gold graph."""

    modeling_plan = SemanticColumnProfiler().profile(
        records, table_name=table_name, record_type=record_type
    )
    policy = modeling_plan.to_modeling_policy()
    resolved = resolve_modeling_plan(modeling_plan, source_name=f"{table_name}.csv")
    source = SourceRecord.from_rows(records, source_type="csv", source_name=f"{table_name}.csv")
    graph = DraftConstructionWorkflow(mappers=[SemanticTableMapper(resolved)]).run(
        [source]
    ).property_graph

    record_nodes = [node for node in graph.nodes if record_type in node.labels]
    edge_counts = Counter(edge.type for edge in graph.edges)
    quality = SchemaQualityEvaluator().evaluate(graph, policy=policy, source_records=records)
    duplication = quality.categorical_duplication_rate or 0.0

    checks: dict[str, bool] = {
        "record_node_count": len(record_nodes) == gold_graph["record_nodes"],
        "categorical_duplication_zero": duplication == gold_graph.get("categorical_duplication_rate", 0.0),
    }
    for reference_type, expected in gold_graph.get("shared_reference_nodes", {}).items():
        actual = sum(1 for node in graph.nodes if reference_type in node.labels)
        checks[f"shared_nodes:{reference_type}"] = actual == expected
    for edge_type, expected in gold_graph.get("edge_counts", {}).items():
        checks[f"edges:{edge_type}"] = edge_counts.get(edge_type, 0) == expected
    for column in gold_graph.get("property_categoricals", []):
        checks[f"property_kept:{column}"] = all(column in node.properties for node in record_nodes)
    for reference_type in gold_graph.get("absent_reference_types", []):
        checks[f"absent:{reference_type}"] = not any(reference_type in node.labels for node in graph.nodes)
    for column in gold_graph.get("record_literal_forbidden", []):
        checks[f"no_literal:{column}"] = not any(column in node.properties for node in record_nodes)

    metrics = {
        "record_nodes": float(len(record_nodes)),
        "categorical_duplication_rate": duplication,
        "orphan_rate": quality.orphan_rate,
        **{f"edges:{edge_type}": float(count) for edge_type, count in sorted(edge_counts.items())},
        **{
            f"nodes:{reference_type}": float(sum(1 for node in graph.nodes if reference_type in node.labels))
            for reference_type in gold_graph.get("shared_reference_nodes", {})
        },
    }
    return ModellingCase(
        case_id=case_id, kind="structured", passed=all(checks.values()), checks=checks, metrics=metrics
    )


def evaluate_unstructured_case(
    *,
    case_id: str,
    text: str,
    gold_entities: list[JsonDict],
    minimum_entity_coverage: float,
    gold_claims: list[JsonDict] | None = None,
    gold_intents: list[str] | None = None,
) -> ModellingCase:
    """Verify entity coverage, and score claims and intents on the same document.

    Entity recall is measured on the extracted batch; claims and intents are
    scored through the evidence-first pipeline, so gold claims and gold intents
    both affect the result and every promoted claim carries complete provenance.
    """

    source = SourceRecord.from_text(text, source_name=f"{case_id}.txt")
    batch = RegexEntityExtractor().extract(source)
    canonical = UnstructuredCanonicalizer().resolve(batch)
    coverage = CoverageEvaluator().evaluate(canonical, gold_entities=gold_entities)

    pipeline_result = UnstructuredPipeline().run(source)
    scores = ClaimEvaluationSuite().evaluate(
        pipeline_result,
        {"gold_entities": gold_entities, "gold_claims": gold_claims or [], "gold_intents": gold_intents or []},
    )
    provenance_complete = all(
        claim.has_complete_provenance(text) for claim in pipeline_result.promoted
    )

    checks = {
        "entity_coverage_meets_minimum": coverage.entity_coverage >= minimum_entity_coverage,
        "claims_have_complete_provenance": provenance_complete,
    }
    if gold_claims:
        checks["claims_evaluated"] = scores["relation"].executed
    if gold_intents:
        checks["intents_evaluated"] = scores["intent_fact"].executed
    metrics = {
        "entity_coverage": coverage.entity_coverage,
        "matched_entity_count": float(coverage.matched_entity_count),
        "gold_entity_count": float(coverage.gold_entity_count),
        "nil_rate": coverage.nil_rate,
        "minimum_entity_coverage": minimum_entity_coverage,
        "claim_coverage": scores["relation"].recall,
        "intent_coverage": scores["intent_fact"].recall,
        "promoted_claim_count": float(len(pipeline_result.promoted)),
    }
    return ModellingCase(
        case_id=case_id, kind="unstructured", passed=all(checks.values()), checks=checks, metrics=metrics
    )


__all__ = ["ModellingCase", "evaluate_structured_case", "evaluate_unstructured_case"]
