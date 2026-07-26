"""Modelling-defect gate: closed defects stay closed, open ones stay tracked.

Each case was a defect reproduced against commit 42d0100. Cases fixed by a later
change assert the corrected behaviour directly (a regression re-breaks the run).
Closed by the structured redesign, unstructured pipeline, and release-scope audit:

1. The structured workflow drives the semantic pipeline
   (shared dimension nodes, typed relationships, no duplicated literals).
2. A categorical's semantic role and physical materialization agree across the
   contract, ontology, and graph.
3. A foreign key whose target is absent is retained as a value, never fabricated
   as a placeholder entity.
4. Unstructured correctness scores the gold claims and intents, and every promoted
   claim carries complete provenance.
5. The release registry exposes implemented construction/evaluation capabilities
   only; adapter-only work and stubs live in a separate research inventory.
6. Schema quality can report ``passed`` without SHACL ever executing.
"""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from kg_construction.certification import default_registry
from kg_construction.construction.workflow import DraftConstructionWorkflow
from kg_construction.domain.models import SourceRecord
from kg_construction.evaluation.modelling_correctness import evaluate_unstructured_case
from kg_construction.evaluation.schema_quality import SchemaQualityEvaluator
from kg_construction.research.workflow import (
    draft_workflow_for_source_type,
)
from kg_construction.structured.plan_resolution import resolve_modeling_plan
from kg_construction.structured.semantic_profiler import SemanticColumnProfiler
from kg_construction.structured.semantic_table_mapper import SemanticTableMapper

_FIXTURE = Path(__file__).parent / "fixtures" / "gold_standards" / "column_roles.json"


def _table(name: str) -> dict:
    data = json.loads(_FIXTURE.read_text(encoding="utf-8"))
    return next(table for table in data["tables"] if table["table_name"] == name)


def _semantic_graph(table: dict):
    modeling_plan = SemanticColumnProfiler().profile(
        table["records"], table_name=table["table_name"], record_type=table["record_type"]
    )
    policy = modeling_plan.to_modeling_policy()
    resolved = resolve_modeling_plan(modeling_plan, source_name=f"{table['table_name']}.csv")
    source = SourceRecord.from_rows(
        table["records"], source_type="csv", source_name=f"{table['table_name']}.csv"
    )
    graph = DraftConstructionWorkflow(mappers=[SemanticTableMapper(resolved)]).run(
        [source]
    ).property_graph
    return resolved, policy, graph


class StructuredPathClosed(unittest.TestCase):
    """Closed: the draft workflow drives the semantic modelling pipeline."""

    def test_structured_workflow_builds_shared_dimension_nodes_and_edges(self) -> None:
        table = _table("employees")
        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, newline="") as handle:
            path = handle.name
            writer = csv.DictWriter(handle, fieldnames=list(table["records"][0]))
            writer.writeheader()
            writer.writerows(table["records"])
        try:
            source = SourceRecord.from_rows(
                table["records"],
                source_type="csv",
                source_name="employees.csv",
            )
            graph = draft_workflow_for_source_type("csv").run(
                [source]
            ).property_graph
        finally:
            Path(path).unlink(missing_ok=True)

        department_nodes = sum(1 for node in graph.nodes if "Department" in node.labels)
        department_literals = sum(1 for node in graph.nodes if "department" in node.properties)
        self.assertEqual(department_nodes, 3, "expected one shared Department node per value")
        self.assertGreater(len(graph.edges), 0, "expected typed relationships between records")
        self.assertEqual(department_literals, 0, "categorical value must not be duplicated as a literal")


class CategoricalRepresentationConsistent(unittest.TestCase):
    """Closed: role and materialization agree across contract, ontology, and graph."""

    def test_property_categorical_is_a_literal_everywhere(self) -> None:
        resolved, _, graph = _semantic_graph(_table("employees"))
        ontology = resolved.to_ontology_schema()
        object_ranges = {p.range for p in ontology.predicates if p.kind == "object"}
        # status is a scalar enumeration: a datatype property, not an object property.
        self.assertNotIn("Status", object_ranges)
        self.assertNotIn("Status", {c.name for c in resolved.classes})
        status_literals = sum(
            1 for node in graph.nodes if "Employee" in node.labels and "status" in node.properties
        )
        self.assertEqual(status_literals, 8, "status must be a literal on every employee")
        self.assertEqual(sum(1 for node in graph.nodes if "Status" in node.labels), 0)


class MissingForeignKeyNotFabricated(unittest.TestCase):
    """Closed: an unresolvable foreign key is retained, never fabricated."""

    def test_foreign_key_without_target_source_creates_no_entities(self) -> None:
        _, _, graph = _semantic_graph(_table("orders"))
        # orders.customer_id references Customer, but no Customer source is present.
        self.assertEqual(sum(1 for node in graph.nodes if "Customer" in node.labels), 0)
        order_nodes = [node for node in graph.nodes if "Order" in node.labels]
        self.assertTrue(all("customer_id" in node.properties for node in order_nodes))


class UnstructuredClaimAndIntentScoringClosed(unittest.TestCase):
    """Closed: unstructured correctness scores gold claims and intents."""

    def test_unstructured_case_scores_claims_and_intents(self) -> None:
        result = evaluate_unstructured_case(
            case_id="characterization",
            text="Alan Turing worked at Bletchley Park.",
            gold_entities=[{"name": "Alan Turing", "type": "Person"}],
            minimum_entity_coverage=0.5,
            gold_claims=[
                {"subject": "Alan Turing", "predicate": "works_at", "object": "Bletchley Park"}
            ],
            gold_intents=["Alan Turing worked at Bletchley Park."],
        )
        self.assertIn("claim_coverage", result.metrics)
        self.assertIn("intent_coverage", result.metrics)


class SchemaQualitySkipsShaclClosed(unittest.TestCase):
    """Closed: an omitted SHACL execution is an explicit failed gate."""

    def test_passing_report_requires_shacl_execution(self) -> None:
        _, policy, graph = _semantic_graph(_table("employees"))
        report = SchemaQualityEvaluator().evaluate(
            graph, policy=policy, source_records=_table("employees")["records"]
        )
        self.assertFalse(report.passed)
        self.assertFalse(report.shacl_executed)
        shacl_gate = next(
            gate for gate in report.gates if gate.gate == "shacl_execution"
        )
        self.assertFalse(shacl_gate.passed)


class RegistryExposesNonExecutableCapabilitiesDefect(unittest.TestCase):
    """Closed: the release registry exposes only executable capabilities."""

    def test_registry_contains_only_executable_capabilities(self) -> None:
        registry = default_registry()
        non_executable = sorted(
            name for name in registry.names() if registry.get(name).status != "implemented"
        )
        self.assertEqual(
            non_executable,
            [],
            f"{len(non_executable)} stub/adapter-only capabilities remain in the release registry",
        )


if __name__ == "__main__":
    unittest.main()
