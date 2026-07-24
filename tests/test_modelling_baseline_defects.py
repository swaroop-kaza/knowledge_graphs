"""Modelling-defect gate: closed defects stay closed, open ones stay tracked.

Each case was a defect reproduced against commit 42d0100. Cases fixed by a later
change assert the corrected behaviour directly (a regression re-breaks the run).
Cases still open assert the *desired* behaviour under ``expectedFailure``, so they
flip to an unexpected success — and fail the run — the moment they are fixed.

Closed by the structured redesign:

1. The public ``construct_kg`` structured path now drives the semantic pipeline
   (shared dimension nodes, typed relationships, no duplicated literals).
2. A categorical's semantic role and physical materialization agree across the
   contract, ontology, and graph.
3. A foreign key whose target is absent is retained as a value, never fabricated
   as a placeholder entity.

Still open:

4. Unstructured correctness ignores the gold claims and intents in the fixture.
5. Schema quality can report ``passed`` without SHACL ever executing.
6. The release registry still exposes stub and adapter-only capabilities.
"""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from kg_construction import construct_kg, default_registry
from kg_construction.application.modelling_correctness import evaluate_unstructured_case
from kg_construction.application.plan_resolution import resolve_modeling_plan
from kg_construction.application.semantic_profiler import SemanticColumnProfiler
from kg_construction.application.semantic_table_mapper import SemanticTableMapper
from kg_construction.application.workflow import ConstructionWorkflow
from kg_construction.domain.models import SourceRecord
from kg_construction.infrastructure.evaluation.schema_quality import SchemaQualityEvaluator

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
    graph = ConstructionWorkflow(mappers=[SemanticTableMapper(resolved)]).run([source]).property_graph
    return resolved, policy, graph


class PublicStructuredPathClosed(unittest.TestCase):
    """Closed: the public API drives the semantic modelling pipeline."""

    def test_public_api_builds_shared_dimension_nodes_and_edges(self) -> None:
        table = _table("employees")
        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, newline="") as handle:
            path = handle.name
            writer = csv.DictWriter(handle, fieldnames=list(table["records"][0]))
            writer.writeheader()
            writer.writerows(table["records"])
        try:
            graph = construct_kg(
                SourceRecord.from_rows(table["records"], source_type="csv", source_name="employees.csv")
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


class UnstructuredClaimAndIntentScoringGap(unittest.TestCase):
    """Open: unstructured correctness must score gold claims and intents."""

    @unittest.expectedFailure
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


class SchemaQualitySkipsShaclDefect(unittest.TestCase):
    """Open: a passing schema-quality report must have actually executed SHACL."""

    @unittest.expectedFailure
    def test_passing_report_requires_shacl_execution(self) -> None:
        _, policy, graph = _semantic_graph(_table("employees"))
        report = SchemaQualityEvaluator().evaluate(
            graph, policy=policy, source_records=_table("employees")["records"]
        )
        if not report.passed:
            self.skipTest("report did not pass; SHACL-skip gate is only relevant on a pass")
        self.assertIsNotNone(
            report.shacl_violation_count,
            "a passing report must reflect an executed SHACL validation, not a skipped one",
        )


class RegistryExposesNonExecutableCapabilitiesDefect(unittest.TestCase):
    """Open: the release registry must expose only executable capabilities."""

    @unittest.expectedFailure
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
