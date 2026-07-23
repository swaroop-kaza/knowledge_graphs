"""Characterization tests that pin known modelling defects.

Each test asserts the *desired* behaviour and is marked ``expectedFailure``
while the defect is present, so the suite stays green today. When a later change
fixes a defect the test becomes an *unexpected success*, which fails the run and
forces the marker to be removed — that is the signal the gap is closed.

Every case here corresponds to a defect reproduced against commit 42d0100:

1. The public ``construct_kg`` structured path uses ``DirectTableMapper``, so it
   emits duplicated categorical literals and no relationships.
2. A categorical column can be a policy node, an ontology object property, and a
   mapper literal at the same time (the three representations disagree).
3. A foreign key whose target source is absent silently fabricates placeholder
   entities.
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
    policy = (
        SemanticColumnProfiler()
        .profile(table["records"], table_name=table["table_name"], record_type=table["record_type"])
        .to_modeling_policy()
    )
    source = SourceRecord.from_rows(
        table["records"], source_type="csv", source_name=f"{table['table_name']}.csv"
    )
    graph = ConstructionWorkflow(mappers=[SemanticTableMapper(policy)]).run([source]).property_graph
    return policy, graph


class PublicStructuredPathDefect(unittest.TestCase):
    """The public API must drive the semantic modelling pipeline, not DirectTableMapper."""

    @unittest.expectedFailure
    def test_public_api_builds_shared_dimension_nodes_and_edges(self) -> None:
        table = _table("employees")
        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, newline="") as handle:
            path = handle.name
            writer = csv.DictWriter(handle, fieldnames=list(table["records"][0]))
            writer.writeheader()
            writer.writerows(table["records"])
        try:
            graph = construct_kg(path, source_type="csv").property_graph
        finally:
            Path(path).unlink(missing_ok=True)

        department_nodes = sum(1 for node in graph.nodes if "Department" in node.labels)
        department_literals = sum(1 for node in graph.nodes if "department" in node.properties)
        self.assertGreater(department_nodes, 0, "expected one shared Department node per value")
        self.assertGreater(len(graph.edges), 0, "expected typed relationships between records")
        self.assertEqual(department_literals, 0, "categorical value must not be duplicated as a literal")


class CategoricalRepresentationDisagreementDefect(unittest.TestCase):
    """Policy, ontology, and mapper must agree on how a categorical is materialised."""

    @unittest.expectedFailure
    def test_object_property_categorical_is_not_stored_as_a_bare_literal(self) -> None:
        table = _table("employees")
        policy, graph = _semantic_graph(table)
        ontology = policy.to_ontology_schema()

        object_ranges = {
            predicate.range
            for predicate in ontology.predicates
            if getattr(predicate, "kind", None) == "object"
        }
        self.assertIn("Status", object_ranges, "ontology declares has_status as an object property")

        status_nodes = sum(1 for node in graph.nodes if "Status" in node.labels)
        status_literals = sum(
            1 for node in graph.nodes if "Employee" in node.labels and "status" in node.properties
        )
        self.assertFalse(
            status_nodes == 0 and status_literals > 0,
            "status is an ontology object property but the mapper stored it as a literal",
        )


class MissingForeignKeyPlaceholderDefect(unittest.TestCase):
    """A foreign key with no target source must not fabricate placeholder entities."""

    @unittest.expectedFailure
    def test_foreign_key_without_target_source_creates_no_entities(self) -> None:
        table = _table("orders")
        _, graph = _semantic_graph(table)
        # orders.customer_id references Customer, but no Customer source is present.
        customer_nodes = sum(1 for node in graph.nodes if "Customer" in node.labels)
        self.assertEqual(
            customer_nodes,
            0,
            "missing foreign-key targets must not silently create placeholder entities",
        )


class UnstructuredClaimAndIntentScoringGap(unittest.TestCase):
    """Unstructured correctness must score gold claims and intents, not just entities."""

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
    """A passing schema-quality report must have actually executed SHACL."""

    @unittest.expectedFailure
    def test_passing_report_requires_shacl_execution(self) -> None:
        table = _table("employees")
        policy, graph = _semantic_graph(table)
        report = SchemaQualityEvaluator().evaluate(
            graph, policy=policy, source_records=table["records"]
        )
        if not report.passed:
            self.skipTest("report did not pass; SHACL-skip gate is only relevant on a pass")
        self.assertIsNotNone(
            report.shacl_violation_count,
            "a passing report must reflect an executed SHACL validation, not a skipped one",
        )


class RegistryExposesNonExecutableCapabilitiesDefect(unittest.TestCase):
    """The release registry must expose only executable capabilities."""

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
