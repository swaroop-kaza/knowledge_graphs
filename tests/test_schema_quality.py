"""Schema and instance quality metrics and benchmark gate wiring."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from kg_construction.application.benchmark_runner import (
    BenchmarkHardGateSettings,
    GoldEvaluationSummary,
    TechniqueEvaluationConfig,
    TechniqueEvaluationRunner,
)
from kg_construction.application.plan_resolution import resolve_modeling_plan
from kg_construction.application.semantic_profiler import SemanticColumnProfiler
from kg_construction.application.semantic_table_mapper import SemanticTableMapper
from kg_construction.application.workflow import ConstructionWorkflow
from kg_construction.domain.graph_objects import PropertyEdge, PropertyGraph, PropertyNode
from kg_construction.domain.modeling import (
    Cardinality,
    OntologyClass,
    OntologyPredicate,
    OntologySchema,
)
from kg_construction.domain.models import SourceRecord
from kg_construction.infrastructure.evaluation.schema_quality import (
    SchemaQualityEvaluator,
    SchemaQualityGateSettings,
)
from kg_construction.infrastructure.ontology.structured_ontology import propose_structured_ontology

_FIXTURE = Path(__file__).parent / "fixtures" / "gold_standards" / "column_roles.json"


def _employees() -> list[dict]:
    tables = json.loads(_FIXTURE.read_text(encoding="utf-8"))["tables"]
    return next(t for t in tables if t["table_name"] == "employees")["records"]


def _semantic_graph_policy_ontology():
    records = _employees()
    source = SourceRecord.from_rows(records, source_type="csv", source_name="employees.csv")
    modeling_plan = SemanticColumnProfiler().profile(records, table_name="employees", record_type="Employee")
    policy = modeling_plan.to_modeling_policy()
    resolved = resolve_modeling_plan(modeling_plan, source_name="employees.csv")
    ontology = propose_structured_ontology(policy).ontology
    graph = ConstructionWorkflow(mappers=[SemanticTableMapper(resolved)]).run([source]).property_graph
    return graph, policy, records, ontology


class SchemaQualityAcceptanceTests(unittest.TestCase):
    def test_stage2_semantic_graph_is_conciseness_perfect_and_passes(self) -> None:
        graph, policy, records, ontology = _semantic_graph_policy_ontology()
        report = SchemaQualityEvaluator().evaluate(
            graph,
            policy=policy,
            source_records=records,
            ontology=ontology,
            shacl_executed=True,
            shacl_violation_count=0,
        )
        self.assertEqual(report.categorical_duplication_rate, 0.0)
        self.assertEqual(report.literal_that_should_be_node_rate, 0.0)
        self.assertEqual(report.duplicate_node_rate, 0.0)
        self.assertEqual(report.orphan_rate, 0.0)
        self.assertEqual(report.consistency_violation_count, 0)
        self.assertTrue(report.passed)
        self.assertTrue(all(gate.passed for gate in report.gates))

    def test_literal_categorical_graph_fails_the_categorical_gate(self) -> None:
        _, policy, records, _ = _semantic_graph_policy_ontology()
        # department stored as a repeated literal (direct-mapper anti-pattern), no edges.
        nodes = [
            PropertyNode(
                node_id=f"emp{i}",
                labels=["Entity", "Employee"],
                properties={"name": str(row["emp_id"]), "department": row["department"]},
            )
            for i, row in enumerate(records)
        ]
        bad = PropertyGraph(nodes=nodes, edges=[])
        report = SchemaQualityEvaluator().evaluate(bad, policy=policy, source_records=records)
        self.assertGreater(report.categorical_duplication_rate or 0.0, 0.0)
        self.assertFalse(report.passed)
        categorical_gate = next(g for g in report.gates if g.gate == "categorical_conciseness")
        self.assertFalse(categorical_gate.passed)


class OntoQAStructuralTests(unittest.TestCase):
    def test_structural_metrics(self) -> None:
        graph, _, _, ontology = _semantic_graph_policy_ontology()
        report = SchemaQualityEvaluator().evaluate(graph, ontology=ontology)
        self.assertEqual(report.class_population.get("Employee"), 8)
        self.assertEqual(report.class_population.get("Department"), 3)
        self.assertEqual(report.connected_component_count, 1)  # all employees reachable
        self.assertGreater(report.average_degree, 0.0)
        self.assertGreater(report.attribute_richness, 0.0)
        self.assertGreaterEqual(report.class_richness or 0.0, 0.0)


class ConsistencyTests(unittest.TestCase):
    def test_domain_range_violation_is_counted(self) -> None:
        ontology = OntologySchema(
            classes=[OntologyClass(name="Employee", role="record"), OntologyClass(name="Organization", role="reference")],
            predicates=[
                OntologyPredicate(name="works_at", domain="Employee", range="Organization", kind="object", cardinality=Cardinality.zero_or_many()),
            ],
        )
        # An edge that violates the range (target is an Employee, not an Organization).
        nodes = [
            PropertyNode(node_id="a", labels=["Entity", "Employee"], properties={"name": "A"}),
            PropertyNode(node_id="b", labels=["Entity", "Employee"], properties={"name": "B"}),
        ]
        edges = [PropertyEdge(edge_id="e", source_id="a", target_id="b", type="works_at")]
        graph = PropertyGraph(nodes=nodes, edges=edges)
        report = SchemaQualityEvaluator().evaluate(graph, ontology=ontology)
        self.assertGreater(report.consistency_violation_count, 0)
        consistency_gate = next(g for g in report.gates if g.gate == "consistency")
        self.assertFalse(consistency_gate.passed)

    def test_shacl_violation_count_flows_into_consistency(self) -> None:
        graph, _, _, ontology = _semantic_graph_policy_ontology()
        report = SchemaQualityEvaluator().evaluate(graph, ontology=ontology, shacl_violation_count=2)
        self.assertEqual(report.shacl_violation_count, 2)
        self.assertGreaterEqual(report.consistency_violation_count, 2)


class BenchmarkGateWiringTests(unittest.TestCase):
    def _runner(self, *, enforce: bool) -> TechniqueEvaluationRunner:
        config = TechniqueEvaluationConfig(hard_gates=BenchmarkHardGateSettings(enforce_schema_quality_gates=enforce))
        return TechniqueEvaluationRunner(config=config)

    def test_schema_quality_gates_are_off_by_default(self) -> None:
        self.assertFalse(BenchmarkHardGateSettings().enforce_schema_quality_gates)
        gates = self._runner(enforce=False)._evaluate_hard_gates(
            {"orphan_rate": 1.0, "categorical_duplication_rate": 0.9}, GoldEvaluationSummary()
        )
        self.assertFalse(any(g.gate.startswith("schema_") or g.gate == "categorical_conciseness" for g in gates))

    def test_schema_quality_gates_enforced_when_enabled(self) -> None:
        gates = self._runner(enforce=True)._evaluate_hard_gates(
            {
                "orphan_rate": 1.0,
                "duplicate_node_rate": 0.0,
                "consistency_violation_count": 0.0,
                "categorical_duplication_rate": 0.5,
            },
            GoldEvaluationSummary(),
        )
        by_gate = {g.gate: g for g in gates}
        self.assertIn("schema_connectivity", by_gate)
        self.assertIn("categorical_conciseness", by_gate)
        self.assertFalse(by_gate["schema_connectivity"].passed)  # orphan_rate 1.0 > 0.05
        self.assertFalse(by_gate["categorical_conciseness"].passed)  # 0.5 > 0.0


class ReportPayloadTests(unittest.TestCase):
    def test_structural_metrics_payload_for_runner(self) -> None:
        graph, _, _, _ = _semantic_graph_policy_ontology()
        payload = SchemaQualityEvaluator().structural_metrics(graph)
        self.assertIn("orphan_rate", payload)
        self.assertIn("duplicate_node_rate", payload)
        self.assertNotIn("categorical_duplication_rate", payload)  # needs a policy

    def test_custom_gate_settings(self) -> None:
        graph, policy, records, _ = _semantic_graph_policy_ontology()
        settings = SchemaQualityGateSettings(maximum_orphan_rate=0.0)
        report = SchemaQualityEvaluator().evaluate(
            graph,
            policy=policy,
            source_records=records,
            shacl_executed=True,
            shacl_violation_count=0,
            gate_settings=settings,
        )
        self.assertTrue(report.passed)  # orphan_rate is 0.0


if __name__ == "__main__":
    unittest.main()
