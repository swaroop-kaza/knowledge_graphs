"""The policy-driven semantic table mapper: dedup, typed edges, round-trip."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from kg_construction.application.plan_resolution import resolve_modeling_plan
from kg_construction.application.semantic_profiler import SemanticColumnProfiler
from kg_construction.application.semantic_table_mapper import SemanticTableMapper
from kg_construction.application.workflow import ConstructionWorkflow
from kg_construction.domain.graph_objects import PropertyGraph
from kg_construction.domain.models import SourceRecord

_FIXTURE = Path(__file__).parent / "fixtures" / "gold_standards" / "column_roles.json"


def _table(name: str) -> dict:
    tables = json.loads(_FIXTURE.read_text(encoding="utf-8"))["tables"]
    return next(t for t in tables if t["table_name"] == name)


def _resolved(table_name: str, record_type: str):
    table = _table(table_name)
    plan = SemanticColumnProfiler().profile(
        table["records"], table_name=table_name, record_type=record_type
    )
    return resolve_modeling_plan(plan, source_name=f"{table_name}.csv")


def _graph(table_name: str, record_type: str) -> tuple[PropertyGraph, dict]:
    table = _table(table_name)
    records = table["records"]
    source = SourceRecord.from_rows(records, source_type="csv", source_name=f"{table_name}.csv")
    resolved = _resolved(table_name, record_type)
    result = ConstructionWorkflow(mappers=[SemanticTableMapper(resolved)]).run([source])
    return result.property_graph, {"records": records, "validation": result.validation_report}


def _nodes_with_label(graph: PropertyGraph, label: str) -> list:
    return [node for node in graph.nodes if label in node.labels]


def _edges_of_type(graph: PropertyGraph, edge_type: str) -> list:
    return [edge for edge in graph.edges if edge.type == edge_type]


class CategoricalDeduplicationTests(unittest.TestCase):
    def test_categorical_duplication_rate_is_zero(self) -> None:
        graph, ctx = _graph("employees", "Employee")
        department_nodes = _nodes_with_label(graph, "Department")
        distinct_departments = {str(row["department"]) for row in ctx["records"]}
        # One shared node per distinct value -> zero duplication.
        self.assertEqual(len(department_nodes), len(distinct_departments))
        duplication_rate = 1 - len(department_nodes) / len(distinct_departments)
        self.assertEqual(duplication_rate, 0.0)
        # The categorical is an edge, never a duplicated literal on the record node.
        employee_nodes = _nodes_with_label(graph, "Employee")
        self.assertTrue(employee_nodes)
        self.assertFalse(any("department" in node.properties for node in employee_nodes))

    def test_reference_candidates_share_a_canonical_id(self) -> None:
        table = _table("employees")
        source = SourceRecord.from_rows(table["records"], source_type="csv", source_name="employees.csv")
        batch = SemanticTableMapper(_resolved("employees", "Employee")).map_source(source)
        department_candidates = [e for e in batch.entities if e.entity_type == "Department"]
        # 8 rows emit a Department reference candidate, but they collapse to 3 ids.
        self.assertEqual(len(department_candidates), 8)
        canonical_ids = {e.properties["canonical_entity_id"] for e in department_candidates}
        self.assertEqual(len(canonical_ids), 3)


class SupernodeGuardTests(unittest.TestCase):
    def test_low_cardinality_categorical_stays_a_property(self) -> None:
        graph, _ = _graph("employees", "Employee")
        # status has 2 distinct values (<= keep_as_property_max_distinct) -> property.
        employee_nodes = _nodes_with_label(graph, "Employee")
        self.assertTrue(all("status" in node.properties for node in employee_nodes))
        self.assertEqual(_nodes_with_label(graph, "Status"), [])


class RelationshipFormationTests(unittest.TestCase):
    def test_foreign_key_becomes_a_typed_edge(self) -> None:
        graph, _ = _graph("employees", "Employee")
        manager_edges = _edges_of_type(graph, "manager")
        self.assertEqual(len(manager_edges), 7)  # 7 employees have a manager
        employee_ids = {node.node_id for node in _nodes_with_label(graph, "Employee")}
        for edge in manager_edges:
            self.assertIn(edge.source_id, employee_ids)
            self.assertIn(edge.target_id, employee_ids)  # self-referential edge
        # The FK value is not copied onto the record as a literal.
        self.assertFalse(any("manager_id" in node.properties for node in _nodes_with_label(graph, "Employee")))

    def test_categorical_becomes_a_typed_edge_to_shared_node(self) -> None:
        graph, _ = _graph("employees", "Employee")
        department_edges = _edges_of_type(graph, "has_department")
        self.assertEqual(len(department_edges), 8)  # every employee has a department
        department_ids = {node.node_id for node in _nodes_with_label(graph, "Department")}
        self.assertTrue(all(edge.target_id in department_ids for edge in department_edges))

    def test_unresolvable_foreign_key_is_retained_not_fabricated(self) -> None:
        # orders.customer_id references Customer, but no Customer source is present.
        graph, _ = _graph("orders", "Order")
        # The doctrine bans placeholder entities: no Customer nodes, no customer edges.
        self.assertEqual(_nodes_with_label(graph, "Customer"), [])
        self.assertEqual(_edges_of_type(graph, "customer"), [])
        # The reference value is retained on the Order so no data is lost.
        order_nodes = _nodes_with_label(graph, "Order")
        self.assertTrue(all("customer_id" in node.properties for node in order_nodes))
        # A categorical dimension still becomes a shared concept node.
        self.assertEqual(len(_nodes_with_label(graph, "Priority")), 3)
        self.assertTrue(_edges_of_type(graph, "has_priority"))


class PropertiesAndRoundTripTests(unittest.TestCase):
    def test_measures_and_temporals_are_node_properties(self) -> None:
        graph, _ = _graph("employees", "Employee")
        employee_nodes = _nodes_with_label(graph, "Employee")
        self.assertTrue(all("salary" in node.properties for node in employee_nodes))
        self.assertTrue(all("hired_on" in node.properties for node in employee_nodes))

    def test_validation_passes(self) -> None:
        _, ctx = _graph("employees", "Employee")
        self.assertTrue(ctx["validation"].valid)
        self.assertEqual(len(ctx["validation"].findings), 0)

    def test_property_graph_json_round_trip(self) -> None:
        graph, _ = _graph("employees", "Employee")
        restored = PropertyGraph.model_validate(graph.to_dict())
        self.assertEqual(len(restored.nodes), len(graph.nodes))
        self.assertEqual(len(restored.edges), len(graph.edges))
        self.assertEqual(
            {node.node_id for node in restored.nodes},
            {node.node_id for node in graph.nodes},
        )

    def test_mapping_is_deterministic(self) -> None:
        first, _ = _graph("employees", "Employee")
        second, _ = _graph("employees", "Employee")
        self.assertEqual(
            sorted(node.node_id for node in first.nodes),
            sorted(node.node_id for node in second.nodes),
        )
        self.assertEqual(
            sorted((e.source_id, e.type, e.target_id) for e in first.edges),
            sorted((e.source_id, e.type, e.target_id) for e in second.edges),
        )


if __name__ == "__main__":
    unittest.main()
