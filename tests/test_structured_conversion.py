"""Stage-2 exit gate: structured and multi-table modelling in draft workflows.

Covers: the employee gold fixture through the semantic workflow (8 employees, 3
departments, 8 department relationships, 7 manager relationships, no duplicated
department literals); 100% declared key and foreign-key mapping; deterministic
re-ingestion (zero semantic delta); and cross-file business keys resolving by
declared identity scope. Also exercises the source catalog and governed
dictionaries.
"""

from __future__ import annotations

import json
import unittest
from collections import Counter
from pathlib import Path

from kg_construction import default_workflow_for_source_type
from kg_construction.application.config import DataToKGConfig
from kg_construction.application.resolved_plan import ClassKind, IdentityScope
from kg_construction.application.semantic_table_mapper import SemanticTableMapper
from kg_construction.application.source_catalog import (
    CatalogTableMapper,
    DeclaredForeignKey,
    SourceCatalog,
    TableDeclaration,
)
from kg_construction.application.workflow import ConstructionWorkflow
from kg_construction.domain.modeling import ControlledVocabulary
from kg_construction.domain.models import SourceRecord
from kg_construction.infrastructure.resolution.exact import ExactKeyEntityResolver

_FIXTURE = Path(__file__).parent / "fixtures" / "gold_standards" / "column_roles.json"


def _employees() -> list[dict]:
    tables = json.loads(_FIXTURE.read_text(encoding="utf-8"))["tables"]
    return next(t for t in tables if t["table_name"] == "employees")["records"]


def _employee_source(source_name: str = "employees.csv") -> SourceRecord:
    return SourceRecord.from_rows(_employees(), source_type="csv", source_name=source_name)


def _draft_result(
    source: SourceRecord,
    config: DataToKGConfig | None = None,
):
    return default_workflow_for_source_type("csv", config).run([source])


class ExitGateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.graph = _draft_result(_employee_source()).property_graph
        self.labels = Counter(label for node in self.graph.nodes for label in node.labels)
        self.edges = Counter(edge.type for edge in self.graph.edges)

    def test_employee_gold_fixture_shape(self) -> None:
        self.assertEqual(self.labels["Employee"], 8)
        self.assertEqual(self.labels["Department"], 3)
        self.assertEqual(self.edges["has_department"], 8)
        self.assertEqual(self.edges["manager"], 7)

    def test_no_department_literal_is_duplicated(self) -> None:
        employees = [node for node in self.graph.nodes if "Employee" in node.labels]
        self.assertFalse(any("department" in node.properties for node in employees))

    def test_source_row_identity_is_separate_from_business_identity(self) -> None:
        employees = [node for node in self.graph.nodes if "Employee" in node.labels]
        self.assertTrue(all("source_row_id" in node.properties for node in employees))
        self.assertTrue(all("business_key" in node.properties for node in employees))

    def test_direct_baseline_is_available_but_not_the_default(self) -> None:
        config = DataToKGConfig()
        config.pipeline.structured_mapping = "direct"
        baseline = _draft_result(_employee_source(), config=config).property_graph
        # The baseline duplicates the categorical literal; the default does not.
        baseline_employees = [n for n in baseline.nodes if "department" in n.properties]
        self.assertTrue(baseline_employees)


class ReingestionDeterminismTests(unittest.TestCase):
    def test_second_ingestion_produces_zero_semantic_delta(self) -> None:
        first = _draft_result(_employee_source()).property_graph
        second = _draft_result(_employee_source()).property_graph
        self.assertEqual(
            sorted(node.node_id for node in first.nodes),
            sorted(node.node_id for node in second.nodes),
        )
        self.assertEqual(
            sorted((e.source_id, e.type, e.target_id) for e in first.edges),
            sorted((e.source_id, e.type, e.target_id) for e in second.edges),
        )


class DeclaredMappingCoverageTests(unittest.TestCase):
    def test_declared_keys_and_foreign_keys_map_fully(self) -> None:
        catalog = SourceCatalog().declare(
            TableDeclaration(
                table_name="employees.csv",
                record_type="Employee",
                primary_key=["emp_id"],
                foreign_keys=[DeclaredForeignKey(columns=["manager_id"], target_table="employees")],
            )
        )
        plan = catalog.resolved_plan(_employee_source())
        self.assertEqual(plan.clazz("Employee").identity_keys, ["emp_id"])
        self.assertTrue(any(fk.columns == ["manager_id"] for fk in plan.foreign_keys))
        coverage = catalog.declared_key_mapping_coverage({"Employee": plan})
        self.assertEqual(coverage, 1.0)


class CrossFileIdentityScopeTests(unittest.TestCase):
    def _plan(self, source_name: str, scope: IdentityScope):
        catalog = SourceCatalog().declare(
            TableDeclaration(
                table_name=source_name,
                record_type="Employee",
                primary_key=["emp_id"],
                identity_scope=scope,
            )
        )
        return catalog.resolved_plan(_employee_source(source_name))

    def _employee_node_count(self, scope: IdentityScope) -> int:
        sources = [_employee_source("file_a.csv"), _employee_source("file_b.csv")]
        names = ("file_a.csv", "file_b.csv")
        by_name = {name: SemanticTableMapper(self._plan(name, scope)) for name in names}
        # One mapper per source, matched by source name at run time.
        graph = ConstructionWorkflow(mappers=[_ByNameMapper(by_name)]).run(sources).property_graph
        return sum(1 for node in graph.nodes if "Employee" in node.labels)

    def test_global_scope_merges_equal_business_keys_across_files(self) -> None:
        self.assertEqual(self._employee_node_count(IdentityScope.GLOBAL), 8)

    def test_source_scope_keeps_files_distinct(self) -> None:
        self.assertEqual(self._employee_node_count(IdentityScope.SOURCE), 16)


class SourceCatalogTests(unittest.TestCase):
    def test_database_metadata_becomes_a_declaration(self) -> None:
        declaration = SourceCatalog.from_database_metadata(
            {
                "table_name": "employees",
                "record_type": "Employee",
                "primary_key": ["emp_id"],
                "foreign_keys": [{"columns": ["manager_id"], "target_table": "employees"}],
                "identity_scope": "global",
            }
        )
        self.assertEqual(declaration.primary_key, ["emp_id"])
        self.assertEqual(declaration.identity_scope, IdentityScope.GLOBAL)
        self.assertEqual(declaration.foreign_keys[0].columns, ["manager_id"])

    def test_csvw_metadata_becomes_a_declaration(self) -> None:
        declaration = SourceCatalog.from_csvw(
            {
                "table_name": "employees.csv",
                "record_type": "Employee",
                "tableSchema": {
                    "columns": [
                        {"name": "emp_id", "datatype": "integer"},
                        {"name": "salary", "datatype": "decimal"},
                    ],
                    "primaryKey": "emp_id",
                },
            }
        )
        self.assertEqual(declaration.primary_key, ["emp_id"])
        self.assertEqual(declaration.datatypes["salary"], "number")

    def test_governed_dictionary_records_code_and_raw_observation(self) -> None:
        catalog = SourceCatalog().declare(
            TableDeclaration(
                table_name="employees.csv",
                record_type="Employee",
                primary_key=["emp_id"],
                controlled_vocabularies=[
                    ControlledVocabulary(
                        name="Department",
                        permissible_values=["engineering", "sales", "support"],
                        aliases={"eng": "engineering"},
                        closed=True,
                    )
                ],
            )
        )
        plan = catalog.resolved_plan(_employee_source())
        source = _employee_source()
        batch = SemanticTableMapper(plan).map_source(source)
        departments = [e for e in batch.entities if e.entity_type == "Department"]
        self.assertTrue(departments)
        self.assertTrue(all("code" in e.properties and "raw_value" in e.properties for e in departments))

    def test_cross_table_foreign_key_links_to_the_actual_target_entity(self) -> None:
        customers = SourceRecord.from_rows(
            [
                {"customer_id": "C-17", "name": "Analytical Engines"},
                {"customer_id": "C-42", "name": "Computing Machinery"},
            ],
            source_type="csv",
            source_name="customers.csv",
        )
        orders = SourceRecord.from_rows(
            [
                {"order_id": "O-1", "customer_id": "C-17", "amount": 10},
                {"order_id": "O-2", "customer_id": "C-42", "amount": 20},
                {"order_id": "O-3", "customer_id": "MISSING", "amount": 30},
            ],
            source_type="csv",
            source_name="orders.csv",
        )
        catalog = (
            SourceCatalog()
            .declare(
                TableDeclaration(
                    table_name="customers.csv",
                    record_type="Customer",
                    primary_key=["customer_id"],
                )
            )
            .declare(
                TableDeclaration(
                    table_name="orders.csv",
                    record_type="Order",
                    primary_key=["order_id"],
                    foreign_keys=[
                        DeclaredForeignKey(
                            columns=["customer_id"],
                            target_table="customers.csv",
                            target_key=["customer_id"],
                            relationship_name="placed_by",
                        )
                    ],
                )
            )
        )
        result = ConstructionWorkflow(
            mappers=[CatalogTableMapper(catalog)],
            resolvers=[ExactKeyEntityResolver()],
        ).run([customers, orders])
        graph = result.property_graph

        customer_ids = {node.node_id for node in graph.nodes if "Customer" in node.labels}
        order_ids = {node.node_id for node in graph.nodes if "Order" in node.labels}
        placed_by = [edge for edge in graph.edges if edge.type == "placed_by"]
        self.assertEqual(len(customer_ids), 2)
        self.assertEqual(len(order_ids), 3)
        self.assertEqual(len(placed_by), 2)
        self.assertTrue(all(edge.source_id in order_ids for edge in placed_by))
        self.assertTrue(all(edge.target_id in customer_ids for edge in placed_by))
        missing_order = next(
            node for node in graph.nodes if node.properties.get("order_id") == "O-3"
        )
        self.assertEqual(missing_order.properties["customer_id"], "MISSING")

    def test_composite_foreign_key_uses_collision_safe_key_serialization(self) -> None:
        accounts = SourceRecord.from_rows(
            [
                {"country": "US", "account_no": "A|1", "name": "First"},
                {"country": "US|A", "account_no": "1", "name": "Second"},
            ],
            source_type="csv",
            source_name="accounts.csv",
        )
        transactions = SourceRecord.from_rows(
            [
                {"txn_id": "T1", "country": "US", "account_no": "A|1"},
                {"txn_id": "T2", "country": "US|A", "account_no": "1"},
            ],
            source_type="csv",
            source_name="transactions.csv",
        )
        catalog = (
            SourceCatalog()
            .declare(
                TableDeclaration(
                    table_name="accounts.csv",
                    record_type="Account",
                    primary_key=["country", "account_no"],
                )
            )
            .declare(
                TableDeclaration(
                    table_name="transactions.csv",
                    record_type="Transaction",
                    primary_key=["txn_id"],
                    foreign_keys=[
                        DeclaredForeignKey(
                            columns=["country", "account_no"],
                            target_table="accounts.csv",
                            target_key=["country", "account_no"],
                            relationship_name="posted_to",
                        )
                    ],
                )
            )
        )
        graph = ConstructionWorkflow(
            mappers=[CatalogTableMapper(catalog)],
            resolvers=[ExactKeyEntityResolver()],
        ).run([accounts, transactions]).property_graph

        account_ids = {node.node_id for node in graph.nodes if "Account" in node.labels}
        posted_to = [edge for edge in graph.edges if edge.type == "posted_to"]
        self.assertEqual(len(account_ids), 2)
        self.assertEqual(len(posted_to), 2)
        self.assertEqual({edge.target_id for edge in posted_to}, account_ids)

    def test_pure_two_foreign_key_join_table_becomes_one_relationship(self) -> None:
        people = SourceRecord.from_rows(
            [{"person_id": "P1", "name": "Ada"}],
            source_type="csv",
            source_name="people.csv",
        )
        projects = SourceRecord.from_rows(
            [{"project_id": "G1", "name": "Graph"}],
            source_type="csv",
            source_name="projects.csv",
        )
        memberships = SourceRecord.from_rows(
            [{"person_id": "P1", "project_id": "G1"}],
            source_type="csv",
            source_name="memberships.csv",
        )
        catalog = (
            SourceCatalog()
            .declare(
                TableDeclaration(
                    table_name="people.csv",
                    record_type="Person",
                    primary_key=["person_id"],
                )
            )
            .declare(
                TableDeclaration(
                    table_name="projects.csv",
                    record_type="Project",
                    primary_key=["project_id"],
                )
            )
            .declare(
                TableDeclaration(
                    table_name="memberships.csv",
                    record_type="Membership",
                    primary_key=["person_id", "project_id"],
                    foreign_keys=[
                        DeclaredForeignKey(
                            columns=["person_id"],
                            target_table="people.csv",
                            target_key=["person_id"],
                        ),
                        DeclaredForeignKey(
                            columns=["project_id"],
                            target_table="projects.csv",
                            target_key=["project_id"],
                        ),
                    ],
                    association_relationship_name="assigned_to",
                )
            )
        )
        graph = ConstructionWorkflow(
            mappers=[CatalogTableMapper(catalog)],
            resolvers=[ExactKeyEntityResolver()],
        ).run([people, projects, memberships]).property_graph

        self.assertFalse(any("Membership" in node.labels for node in graph.nodes))
        assigned = [edge for edge in graph.edges if edge.type == "assigned_to"]
        self.assertEqual(len(assigned), 1)
        self.assertIn("Person", next(node.labels for node in graph.nodes if node.node_id == assigned[0].source_id))
        self.assertIn("Project", next(node.labels for node in graph.nodes if node.node_id == assigned[0].target_id))

    def test_qualified_join_table_becomes_an_association_node(self) -> None:
        catalog = (
            SourceCatalog()
            .declare(
                TableDeclaration(
                    table_name="people.csv",
                    record_type="Person",
                    primary_key=["person_id"],
                )
            )
            .declare(
                TableDeclaration(
                    table_name="projects.csv",
                    record_type="Project",
                    primary_key=["project_id"],
                )
            )
            .declare(
                TableDeclaration(
                    table_name="assignments.csv",
                    record_type="Assignment",
                    primary_key=["person_id", "project_id"],
                    foreign_keys=[
                        DeclaredForeignKey(
                            columns=["person_id"],
                            target_table="people.csv",
                            target_key=["person_id"],
                            relationship_name="has_person",
                        ),
                        DeclaredForeignKey(
                            columns=["project_id"],
                            target_table="projects.csv",
                            target_key=["project_id"],
                            relationship_name="has_project",
                        ),
                    ],
                )
            )
        )
        assignment_source = SourceRecord.from_rows(
            [{"person_id": "P1", "project_id": "G1", "role": "lead"}],
            source_type="csv",
            source_name="assignments.csv",
        )
        plan = catalog.resolved_plan(assignment_source)

        self.assertEqual(plan.clazz("Assignment").kind, ClassKind.ASSOCIATION)


class _ByNameMapper:
    """Route each source to its own resolved mapper by source name (test helper)."""

    name = "by_name_mapper"

    def __init__(self, mappers: dict[str, SemanticTableMapper]) -> None:
        self._mappers = mappers

    def map_source(self, source: SourceRecord):
        return self._mappers[source.source_name].map_source(source)


if __name__ == "__main__":
    unittest.main()
