"""Stage-2 exit gate: structured and multi-table conversion through the public path.

Covers: the employee gold fixture through ``construct_kg`` (8 employees, 3
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

from kg_construction import construct_kg
from kg_construction.application.resolved_plan import IdentityScope
from kg_construction.application.semantic_table_mapper import SemanticTableMapper
from kg_construction.application.source_catalog import (
    DeclaredForeignKey,
    SourceCatalog,
    TableDeclaration,
)
from kg_construction.application.workflow import ConstructionWorkflow
from kg_construction.domain.modeling import ControlledVocabulary
from kg_construction.domain.models import SourceRecord

_FIXTURE = Path(__file__).parent / "fixtures" / "gold_standards" / "column_roles.json"


def _employees() -> list[dict]:
    tables = json.loads(_FIXTURE.read_text(encoding="utf-8"))["tables"]
    return next(t for t in tables if t["table_name"] == "employees")["records"]


def _employee_source(source_name: str = "employees.csv") -> SourceRecord:
    return SourceRecord.from_rows(_employees(), source_type="csv", source_name=source_name)


class ExitGateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.graph = construct_kg(_employee_source()).property_graph
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
        from kg_construction.application.config import DataToKGConfig

        config = DataToKGConfig()
        config.pipeline.structured_mapping = "direct"
        baseline = construct_kg(_employee_source(), config=config).property_graph
        # The baseline duplicates the categorical literal; the default does not.
        baseline_employees = [n for n in baseline.nodes if "department" in n.properties]
        self.assertTrue(baseline_employees)


class ReingestionDeterminismTests(unittest.TestCase):
    def test_second_ingestion_produces_zero_semantic_delta(self) -> None:
        first = construct_kg(_employee_source()).property_graph
        second = construct_kg(_employee_source()).property_graph
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


class _ByNameMapper:
    """Route each source to its own resolved mapper by source name (test helper)."""

    name = "by_name_mapper"

    def __init__(self, mappers: dict[str, SemanticTableMapper]) -> None:
        self._mappers = mappers

    def map_source(self, source: SourceRecord):
        return self._mappers[source.source_name].map_source(source)


if __name__ == "__main__":
    unittest.main()
