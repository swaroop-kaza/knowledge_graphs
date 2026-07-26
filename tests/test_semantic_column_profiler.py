"""Semantic column-role profiling, measured against a gold fixture."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from kg_construction.domain.modeling import ColumnRole
from kg_construction.structured.modeling_policy import ModelingPolicy, PromotionThresholds
from kg_construction.structured.semantic_profiler import (
    ModelingPlan,
    SemanticColumnProfiler,
)

_FIXTURE = Path(__file__).parent / "fixtures" / "gold_standards" / "column_roles.json"


def _load_gold() -> list[dict]:
    return json.loads(_FIXTURE.read_text(encoding="utf-8"))["tables"]


class GoldColumnRoleAccuracyTests(unittest.TestCase):
    def test_role_classification_matches_gold(self) -> None:
        profiler = SemanticColumnProfiler()
        total = 0
        correct = 0
        mismatches: list[str] = []
        for table in _load_gold():
            plan = profiler.profile(
                table["records"],
                table_name=table["table_name"],
                record_type=table["record_type"],
            )
            predicted = plan.roles()
            for column, gold_role in table["gold_roles"].items():
                total += 1
                got = predicted.get(column)
                if got is not None and got.value == gold_role:
                    correct += 1
                else:
                    mismatches.append(
                        f"{table['table_name']}.{column}: gold={gold_role} predicted={got.value if got else None}"
                    )
        accuracy = correct / total
        # The fixture is deterministic and should classify exactly.
        self.assertEqual(mismatches, [], f"accuracy={accuracy:.2f}; mismatches={mismatches}")
        self.assertEqual(accuracy, 1.0)


class ClassificationBehaviourTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tables = {table["table_name"]: table for table in _load_gold()}
        self.profiler = SemanticColumnProfiler()

    def _plan(self, table_name: str) -> ModelingPlan:
        table = self.tables[table_name]
        return self.profiler.profile(
            table["records"], table_name=table_name, record_type=table["record_type"]
        )

    def test_identity_selection_prefers_id_named_over_other_unique_columns(self) -> None:
        plan = self._plan("employees")
        # salary/hired_on/notes/full_name are all unique in this tiny table, but only
        # the id-named emp_id must become the identity.
        self.assertEqual(plan.identity_columns, ["emp_id"])
        self.assertEqual(plan.assignment("emp_id").role, ColumnRole.IDENTIFIER)
        self.assertEqual(plan.assignment("salary").role, ColumnRole.MEASURE)
        self.assertEqual(plan.assignment("full_name").role, ColumnRole.FREE_TEXT)

    def test_self_referential_foreign_key_targets_record_type(self) -> None:
        plan = self._plan("employees")
        manager = plan.assignment("manager_id")
        self.assertEqual(manager.role, ColumnRole.ENTITY_REFERENCE)
        self.assertEqual(manager.reference_type, "Employee")  # self-reference
        self.assertTrue(any(link.self_referential for link in plan.foreign_keys))

    def test_name_based_reference_derives_class_name(self) -> None:
        plan = self._plan("orders")
        customer = plan.assignment("customer_id")
        self.assertEqual(customer.role, ColumnRole.ENTITY_REFERENCE)
        self.assertEqual(customer.reference_type, "Customer")

    def test_categorical_materialisation_respects_supernode_guard(self) -> None:
        plan = self._plan("employees")
        # Department has reusable business-dimension semantics and enough values
        # to justify a shared node.
        self.assertEqual(plan.assignment("department").materialization, "node")
        # Status is a scalar enum, so it stays a property.
        self.assertEqual(plan.assignment("status").materialization, "property")

    def test_low_cardinality_alone_does_not_promote_a_column_to_a_node(self) -> None:
        records = [
            {"id": index, "bucket": value}
            for index, value in enumerate(("a", "b", "c", "a", "b", "c"), start=1)
        ]
        plan = self.profiler.profile(records, table_name="samples", record_type="Sample")
        assignment = plan.assignment("bucket")
        self.assertEqual(assignment.role, ColumnRole.CATEGORICAL_DIMENSION)
        self.assertEqual(assignment.materialization, "property")
        self.assertIn("no independent reusable dimension semantics", assignment.rationale)

    def test_temporal_detected_by_iso_date_values(self) -> None:
        plan = self._plan("employees")
        self.assertEqual(plan.assignment("hired_on").features.dtype, "date")
        self.assertEqual(plan.assignment("hired_on").role, ColumnRole.TEMPORAL)

    def test_features_are_computed(self) -> None:
        plan = self._plan("employees")
        dept = plan.assignment("department").features
        self.assertEqual(dept.distinct_count, 3)  # Engineering, Sales, Support
        self.assertEqual(dept.row_count, 8)
        self.assertFalse(dept.is_unique)
        emp_id = plan.assignment("emp_id").features
        self.assertTrue(emp_id.is_unique)
        self.assertEqual(emp_id.null_rate, 0.0)

    def test_confidence_bounds(self) -> None:
        plan = self._plan("orders")
        for assignment in plan.assignments:
            self.assertGreater(assignment.confidence, 0.0)
            self.assertLessEqual(assignment.confidence, 1.0)


class ModelingPolicyEmissionTests(unittest.TestCase):
    def test_plan_lowers_to_valid_modeling_policy(self) -> None:
        table = next(t for t in _load_gold() if t["table_name"] == "employees")
        plan = SemanticColumnProfiler().profile(
            table["records"], table_name="employees", record_type="Employee"
        )
        policy = plan.to_modeling_policy()
        self.assertIsInstance(policy, ModelingPolicy)
        self.assertEqual(policy.record_type, "Employee")
        self.assertEqual(policy.id_columns, ["emp_id"])
        # Node-role columns must carry a reference_type in the emitted policy.
        department = policy.column_policy("department")
        self.assertEqual(department.role, ColumnRole.CATEGORICAL_DIMENSION)
        self.assertEqual(department.reference_type, "Department")
        manager = policy.column_policy("manager_id")
        self.assertEqual(manager.reference_type, "Employee")
        # Property-role columns carry a datatype (salary values are whole numbers).
        self.assertEqual(policy.column_policy("salary").datatype, "integer")
        # The emitted policy derives a coherent ontology.
        ontology = policy.to_ontology_schema()
        self.assertIn("Department", {cls.name for cls in ontology.classes})

    def test_custom_thresholds_flow_through(self) -> None:
        table = next(t for t in _load_gold() if t["table_name"] == "employees")
        thresholds = PromotionThresholds(keep_as_property_max_distinct=5)
        plan = SemanticColumnProfiler(thresholds).profile(
            table["records"], table_name="employees", record_type="Employee"
        )
        # Even a reusable dimension stays a property below the configured cutoff.
        self.assertEqual(plan.assignment("department").materialization, "property")


class EdgeCaseTests(unittest.TestCase):
    def test_all_null_and_internal_columns_are_ignored(self) -> None:
        records = [
            {"id": 1, "empty_col": None, "_debug": "x", "name": "A"},
            {"id": 2, "empty_col": None, "_debug": "y", "name": "B"},
        ]
        plan = SemanticColumnProfiler().profile(records, table_name="t", record_type="Row")
        self.assertEqual(plan.assignment("empty_col").role, ColumnRole.IGNORE)
        self.assertEqual(plan.assignment("_debug").role, ColumnRole.IGNORE)
        self.assertEqual(plan.assignment("id").role, ColumnRole.IDENTIFIER)


if __name__ == "__main__":
    unittest.main()
