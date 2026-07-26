"""The resolved modelling contract and its six agreeing schema projections.

Covers the Stage-1 exit gate: role and materialization are separate decisions,
and the property-graph schema, JSON-LD context, OWL 2 RL ontology, SHACL shapes,
Neo4j constraints, and evaluation expectations all agree on class, property,
relationship, identity, datatype, and cardinality.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from kg_construction.structured.plan_resolution import resolve_modeling_plan
from kg_construction.structured.resolved_plan import (
    Cardinality,
    ClassKind,
    Materialization,
    OnMissingTarget,
    ResolvedClass,
    ResolvedModelingPlan,
    ResolvedProperty,
)
from kg_construction.structured.schema_projections import (
    assert_projections_agree,
    project_all,
    projection_disagreements,
)
from kg_construction.structured.semantic_profiler import SemanticColumnProfiler

_FIXTURE = Path(__file__).parent / "fixtures" / "gold_standards" / "column_roles.json"


def _resolve(table_name: str) -> ResolvedModelingPlan:
    data = json.loads(_FIXTURE.read_text(encoding="utf-8"))
    table = next(t for t in data["tables"] if t["table_name"] == table_name)
    plan = SemanticColumnProfiler().profile(
        table["records"], table_name=table_name, record_type=table["record_type"]
    )
    return resolve_modeling_plan(plan, source_name=f"{table_name}.csv")


def _composite_plan() -> ResolvedModelingPlan:
    return ResolvedModelingPlan(
        source_name="warehouse",
        table_name="line_items",
        record_class="LineItem",
        classes=[
            ResolvedClass(name="LineItem", kind=ClassKind.ENTITY, identity_keys=["order_id", "sku"])
        ],
        properties=[
            ResolvedProperty(
                class_name="LineItem", name="order_id", datatype="string",
                nullable=False, cardinality=Cardinality.exactly_one(), source_column="order_id",
            ),
            ResolvedProperty(
                class_name="LineItem", name="sku", datatype="string",
                nullable=False, cardinality=Cardinality.exactly_one(), source_column="sku",
            ),
            ResolvedProperty(class_name="LineItem", name="qty", datatype="integer"),
        ],
    )


class RoleMaterializationSeparationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.plan = _resolve("employees")

    def test_property_categorical_is_a_literal_not_an_object_property(self) -> None:
        status = next(a for a in self.plan.attributes if a.column == "status")
        self.assertEqual(status.materialization, Materialization.PROPERTY)
        # status stays a datatype property; no Status class, no has_status relationship.
        self.assertNotIn("Status", {c.name for c in self.plan.classes})
        self.assertIn(("Employee", "status", "string"), self.plan.definitions().properties)
        self.assertNotIn("has_status", {r.name for r in self.plan.relationships})

    def test_node_categorical_is_a_governed_concept_plus_object_property(self) -> None:
        department = self.plan.clazz("Department")
        self.assertEqual(department.kind, ClassKind.CONCEPT)
        self.assertTrue(department.concept_scheme)
        names = {(r.name, r.domain, r.range) for r in self.plan.relationships}
        self.assertIn(("has_department", "Employee", "Department"), names)

    def test_identity_key_is_resolved(self) -> None:
        employee = self.plan.clazz("Employee")
        self.assertEqual(employee.identity_keys, ["emp_id"])
        self.assertFalse(employee.composite)


class ForeignKeyResolutionTests(unittest.TestCase):
    def test_self_referential_foreign_key_targets_record_class(self) -> None:
        plan = _resolve("employees")
        manager = next(fk for fk in plan.foreign_keys if fk.relationship_name == "manager")
        self.assertEqual(manager.target_class, "Employee")
        self.assertEqual(manager.target_key, ["emp_id"])
        self.assertEqual(manager.on_missing, OnMissingTarget.SKIP)

    def test_external_foreign_key_never_declares_placeholder_creation(self) -> None:
        plan = _resolve("orders")
        customer = next(fk for fk in plan.foreign_keys if fk.target_class == "Customer")
        # No Customer source is present; the contract skips missing targets.
        self.assertEqual(customer.on_missing, OnMissingTarget.SKIP)
        self.assertEqual(customer.target_key, [])
        self.assertEqual(plan.clazz("Customer").kind, ClassKind.ENTITY)


class ProjectionAgreementTests(unittest.TestCase):
    """The exit gate: every projection agrees with the plan."""

    def test_employees_projections_agree(self) -> None:
        self.assertEqual(projection_disagreements(_resolve("employees")), [])

    def test_orders_projections_agree(self) -> None:
        self.assertEqual(projection_disagreements(_resolve("orders")), [])

    def test_composite_key_projections_agree(self) -> None:
        assert_projections_agree(_composite_plan())


class ProjectionContentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.plan = _resolve("employees")
        self.bundle = project_all(self.plan)

    def test_owl2rl_declares_classes_and_typed_properties(self) -> None:
        owl = self.bundle.owl2rl_turtle
        self.assertIn("OWL 2 RL profile", owl)
        self.assertIn("kg:Employee a owl:Class .", owl)
        self.assertIn("owl:DatatypeProperty", owl)
        self.assertIn("owl:ObjectProperty", owl)
        self.assertIn("owl:FunctionalProperty", owl)  # emp_id is exactly-one

    def test_shacl_declares_node_and_property_shapes(self) -> None:
        shacl = self.bundle.shacl_turtle
        self.assertIn("sh:NodeShape", shacl)
        self.assertIn("sh:targetClass kg:Employee", shacl)
        self.assertIn("sh:datatype", shacl)
        self.assertIn("sh:class kg:Department", shacl)

    def test_jsonld_context_types_properties_and_relationships(self) -> None:
        context = self.bundle.jsonld_context["@context"]
        assert isinstance(context, dict)
        self.assertEqual(context["@version"], 1.1)
        self.assertEqual(context["salary"]["@type"], "http://www.w3.org/2001/XMLSchema#integer")
        self.assertEqual(context["has_department"]["@type"], "@id")

    def test_neo4j_emits_uniqueness_constraint(self) -> None:
        constraints = self.bundle.neo4j_constraints
        self.assertTrue(
            any("FOR (n:Employee) REQUIRE n.emp_id IS UNIQUE" in c for c in constraints)
        )

    def test_composite_key_emits_node_key(self) -> None:
        constraints = _composite_plan().to_neo4j_constraints()
        self.assertTrue(
            any("REQUIRE (n.order_id, n.sku) IS NODE KEY" in c for c in constraints)
        )

    def test_evaluation_expectations_reflect_identity_and_datatypes(self) -> None:
        expectations = self.bundle.evaluation_expectations
        self.assertEqual(expectations.identity["Employee"], ["emp_id"])
        self.assertEqual(expectations.property_datatypes["Employee.salary"], "integer")
        self.assertIn(
            {"name": "has_department", "domain": "Employee", "range": "Department", "cardinality": "exactly_one"},
            expectations.relationships,
        )


class ContractValidationTests(unittest.TestCase):
    def test_record_class_must_be_declared(self) -> None:
        with self.assertRaises(ValueError):
            ResolvedModelingPlan(source_name="s", table_name="t", record_class="Missing")

    def test_concept_class_requires_a_scheme(self) -> None:
        with self.assertRaises(ValueError):
            ResolvedClass(name="Priority", kind=ClassKind.CONCEPT)


if __name__ == "__main__":
    unittest.main()
