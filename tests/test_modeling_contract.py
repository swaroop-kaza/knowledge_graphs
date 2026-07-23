"""The modelling contract: roles, reference nodes, ontology, and policy."""

from __future__ import annotations

import unittest

from pydantic import ValidationError

from kg_construction.application.modeling_policy import (
    CanonicalizationRule,
    ColumnPolicy,
    ModelingPolicy,
    PromotionThresholds,
    RelationshipDeclaration,
    default_modeling_policy,
)
from kg_construction.domain.modeling import (
    Cardinality,
    ColumnRole,
    ControlledVocabulary,
    OntologyClass,
    OntologyPredicate,
    OntologySchema,
    ReferenceNode,
)


class ColumnRoleTests(unittest.TestCase):
    def test_node_and_property_partitions(self) -> None:
        self.assertTrue(ColumnRole.CATEGORICAL_DIMENSION.becomes_node)
        self.assertTrue(ColumnRole.ENTITY_REFERENCE.becomes_node)
        self.assertFalse(ColumnRole.MEASURE.becomes_node)
        self.assertTrue(ColumnRole.MEASURE.becomes_property)
        self.assertTrue(ColumnRole.TEMPORAL.becomes_property)
        self.assertTrue(ColumnRole.FREE_TEXT.becomes_property)
        self.assertFalse(ColumnRole.IDENTIFIER.becomes_node)
        self.assertFalse(ColumnRole.IDENTIFIER.becomes_property)
        self.assertFalse(ColumnRole.IGNORE.becomes_node)


class CardinalityTests(unittest.TestCase):
    def test_labels_and_flags(self) -> None:
        self.assertEqual(Cardinality.exactly_one().label, "exactly_one")
        self.assertEqual(Cardinality.zero_or_one().label, "zero_or_one")
        self.assertEqual(Cardinality.one_or_many().label, "one_or_many")
        self.assertEqual(Cardinality.zero_or_many().label, "zero_or_many")
        self.assertEqual(Cardinality(min_count=2, max_count=5).label, "2..5")
        self.assertEqual(Cardinality(min_count=2, max_count=None).label, "2..*")
        self.assertTrue(Cardinality.exactly_one().required)
        self.assertTrue(Cardinality.exactly_one().functional)
        self.assertFalse(Cardinality.zero_or_many().functional)
        self.assertFalse(Cardinality.zero_or_one().required)

    def test_invalid_bounds_raise(self) -> None:
        with self.assertRaises(ValidationError):
            Cardinality(min_count=3, max_count=1)
        with self.assertRaises(ValidationError):
            Cardinality(min_count=-1)


class ControlledVocabularyTests(unittest.TestCase):
    def test_closed_vocabulary_rejects_unknown(self) -> None:
        vocab = ControlledVocabulary(
            name="Status",
            permissible_values=["open", "closed"],
            aliases={"OPEN": "open"},
            closed=True,
        )
        self.assertEqual(vocab.canonical_value("OPEN"), "open")
        self.assertTrue(vocab.is_permitted("open"))
        self.assertIsNone(vocab.canonical_value("unknown"))
        self.assertFalse(vocab.is_permitted("unknown"))

    def test_open_vocabulary_admits_new_values(self) -> None:
        vocab = ControlledVocabulary(name="Tag", permissible_values=["a"], closed=False)
        self.assertEqual(vocab.canonical_value("brand-new"), "brand-new")
        self.assertTrue(vocab.is_permitted("brand-new"))

    def test_empty_name_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            ControlledVocabulary(name="  ")


class ReferenceNodeTests(unittest.TestCase):
    def test_surface_variants_collapse_to_one_node(self) -> None:
        a = ReferenceNode.build(reference_type="Department", value="Engineering")
        b = ReferenceNode.build(reference_type="Department", value="  ENGINEERING ")
        c = ReferenceNode.build(reference_type="Department", value="Sales")
        self.assertEqual(a.node_id, b.node_id)
        self.assertNotEqual(a.node_id, c.node_id)

    def test_identity_is_type_scoped(self) -> None:
        dept = ReferenceNode.build(reference_type="Department", value="X")
        team = ReferenceNode.build(reference_type="Team", value="X")
        self.assertNotEqual(dept.node_id, team.node_id)

    def test_labels_and_raw_values(self) -> None:
        node = ReferenceNode.build(reference_type="Department", value="Engineering", raw_value="ENGINEERING")
        self.assertEqual(node.labels, ["Reference", "Department"])
        self.assertEqual(node.raw_values, ["ENGINEERING"])
        self.assertEqual(ReferenceNode(reference_type="Reference", value="x").labels, ["Reference"])

    def test_normalize_false_preserves_value(self) -> None:
        node = ReferenceNode.build(reference_type="T", value="Mixed Case", normalize=False)
        self.assertEqual(node.value, "Mixed Case")

    def test_node_id_is_deterministic(self) -> None:
        first = ReferenceNode.build(reference_type="T", value="v").node_id
        second = ReferenceNode.build(reference_type="T", value="v").node_id
        self.assertEqual(first, second)
        self.assertTrue(first.startswith("ref_"))


class OntologyModelTests(unittest.TestCase):
    def test_class_role_validation(self) -> None:
        OntologyClass(name="Employee", role="record")
        OntologyClass(name="Department", role="reference")
        with self.assertRaises(ValidationError):
            OntologyClass(name="Bad", role="not_a_role")

    def test_predicate_kind_and_datatype_validation(self) -> None:
        OntologyPredicate(name="in_department", domain="Employee", range="Department", kind="object")
        OntologyPredicate(name="salary", domain="Employee", range="number", kind="datatype")
        with self.assertRaises(ValidationError):
            OntologyPredicate(name="p", domain="A", range="B", kind="weird")
        with self.assertRaises(ValidationError):
            OntologyPredicate(name="p", domain="A", range="NotAType", kind="datatype")

    def test_unresolved_references_detected(self) -> None:
        schema = OntologySchema(
            classes=[OntologyClass(name="Employee", role="record")],
            predicates=[
                OntologyPredicate(name="in_department", domain="Employee", range="Department", kind="object"),
            ],
        )
        issues = schema.unresolved_predicate_references()
        self.assertEqual(len(issues), 1)
        self.assertIn("Department", issues[0])

    def test_bridge_to_graph_schema(self) -> None:
        schema = OntologySchema(
            schema_id="employees",
            classes=[
                OntologyClass(name="Employee", role="record", key_properties=["emp_id"]),
                OntologyClass(name="Department", role="reference"),
            ],
            predicates=[
                OntologyPredicate(
                    name="in_department", domain="Employee", range="Department", kind="object",
                    cardinality=Cardinality.exactly_one(),
                ),
                OntologyPredicate(
                    name="salary", domain="Employee", range="number", kind="datatype",
                    cardinality=Cardinality.exactly_one(),
                ),
            ],
        )
        graph_schema = schema.to_graph_schema()
        node_names = {node.name for node in graph_schema.node_types}
        self.assertEqual(node_names, {"Employee", "Department"})
        rel_names = {rel.name for rel in graph_schema.relationship_types}
        self.assertEqual(rel_names, {"in_department"})
        employee = next(node for node in graph_schema.node_types if node.name == "Employee")
        salary = next(prop for prop in employee.properties if prop.name == "salary")
        self.assertTrue(salary.required)  # from cardinality min_count >= 1
        self.assertEqual(salary.value_type, "number")
        # The existing SHACL emitter still works on the derived schema.
        self.assertIn("sh:NodeShape", graph_schema.to_shacl_turtle())
        self.assertEqual(graph_schema.metadata["reference_classes"], ["Department"])


class CanonicalizationRuleTests(unittest.TestCase):
    def test_default_normalisation(self) -> None:
        rule = CanonicalizationRule()
        self.assertEqual(rule.apply("  Engineering   Team  "), "Engineering Team")

    def test_casefold_and_aliases(self) -> None:
        rule = CanonicalizationRule(casefold=True, aliases={"eng": "engineering"})
        self.assertEqual(rule.apply("  ENG "), "engineering")

    def test_lowercase_and_strip_characters(self) -> None:
        rule = CanonicalizationRule(lowercase=True, strip_characters=".")
        self.assertEqual(rule.apply("...Sales..."), "sales")


class PromotionThresholdsTests(unittest.TestCase):
    def test_defaults(self) -> None:
        thresholds = PromotionThresholds()
        self.assertTrue(thresholds.promote_categoricals_to_nodes)
        self.assertEqual(thresholds.categorical_max_distinct_ratio, 0.5)
        self.assertEqual(thresholds.keep_as_property_max_distinct, 2)
        self.assertEqual(thresholds.supernode_policy, "warn")

    def test_validation(self) -> None:
        with self.assertRaises(ValidationError):
            PromotionThresholds(categorical_max_distinct_ratio=1.5)
        with self.assertRaises(ValidationError):
            PromotionThresholds(supernode_policy="explode")


class ColumnPolicyTests(unittest.TestCase):
    def test_node_role_requires_reference_type(self) -> None:
        with self.assertRaises(ValidationError):
            ColumnPolicy(column="department", role=ColumnRole.CATEGORICAL_DIMENSION)

    def test_property_role_forbids_reference_type(self) -> None:
        with self.assertRaises(ValidationError):
            ColumnPolicy(column="salary", role=ColumnRole.MEASURE, reference_type="X")

    def test_relationship_and_property_defaults(self) -> None:
        node_policy = ColumnPolicy(
            column="department", role=ColumnRole.CATEGORICAL_DIMENSION, reference_type="Department"
        )
        self.assertEqual(node_policy.target_relationship(), "has_department")
        explicit = ColumnPolicy(
            column="manager_id", role=ColumnRole.ENTITY_REFERENCE,
            reference_type="Employee", relationship_type="REPORTS_TO",
        )
        self.assertEqual(explicit.target_relationship(), "REPORTS_TO")
        measure = ColumnPolicy(column="salary", role=ColumnRole.MEASURE)
        self.assertEqual(measure.target_property(), "salary")
        self.assertEqual(
            ColumnPolicy(column="c", role=ColumnRole.MEASURE, property_name="amount").target_property(),
            "amount",
        )


class ModelingPolicyTests(unittest.TestCase):
    def _employee_policy(self) -> ModelingPolicy:
        return ModelingPolicy(
            table_name="employees",
            record_type="Employee",
            id_columns=["emp_id"],
            columns=[
                ColumnPolicy(column="emp_id", role=ColumnRole.IDENTIFIER),
                ColumnPolicy(
                    column="department", role=ColumnRole.CATEGORICAL_DIMENSION,
                    reference_type="Department", relationship_type="IN_DEPARTMENT",
                    cardinality=Cardinality.exactly_one(),
                ),
                ColumnPolicy(
                    column="manager_id", role=ColumnRole.ENTITY_REFERENCE,
                    reference_type="Employee", relationship_type="REPORTS_TO",
                ),
                ColumnPolicy(column="salary", role=ColumnRole.MEASURE),
                ColumnPolicy(column="hired_on", role=ColumnRole.TEMPORAL),
                ColumnPolicy(column="notes", role=ColumnRole.FREE_TEXT),
                ColumnPolicy(column="_internal", role=ColumnRole.IGNORE),
            ],
            relationships=[
                RelationshipDeclaration(
                    name="COLLABORATES_WITH", source_type="Employee", target_type="Employee",
                    cardinality=Cardinality.zero_or_many(),
                ),
            ],
            controlled_vocabularies=[
                ControlledVocabulary(name="Department", permissible_values=["Engineering", "Sales"], closed=False),
            ],
        )

    def test_lookups_and_reference_types(self) -> None:
        policy = self._employee_policy()
        self.assertEqual(policy.column_policy("department").role, ColumnRole.CATEGORICAL_DIMENSION)
        self.assertIsNone(policy.column_policy("missing"))
        self.assertEqual(policy.reference_types(), {"Department", "Employee"})
        self.assertEqual(policy.vocabulary("Department").permissible_values, ["Engineering", "Sales"])
        self.assertIsNone(policy.vocabulary("nope"))

    def test_canonicalization_for_falls_back_to_default(self) -> None:
        policy = self._employee_policy()
        self.assertIs(policy.canonicalization_for("department"), policy.default_canonicalization)
        custom = CanonicalizationRule(casefold=True)
        policy2 = ModelingPolicy(
            columns=[ColumnPolicy(column="dept", role=ColumnRole.CATEGORICAL_DIMENSION,
                                  reference_type="Department", canonicalization=custom)],
        )
        self.assertIs(policy2.canonicalization_for("dept"), custom)

    def test_duplicate_columns_and_vocabs_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            ModelingPolicy(columns=[
                ColumnPolicy(column="c", role=ColumnRole.MEASURE),
                ColumnPolicy(column="c", role=ColumnRole.TEMPORAL),
            ])
        with self.assertRaises(ValidationError):
            ModelingPolicy(controlled_vocabularies=[
                ControlledVocabulary(name="V"),
                ControlledVocabulary(name="V"),
            ])

    def test_to_ontology_schema(self) -> None:
        onto = self._employee_policy().to_ontology_schema()
        class_roles = {cls.name: cls.role for cls in onto.classes}
        self.assertEqual(class_roles["Employee"], "record")
        self.assertEqual(class_roles["Department"], "reference")
        predicate_names = {pred.name for pred in onto.predicates}
        # object predicates for the categorical + reference + declared relationship
        self.assertIn("IN_DEPARTMENT", predicate_names)
        self.assertIn("REPORTS_TO", predicate_names)
        self.assertIn("COLLABORATES_WITH", predicate_names)
        # datatype predicates for measure/temporal/free_text (ignore excluded)
        self.assertIn("salary", predicate_names)
        self.assertIn("hired_on", predicate_names)
        self.assertIn("notes", predicate_names)
        self.assertNotIn("_internal", predicate_names)
        in_dept = next(p for p in onto.predicates if p.name == "IN_DEPARTMENT")
        self.assertEqual(in_dept.cardinality.label, "exactly_one")
        self.assertEqual(onto.unresolved_predicate_references(), [])

    def test_default_policy_and_round_trip(self) -> None:
        policy = default_modeling_policy(table_name="t", record_type="Row")
        self.assertEqual(policy.record_type, "Row")
        self.assertEqual(policy.columns, [])
        rebuilt = ModelingPolicy.model_validate(self._employee_policy().model_dump(mode="json"))
        self.assertEqual(rebuilt.record_type, "Employee")

    def test_extra_fields_forbidden(self) -> None:
        with self.assertRaises(ValidationError):
            ModelingPolicy(unexpected=True)


if __name__ == "__main__":
    unittest.main()
