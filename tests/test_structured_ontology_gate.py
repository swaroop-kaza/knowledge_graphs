"""Ontology formation and the SHACL conformance gate for structured graphs."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from kg_construction.domain.graph_objects import RDFGraph, RDFTriple
from kg_construction.domain.modeling import (
    Cardinality,
    ColumnRole,
    OntologyClass,
    OntologyPredicate,
    OntologySchema,
)
from kg_construction.infrastructure.ontology.structured_ontology import (
    StructuredOntologyProposal,
    StructuredReleaseGate,
    generate_shacl_shapes,
    materialize_structured_rdf,
    propose_structured_ontology,
)
from kg_construction.structured.modeling_policy import ColumnPolicy, ModelingPolicy
from kg_construction.structured.semantic_profiler import SemanticColumnProfiler

_FIXTURE = Path(__file__).parent / "fixtures" / "gold_standards" / "column_roles.json"
_XSD_STRING = "http://www.w3.org/2001/XMLSchema#string"


def _employees() -> list[dict]:
    tables = json.loads(_FIXTURE.read_text(encoding="utf-8"))["tables"]
    return next(t for t in tables if t["table_name"] == "employees")["records"]


def _employee_policy() -> ModelingPolicy:
    plan = SemanticColumnProfiler().profile(_employees(), table_name="employees", record_type="Employee")
    return plan.to_modeling_policy()


class ReleaseGateAcceptanceTests(unittest.TestCase):
    def test_release_graph_has_zero_shacl_violations(self) -> None:
        policy = _employee_policy()
        proposal = propose_structured_ontology(policy)
        graph = materialize_structured_rdf(policy, _employees())
        report = StructuredReleaseGate().validate(graph, proposal=proposal)
        self.assertTrue(report.conforms)
        self.assertEqual(report.violation_count, 0)
        self.assertTrue(report.release_ok)

    def test_categorical_values_deduplicate_into_shared_nodes(self) -> None:
        # Pillar 1.1 at the graph level: 8 employees, 3 distinct departments ->
        # exactly 3 Department nodes, not one literal per row.
        policy = _employee_policy()
        graph = materialize_structured_rdf(policy, _employees())
        department_nodes = {
            triple.subject
            for triple in graph.triples
            if triple.object_type == "iri" and triple.object.endswith("Department")
        }
        self.assertEqual(len(department_nodes), 3)


class GateHasTeethTests(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = _employee_policy()
        self.proposal = propose_structured_ontology(self.policy)
        self.graph = materialize_structured_rdf(self.policy, _employees())
        self.gate = StructuredReleaseGate()

    def test_categorical_stored_as_literal_is_a_violation(self) -> None:
        # Storing a dimension as a literal instead of an edge to a shared node is
        # exactly the anti-pattern the ontology forbids (sh:nodeKind sh:IRI + sh:class).
        dept_edge = next(
            t for t in self.graph.triples
            if t.object_type == "iri" and "has_department" in t.predicate
        )
        broken = RDFGraph(
            triples=[
                *self.graph.triples,
                RDFTriple(
                    subject=dept_edge.subject,
                    predicate=dept_edge.predicate,
                    object="Engineering",
                    object_type="literal",
                    datatype=_XSD_STRING,
                ),
            ]
        )
        report = self.gate.validate(broken, proposal=self.proposal)
        self.assertGreater(report.violation_count, 0)
        self.assertFalse(report.release_ok)

    def test_missing_required_property_is_a_violation(self) -> None:
        policy = ModelingPolicy(
            table_name="t",
            record_type="Row",
            id_columns=["id"],
            columns=[
                ColumnPolicy(column="id", role=ColumnRole.IDENTIFIER),
                ColumnPolicy(column="name", role=ColumnRole.FREE_TEXT, cardinality=Cardinality.exactly_one()),
            ],
        )
        proposal = propose_structured_ontology(policy)
        self.assertIn("sh:minCount 1", proposal.shacl_turtle)
        graph = materialize_structured_rdf(policy, [{"id": 1}])  # name missing
        report = self.gate.validate(graph, proposal=proposal)
        self.assertGreater(report.violation_count, 0)


class ShaclShapeGenerationTests(unittest.TestCase):
    def test_shapes_encode_class_datatype_and_object_constraints(self) -> None:
        proposal = propose_structured_ontology(_employee_policy())
        shapes = proposal.shacl_turtle
        self.assertIn("a sh:NodeShape", shapes)
        self.assertIn("sh:targetClass kg:Employee", shapes)
        self.assertIn("sh:class kg:Department", shapes)
        self.assertIn("sh:nodeKind sh:IRI", shapes)
        self.assertIn("sh:datatype xsd:integer", shapes)  # salary
        self.assertIn("sh:maxCount 1", shapes)
        # status singularises correctly (no "Statu"); it is a reference class.
        class_names = {cls.name for cls in proposal.ontology.classes}
        self.assertIn("Status", class_names)
        self.assertNotIn("Statu", class_names)

    def test_required_and_multivalued_cardinality(self) -> None:
        ontology = OntologySchema(
            classes=[
                OntologyClass(name="Doc", role="record"),
                OntologyClass(name="Tag", role="reference"),
            ],
            predicates=[
                OntologyPredicate(name="title", domain="Doc", range="string", kind="datatype", cardinality=Cardinality.exactly_one()),
                OntologyPredicate(name="tagged", domain="Doc", range="Tag", kind="object", cardinality=Cardinality.one_or_many()),
            ],
        )
        shapes = generate_shacl_shapes(ontology)
        self.assertIn("sh:minCount 1", shapes)  # required title
        self.assertIn("sh:maxCount 1", shapes)  # functional title
        self.assertIn("sh:class kg:Tag", shapes)  # object range


class ProposalLifecycleTests(unittest.TestCase):
    def test_proposed_to_promoted(self) -> None:
        proposal = propose_structured_ontology(_employee_policy())
        self.assertEqual(proposal.status, "proposed")
        self.assertEqual(proposal.quarantined_terms, [])
        self.assertTrue(proposal.promotable)
        approved = proposal.approve()
        self.assertEqual(approved.status, "approved")
        self.assertEqual(approved.graph_schema.status, "approved")
        promoted = approved.promote()
        self.assertEqual(promoted.status, "promoted")

    def test_quarantined_terms_block_promotion(self) -> None:
        proposal = StructuredOntologyProposal(
            ontology=OntologySchema(),
            graph_schema=propose_structured_ontology(_employee_policy()).graph_schema,
            shacl_turtle="",
            quarantined_terms=["orphan_predicate: range class 'Missing' is not declared"],
        )
        self.assertFalse(proposal.promotable)
        with self.assertRaises(ValueError):
            proposal.promote()


if __name__ == "__main__":
    unittest.main()
