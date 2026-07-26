from __future__ import annotations

import unittest

from kg_construction.domain.models import (
    ConstructionBatch,
    EntityCandidate,
    EvidenceRef,
    GraphFact,
    RelationCandidate,
    SourceRecord,
)
from kg_construction.evaluation.validation_metrics import ValidationMetrics
from kg_construction.infrastructure.graph_builders.property_graph import PropertyGraphBuilder
from kg_construction.infrastructure.ontology.validation import (
    ValidationFinding,
    ValidationReport,
)
from kg_construction.infrastructure.resolution.exact import ExactKeyEntityResolver


class CanonicalSourceIdentityTests(unittest.TestCase):
    def test_text_snapshot_hash_preserves_case_and_whitespace_changes(self) -> None:
        first = SourceRecord.from_text("Ada Lovelace", source_name="person.txt")
        second = SourceRecord.from_text("ada  lovelace", source_name="person.txt")

        self.assertNotEqual(first.source_id, second.source_id)

    def test_json_source_hash_is_independent_of_dictionary_insertion_order(self) -> None:
        first = SourceRecord.from_json(
            {
                "person": {
                    "name": "Ada",
                    "identifiers": {"orcid": "0000-0001", "local": 7},
                },
                "active": True,
            },
            source_name="people.json",
        )
        second = SourceRecord.from_json(
            {
                "active": True,
                "person": {
                    "identifiers": {"local": 7, "orcid": "0000-0001"},
                    "name": "Ada",
                },
            },
            source_name="people.json",
        )

        self.assertEqual(first.source_id, second.source_id)
        self.assertEqual(first.content, second.content)

    def test_row_source_hash_is_independent_of_key_and_row_order(self) -> None:
        first = SourceRecord.from_rows(
            [
                {"id": 1, "profile": {"name": "Ada", "country": "GB"}},
                {"id": 2, "profile": {"name": "Grace", "country": "US"}},
            ],
            source_type="csv",
            source_name="people.csv",
        )
        second = SourceRecord.from_rows(
            [
                {"profile": {"country": "US", "name": "Grace"}, "id": 2},
                {"profile": {"country": "GB", "name": "Ada"}, "id": 1},
            ],
            source_type="csv",
            source_name="people.csv",
        )

        self.assertEqual(first.source_id, second.source_id)

    def test_batch_merge_keeps_one_copy_of_a_source_snapshot(self) -> None:
        source = SourceRecord.from_text("Ada works at Analytical Engines.", source_name="a.txt")

        merged = ConstructionBatch(source_records=[source]).merged(
            ConstructionBatch(source_records=[source])
        )

        self.assertEqual(merged.source_records, [source])

    def test_evidence_identity_is_independent_of_nested_location_key_order(self) -> None:
        source = SourceRecord.from_text("Ada", source_name="people.txt")
        first = EvidenceRef.build(
            source,
            location={"kind": "span", "position": {"start": 0, "end": 3}},
            text="Ada",
        )
        second = EvidenceRef.build(
            source,
            location={"position": {"end": 3, "start": 0}, "kind": "span"},
            text="Ada",
        )

        self.assertEqual(first.evidence_id, second.evidence_id)


class CanonicalPropertyGraphTests(unittest.TestCase):
    @staticmethod
    def _canonical_entities() -> tuple[EntityCandidate, EntityCandidate]:
        first_source = SourceRecord.from_rows(
            [{"employee_id": "E-1", "name": "Ada Lovelace"}],
            source_type="hr",
            source_name="hr.csv",
        )
        second_source = SourceRecord.from_json(
            {"orcid": "0000-0001", "display_name": "A. Lovelace"},
            source_name="researchers.json",
        )
        first_evidence = EvidenceRef.build(
            first_source,
            location={"kind": "row", "row_index": 0},
            text="Ada Lovelace",
            metadata={"authority": "human-resources"},
        )
        second_evidence = EvidenceRef.build(
            second_source,
            location={"kind": "jsonpath", "path": "$.display_name"},
            text="A. Lovelace",
            metadata={"authority": "orcid-export"},
        )
        first = EntityCandidate.build(
            entity_type="Person",
            name="Ada Lovelace",
            properties={
                "canonical_entity_id": "person_ada",
                "department": "Research",
                "status": "employee",
            },
            evidence=[first_evidence],
            confidence=0.98,
            extractor_name="hr_mapper",
            technique="primary_key_mapping",
            provenance_metadata={"mapping_version": "hr-v2"},
        )
        second = EntityCandidate.build(
            entity_type="Researcher",
            name="A. Lovelace",
            properties={
                "canonical_entity_id": "person_ada",
                "orcid": "0000-0001",
                "status": "active-researcher",
            },
            evidence=[second_evidence],
            confidence=0.93,
            extractor_name="orcid_mapper",
            technique="authoritative_identifier_mapping",
            provenance_metadata={"mapping_version": "orcid-v1"},
        )
        return first, second

    def test_canonical_node_merge_is_order_invariant_and_preserves_support(self) -> None:
        first, second = self._canonical_entities()
        builder = PropertyGraphBuilder()

        forward = builder.build(ConstructionBatch(entities=[first, second]))
        reverse = builder.build(ConstructionBatch(entities=[second, first]))

        self.assertEqual(forward.to_dict(), reverse.to_dict())
        canonical_node = next(node for node in forward.nodes if node.node_id == "person_ada")
        self.assertEqual(canonical_node.labels, ["Entity", "Person", "Researcher"])
        self.assertEqual(canonical_node.properties["name"], "Ada Lovelace")
        self.assertEqual(canonical_node.properties["status"], "employee")
        self.assertEqual(canonical_node.properties["department"], "Research")
        self.assertEqual(canonical_node.properties["orcid"], "0000-0001")
        self.assertEqual(
            canonical_node.properties["evidence_ids"],
            sorted([first.evidence[0].evidence_id, second.evidence[0].evidence_id]),
        )
        self.assertEqual(
            canonical_node.properties["source_names"],
            ["hr.csv", "researchers.json"],
        )
        self.assertEqual(len(canonical_node.properties["candidate_provenance"]), 2)
        self.assertEqual(
            {
                row["provenance_metadata"]["mapping_version"]
                for row in canonical_node.properties["candidate_provenance"]
            },
            {"hr-v2", "orcid-v1"},
        )
        self.assertIn("status", canonical_node.properties["property_conflicts"])

    def test_semantic_edges_union_assertions_but_keep_distinct_events(self) -> None:
        person_a, person_b = self._canonical_entities()
        organization_source = SourceRecord.from_text("Analytical Engines", source_name="org.txt")
        organization_evidence = EvidenceRef.build(
            organization_source,
            location={"kind": "span", "start": 0, "end": 18},
            text="Analytical Engines",
        )
        organization = EntityCandidate.build(
            entity_type="Organization",
            name="Analytical Engines",
            properties={"canonical_entity_id": "org_analytical_engines"},
            evidence=[organization_evidence],
            extractor_name="organization_mapper",
            technique="authoritative_identifier_mapping",
        )
        first_relation = RelationCandidate.build(
            source=person_a.candidate_id,
            relation_type="WORKS_AT",
            target=organization.candidate_id,
            evidence=person_a.evidence,
            extractor_name="hr_mapper",
            technique="foreign_key_mapping",
            provenance_metadata={"mapping_version": "hr-v2"},
        )
        second_relation = RelationCandidate.build(
            source=person_b.candidate_id,
            relation_type="WORKS_AT",
            target=organization.candidate_id,
            evidence=person_b.evidence,
            extractor_name="orcid_mapper",
            technique="relation_extraction",
            provenance_metadata={"mapping_version": "orcid-v1"},
        )
        first_purchase = RelationCandidate.build(
            source=person_a.candidate_id,
            relation_type="PURCHASED_FROM",
            target=organization.candidate_id,
            properties={"event_id": "purchase-1", "timestamp": "2026-01-02T10:00:00Z"},
            evidence=person_a.evidence,
            extractor_name="order_mapper",
            technique="event_mapping",
        )
        second_purchase = RelationCandidate.build(
            source=person_a.candidate_id,
            relation_type="PURCHASED_FROM",
            target=organization.candidate_id,
            properties={"event_id": "purchase-2", "timestamp": "2026-01-02T10:05:00Z"},
            evidence=person_a.evidence,
            extractor_name="order_mapper",
            technique="event_mapping",
        )
        self.assertNotEqual(first_purchase.candidate_id, second_purchase.candidate_id)
        batch = ConstructionBatch(
            entities=[person_a, person_b, organization],
            relations=[first_relation, second_relation, first_purchase, second_purchase],
        )

        graph = PropertyGraphBuilder().build(batch)
        works_at = [edge for edge in graph.edges if edge.type == "WORKS_AT"]
        purchases = [edge for edge in graph.edges if edge.type == "PURCHASED_FROM"]

        self.assertEqual(len(works_at), 1)
        self.assertEqual(works_at[0].source_id, "person_ada")
        self.assertEqual(works_at[0].target_id, "org_analytical_engines")
        self.assertEqual(works_at[0].properties["assertion_count"], 2)
        self.assertEqual(len(works_at[0].properties["assertion_support"]), 2)
        self.assertEqual(
            works_at[0].properties["source_names"],
            ["hr.csv", "researchers.json"],
        )
        self.assertEqual(len(purchases), 2)
        self.assertEqual(
            {edge.properties["event_id"] for edge in purchases},
            {"purchase-1", "purchase-2"},
        )
        self.assertEqual(ValidationMetrics._duplicate_edge_rate(graph), 0.0)
        self.assertTrue(
            all(
                any(node.properties.get("canonical_edge_id") == edge.edge_id for node in graph.nodes)
                for edge in graph.edges
            )
        )

    def test_resolution_lineage_does_not_split_one_semantic_edge(self) -> None:
        first_source = SourceRecord.from_text(
            "Ada Lovelace works at Analytical Engines.",
            source_name="biography-a.txt",
        )
        second_source = SourceRecord.from_text(
            "Ada Lovelace works at Analytical Engines.",
            source_name="biography-b.txt",
        )
        first_evidence = EvidenceRef.build(first_source, text=first_source.content)
        second_evidence = EvidenceRef.build(second_source, text=second_source.content)

        def entity(
            entity_type: str,
            name: str,
            evidence: EvidenceRef,
            extractor_name: str,
        ) -> EntityCandidate:
            return EntityCandidate.build(
                entity_type=entity_type,
                name=name,
                evidence=[evidence],
                extractor_name=extractor_name,
                technique="relation_extraction",
            )

        first_person = entity("Person", "Ada Lovelace", first_evidence, "extractor_a")
        second_person = entity("Person", "Ada Lovelace", second_evidence, "extractor_b")
        first_org = entity("Organization", "Analytical Engines", first_evidence, "extractor_a")
        second_org = entity("Organization", "Analytical Engines", second_evidence, "extractor_b")
        first_relation = RelationCandidate.build(
            source=first_person.candidate_id,
            relation_type="WORKS_AT",
            target=first_org.candidate_id,
            evidence=[first_evidence],
            extractor_name="extractor_a",
            technique="relation_extraction",
        )
        second_relation = RelationCandidate.build(
            source=second_person.candidate_id,
            relation_type="WORKS_AT",
            target=second_org.candidate_id,
            evidence=[second_evidence],
            extractor_name="extractor_b",
            technique="relation_extraction",
        )
        resolved = ExactKeyEntityResolver(allow_cross_source_name_fallback=True).resolve(
            ConstructionBatch(
                entities=[first_person, second_person, first_org, second_org],
                relations=[first_relation, second_relation],
            )
        )

        self.assertEqual(
            {relation.properties["original_source"] for relation in resolved.relations},
            {first_person.candidate_id, second_person.candidate_id},
        )
        graph = PropertyGraphBuilder().build(resolved)
        works_at = [edge for edge in graph.edges if edge.type == "WORKS_AT"]
        self.assertEqual(len(works_at), 1)
        self.assertEqual(works_at[0].properties["assertion_count"], 2)

    def test_graph_facts_consolidate_claims_and_materialize_valid_endpoints(self) -> None:
        first_source = SourceRecord.from_text("Ada knows Grace.", source_name="first.txt")
        second_source = SourceRecord.from_text("Ada knows Grace.", source_name="second.txt")
        first_evidence = EvidenceRef.build(first_source, text=first_source.content)
        second_evidence = EvidenceRef.build(second_source, text=second_source.content)
        facts = [
            GraphFact.build(
                subject="urn:person:ada",
                predicate="urn:schema:knows",
                object="urn:person:grace",
                evidence=[evidence],
                extractor_name="rdf_mapper",
                technique="standards_mapping",
                properties={
                    "rdf_object_type": "iri",
                    **(
                        {"subject_map": {"template": "urn:person:{id}"}}
                        if index == 0
                        else {"csvw_column": "related_person"}
                    ),
                },
            )
            for index, evidence in enumerate((first_evidence, second_evidence))
        ]
        literal_fact = GraphFact.build(
            subject="urn:person:ada",
            predicate="urn:schema:displayName",
            object="Ada Lovelace",
            evidence=[first_evidence],
            extractor_name="rdf_mapper",
            technique="standards_mapping",
            properties={
                "rdf_object_type": "literal",
                "rdf_datatype": "http://www.w3.org/2001/XMLSchema#string",
            },
        )

        graph = PropertyGraphBuilder().build(
            ConstructionBatch(facts=[*facts, literal_fact])
        )
        knows_edges = [edge for edge in graph.edges if edge.type == "urn:schema:knows"]

        self.assertEqual(len(knows_edges), 1)
        self.assertEqual(knows_edges[0].properties["assertion_count"], 2)
        self.assertEqual(len(knows_edges[0].properties["evidence_ids"]), 2)
        self.assertEqual(
            len([node for node in graph.nodes if "Fact" in node.labels]),
            3,
        )
        node_ids = {node.node_id for node in graph.nodes}
        self.assertTrue(
            all(
                edge.source_id in node_ids and edge.target_id in node_ids
                for edge in graph.edges
            )
        )
        self.assertEqual(ValidationMetrics._duplicate_edge_rate(graph), 0.0)

    def test_validation_failure_rate_counts_failed_objects_not_findings(self) -> None:
        entity, _ = self._canonical_entities()
        report = ValidationReport(
            findings=[
                ValidationFinding(
                    object_id=entity.candidate_id,
                    object_type="entity",
                    rule="required_property",
                    message="missing A",
                    policy="reject",
                ),
                ValidationFinding(
                    object_id=entity.candidate_id,
                    object_type="entity",
                    rule="allowed_label",
                    message="bad B",
                    policy="reject",
                ),
            ],
            checked_count=2,
        )

        metrics = ValidationMetrics().evaluate(
            ConstructionBatch(entities=[entity]),
            report,
        )

        self.assertEqual(metrics["validation_failure_rate"], 1.0)
        self.assertEqual(metrics["validation_finding_density"], 2.0)


if __name__ == "__main__":
    unittest.main()
