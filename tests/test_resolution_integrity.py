from __future__ import annotations

import unittest

from kg_construction.construction.workflow import DraftConstructionWorkflow
from kg_construction.domain.models import (
    ConstructionBatch,
    EntityCandidate,
    EvidenceRef,
    GraphFact,
    RelationCandidate,
    SourceRecord,
    stable_id,
)
from kg_construction.domain.schema import (
    GraphSchema,
    RelationshipTypeDefinition,
    SchemaRegistry,
)
from kg_construction.infrastructure.ontology.validation import (
    ReferentialIntegrityValidator,
    SchemaRegistryValidator,
)
from kg_construction.infrastructure.resolution import AliasTableResolver, NormalizedStringResolver
from kg_construction.infrastructure.resolution.exact import ExactKeyEntityResolver


def _evidence(source: SourceRecord, text: str) -> EvidenceRef:
    return EvidenceRef.build(
        source,
        location={"kind": "test", "value": text},
        text=text,
    )


def _entity(
    source: SourceRecord,
    name: str,
    *,
    identity_key: str | None = None,
    entity_type: str = "System",
) -> EntityCandidate:
    properties = {"identity_key": identity_key} if identity_key is not None else {}
    return EntityCandidate.build(
        entity_type=entity_type,
        name=name,
        properties=properties,
        evidence=[_evidence(source, name)],
        extractor_name="unit_test",
        technique="manual",
    )


class _StaticMapper:
    name = "static_mapper"

    def map_source(self, source: SourceRecord) -> ConstructionBatch:
        left = _entity(source, "Left")
        right = _entity(source, "Right")
        evidence = [_evidence(source, "Left depends on Right")]
        return ConstructionBatch(
            entities=[left, right],
            relations=[
                RelationCandidate.build(
                    source=left.candidate_id,
                    relation_type="DEPENDS_ON",
                    target=right.candidate_id,
                    evidence=evidence,
                    extractor_name=self.name,
                    technique="manual",
                )
            ],
            facts=[
                GraphFact.build(
                    subject=left.candidate_id,
                    predicate="DEPENDS_ON",
                    object=right.candidate_id,
                    evidence=evidence,
                    extractor_name=self.name,
                    technique="manual",
                )
            ],
        )


class _EndpointBlindResolver:
    """Simulate a third-party resolver that annotates entities but does not rewrite edges."""

    name = "endpoint_blind_resolver"

    def resolve(self, batch: ConstructionBatch) -> ConstructionBatch:
        entities = []
        for entity in batch.entities:
            canonical_id = stable_id("canonical_entity", entity.entity_type, entity.name.casefold())
            properties = {**entity.properties, "canonical_entity_id": canonical_id}
            entities.append(entity.model_copy(update={"properties": properties}))
        return batch.model_copy(update={"entities": entities})


class ResolutionIntegrityTests(unittest.TestCase):
    def test_sequential_resolvers_rewrite_relations_and_facts_to_final_ids(self) -> None:
        source = SourceRecord.from_text(
            "Recommendation API Rec API Platform Team",
            source_name="memory://sequential-resolution",
        )
        service = _entity(source, "Recommendation API")
        alias = _entity(source, "Rec API")
        team = _entity(source, "Platform Team")
        evidence = [_evidence(source, "Rec API owned by Platform Team")]
        relation = RelationCandidate.build(
            source=alias.candidate_id,
            relation_type="OWNED_BY",
            target=team.candidate_id,
            evidence=evidence,
            extractor_name="unit_test",
            technique="manual",
        )
        fact = GraphFact.build(
            subject=alias.candidate_id,
            predicate="OWNED_BY",
            object=team.candidate_id,
            evidence=evidence,
            extractor_name="unit_test",
            technique="manual",
        )
        batch = ConstructionBatch(
            source_records=[source],
            entities=[service, alias, team],
            relations=[relation],
            facts=[fact],
        )

        first_pass = NormalizedStringResolver().resolve(batch)
        obsolete_alias_id = first_pass.relations[0].source
        resolved = AliasTableResolver(
            aliases={"Recommendation Service": ["Recommendation API", "Rec API"]}
        ).resolve(first_pass)

        current_ids = {
            str(entity.properties["canonical_entity_id"])
            for entity in resolved.entities
        }
        self.assertIn(resolved.relations[0].source, current_ids)
        self.assertIn(resolved.relations[0].target, current_ids)
        self.assertIn(resolved.facts[0].subject, current_ids)
        self.assertIn(resolved.facts[0].object, current_ids)
        self.assertNotEqual(resolved.relations[0].source, obsolete_alias_id)
        self.assertEqual(
            resolved.metadata["entity_resolution_aliases"][obsolete_alias_id],
            resolved.relations[0].source,
        )
        self.assertTrue(ReferentialIntegrityValidator().validate(resolved).valid)

    def test_workflow_reconciles_an_endpoint_blind_custom_resolver(self) -> None:
        source = SourceRecord.from_text(
            "Left depends on Right",
            source_name="memory://custom-resolver",
        )

        result = DraftConstructionWorkflow(
            mappers=[_StaticMapper()],
            resolvers=[_EndpointBlindResolver()],
        ).run([source])

        current_ids = {
            str(entity.properties["canonical_entity_id"])
            for entity in result.batch.entities
        }
        self.assertIn(result.batch.relations[0].source, current_ids)
        self.assertIn(result.batch.relations[0].target, current_ids)
        self.assertIn(result.batch.facts[0].subject, current_ids)
        self.assertIn(result.batch.facts[0].object, current_ids)
        self.assertTrue(result.validation_report.valid)
        graph_node_ids = {node.node_id for node in result.property_graph.nodes}
        for edge in result.property_graph.edges:
            self.assertIn(edge.source_id, graph_node_ids)
            self.assertIn(edge.target_id, graph_node_ids)

    def test_schema_domain_validation_survives_endpoint_canonicalization(self) -> None:
        source = SourceRecord.from_text(
            "Left depends on Right",
            source_name="memory://canonical-schema-validation",
        )
        registry = SchemaRegistry(
            GraphSchema(
                relationship_types=[
                    RelationshipTypeDefinition(
                        name="DEPENDS_ON",
                        source_types=["Person"],
                        target_types=["System"],
                    )
                ]
            )
        )

        result = DraftConstructionWorkflow(
            mappers=[_StaticMapper()],
            resolvers=[_EndpointBlindResolver()],
            validators=[SchemaRegistryValidator(registry)],
        ).run([source])

        canonical_ids = {
            str(entity.properties["canonical_entity_id"])
            for entity in result.batch.entities
        }
        self.assertIn(result.batch.relations[0].source, canonical_ids)
        self.assertIn(result.batch.relations[0].target, canonical_ids)
        domain_findings = [
            finding
            for finding in result.validation_report.findings
            if finding.rule == "relationship_domain_range"
        ]
        self.assertEqual(len(domain_findings), 1)
        self.assertIn("source_types=['System']", domain_findings[0].message)

    def test_schema_validator_rejects_mixed_type_canonical_cluster(self) -> None:
        source = SourceRecord.from_text(
            "Alex owns the platform",
            source_name="memory://mixed-type-cluster",
        )
        person = _entity(source, "Alex", entity_type="Person")
        organization = _entity(source, "Alex LLC", entity_type="Organization")
        platform = _entity(source, "Platform")
        shared_id = "canonical_entity_shared_alex"
        platform_id = "canonical_entity_platform"
        person = person.model_copy(
            update={"properties": {**person.properties, "canonical_entity_id": shared_id}}
        )
        organization = organization.model_copy(
            update={
                "properties": {
                    **organization.properties,
                    "canonical_entity_id": shared_id,
                }
            }
        )
        platform = platform.model_copy(
            update={"properties": {**platform.properties, "canonical_entity_id": platform_id}}
        )
        relation = RelationCandidate.build(
            source=shared_id,
            relation_type="OWNS",
            target=platform_id,
            evidence=[_evidence(source, "Alex owns the platform")],
            extractor_name="unit_test",
            technique="manual",
        )
        registry = SchemaRegistry(
            GraphSchema(
                relationship_types=[
                    RelationshipTypeDefinition(
                        name="OWNS",
                        source_types=["Person"],
                        target_types=["System"],
                    )
                ]
            )
        )

        report = SchemaRegistryValidator(registry).validate(
            ConstructionBatch(
                entities=[person, organization, platform],
                relations=[relation],
            )
        )

        self.assertFalse(report.valid)
        self.assertEqual(
            [finding.rule for finding in report.findings],
            ["relationship_domain_range"],
        )
        self.assertIn("Organization", report.findings[0].message)

    def test_referential_integrity_validator_rejects_obsolete_endpoints(self) -> None:
        source = SourceRecord.from_text("A relates to B", source_name="memory://stale")
        left = _entity(source, "A")
        right = _entity(source, "B")
        canonical_left = "canonical_entity_current_left"
        canonical_right = "canonical_entity_current_right"
        entities = [
            left.model_copy(
                update={"properties": {**left.properties, "canonical_entity_id": canonical_left}}
            ),
            right.model_copy(
                update={"properties": {**right.properties, "canonical_entity_id": canonical_right}}
            ),
        ]
        evidence = [_evidence(source, "A relates to B")]
        stale_relation = RelationCandidate.build(
            source=left.candidate_id,
            relation_type="RELATED_TO",
            target=right.candidate_id,
            evidence=evidence,
            extractor_name="unit_test",
            technique="manual",
        )

        report = ReferentialIntegrityValidator().validate(
            ConstructionBatch(entities=entities, relations=[stale_relation])
        )

        self.assertFalse(report.valid)
        self.assertEqual(len(report.findings), 2)
        self.assertEqual({finding.rule for finding in report.findings}, {"referential_integrity"})

    def test_exact_resolution_prefers_source_scoped_identity_keys(self) -> None:
        first = _entity(
            SourceRecord.from_text("one", source_name="memory://identity-key/one"),
            "[REDACTED]",
            identity_key="source-a:row-1",
        )
        second = _entity(
            SourceRecord.from_text("two", source_name="memory://identity-key/two"),
            "[REDACTED]",
            identity_key="source-a:row-2",
        )
        third = _entity(
            SourceRecord.from_text("three", source_name="memory://identity-key/three"),
            "Original Name",
            identity_key="source-a:row-1",
        )

        resolved = ExactKeyEntityResolver().resolve(
            ConstructionBatch(entities=[first, second, third])
        )
        canonical_ids = {
            entity.candidate_id: entity.properties["canonical_entity_id"]
            for entity in resolved.entities
        }

        self.assertEqual(canonical_ids[first.candidate_id], canonical_ids[third.candidate_id])
        self.assertNotEqual(canonical_ids[first.candidate_id], canonical_ids[second.candidate_id])

    def test_explicit_identity_keys_are_case_sensitive(self) -> None:
        source = SourceRecord.from_text("case-sensitive IDs", source_name="ids.txt")
        upper = _entity(source, "Account", identity_key="issuer:ABC")
        lower = _entity(source, "Account", identity_key="issuer:abc")

        resolved = ExactKeyEntityResolver().resolve(
            ConstructionBatch(entities=[upper, lower])
        )

        self.assertNotEqual(
            resolved.entities[0].properties["canonical_entity_id"],
            resolved.entities[1].properties["canonical_entity_id"],
        )

    def test_name_fallback_does_not_merge_namesakes_across_sources_by_default(self) -> None:
        first = _entity(
            SourceRecord.from_text("Alex Kim joined Research", source_name="hr.txt"),
            "Alex Kim",
            entity_type="Person",
        )
        second = _entity(
            SourceRecord.from_text("Alex Kim opened a ticket", source_name="support.txt"),
            "Alex Kim",
            entity_type="Person",
        )

        conservative = ExactKeyEntityResolver().resolve(
            ConstructionBatch(entities=[first, second])
        )
        global_name_baseline = ExactKeyEntityResolver(
            allow_cross_source_name_fallback=True
        ).resolve(ConstructionBatch(entities=[first, second]))

        self.assertNotEqual(
            conservative.entities[0].properties["canonical_entity_id"],
            conservative.entities[1].properties["canonical_entity_id"],
        )
        self.assertEqual(
            global_name_baseline.entities[0].properties["canonical_entity_id"],
            global_name_baseline.entities[1].properties["canonical_entity_id"],
        )


if __name__ == "__main__":
    unittest.main()
