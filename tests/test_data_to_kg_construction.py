from __future__ import annotations

import json
import tempfile
import unittest
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from kg_construction.certification.registry import default_registry, research_registry
from kg_construction.construction.config import DataToKGConfig
from kg_construction.domain.models import (
    ConstructionBatch,
    EntityCandidate,
    EvidenceRef,
    SourceRecord,
    stable_id,
)
from kg_construction.domain.schema import SchemaRegistry
from kg_construction.evaluation.validation_metrics import ValidationMetrics
from kg_construction.infrastructure.extractors.ollama_graph import OllamaGraphExtractor
from kg_construction.infrastructure.extractors.regex import RegexEntityExtractor
from kg_construction.infrastructure.graph_builders.evidence_graph import EvidenceGraphBuilder
from kg_construction.infrastructure.graph_builders.property_graph import PropertyGraphBuilder
from kg_construction.infrastructure.graph_builders.rdf_graph import RDFGraphBuilder
from kg_construction.infrastructure.graph_stores.exporters import Neo4jExporter, RDFExporter
from kg_construction.infrastructure.llm.ollama import OllamaConfig, OllamaProvider
from kg_construction.infrastructure.loaders.csv_loader import CsvLoader
from kg_construction.infrastructure.loaders.stubs import AdapterNotConfigured
from kg_construction.infrastructure.loaders.text_loader import TextLoader
from kg_construction.infrastructure.mapping.direct_table_mapper import (
    DirectTableMapper,
    DirectTableMappingConfig,
    ForeignKeyMappingConfig,
)
from kg_construction.infrastructure.mapping.semi_structured import (
    JsonArrayEdgeMapping,
    JsonPathMapper,
    JsonPathMappingConfig,
    JsonPropertyMapping,
)
from kg_construction.infrastructure.mapping.standards import (
    CSVWMapper,
    PredicateObjectMapConfig,
    R2RMLMapper,
    RMLMapper,
    RMLMappingConfig,
    TermMapConfig,
    TriplesMapConfig,
)
from kg_construction.infrastructure.ontology import SchemaProposalWorkflow, ShaclValidator
from kg_construction.infrastructure.ontology.validation import (
    EvidenceRequiredValidator,
    SchemaRegistryValidator,
    ValidationReport,
)
from kg_construction.infrastructure.resolution import (
    AliasTableResolver,
    DedupeActiveLearningResolver,
    NormalizedStringResolver,
    RapidFuzzResolver,
    RapidFuzzResolverConfig,
    SplinkProbabilisticResolver,
)
from kg_construction.infrastructure.resolution.exact import ExactKeyEntityResolver
from kg_construction.interfaces.cli.construct import main as construct_main
from kg_construction.research.hybrid_gliner import (
    HybridGlinerPipelineConfig,
    extract_entities_with_ollama_labels_and_gliner,
)
from kg_construction.research.planner import TechniquePlanner


class FakeGlinerModel:
    def predict_entities(
        self,
        text: str,
        labels: list[str],
        *,
        threshold: float = 0.5,
        flat_ner: bool = True,
    ) -> list[dict[str, object]]:
        _ = threshold, flat_ner
        entities: list[dict[str, object]] = []
        spans = {
            "Incident": "INCIDENT-1042",
            "System": "Recommendation API",
            "Organization": "Platform Team",
            "Date": "2026-07-09",
        }
        for label, value in spans.items():
            if label not in labels:
                continue
            start = text.find(value)
            if start >= 0:
                entities.append(
                    {
                        "text": value,
                        "label": label,
                        "start": start,
                        "end": start + len(value),
                        "score": 0.91,
                    }
                )
        return entities


class DataToKGConstructionTests(unittest.TestCase):
    def test_source_and_evidence_ids_are_deterministic(self) -> None:
        left = SourceRecord.from_text("Ada works at Acme.", source_name="memory://demo")
        right = SourceRecord.from_text("Ada works at Acme.", source_name="memory://demo")

        self.assertEqual(left.source_id, right.source_id)
        self.assertEqual(
            EvidenceRef.build(left, location={"kind": "span", "start": 0}, text="Ada").evidence_id,
            EvidenceRef.build(right, location={"kind": "span", "start": 0}, text="Ada").evidence_id,
        )
        self.assertEqual(stable_id("x", "A", 1), stable_id("x", "A", 1))

    def test_csv_direct_mapping_builds_entities_relations_and_evidence(self) -> None:
        source = CsvLoader().load("examples/data_to_kg/sample.csv")[0]
        mapper = DirectTableMapper(
            DirectTableMappingConfig(
                table_name="people",
                node_type="Person",
                id_columns=["id"],
                name_column="name",
                foreign_keys=[
                    ForeignKeyMappingConfig(
                        source_column="manager_id",
                        target_column="id",
                        relationship_type="REPORTS_TO",
                    )
                ],
            )
        )
        batch = mapper.map_source(source)

        self.assertEqual(len(batch.entities), 3)
        self.assertEqual(len(batch.relations), 2)
        self.assertTrue(all(entity.evidence for entity in batch.entities))
        self.assertIn("profile", batch.metadata)

    def test_schema_proposal_workflow_generates_schema_shapes_and_registry_validation(self) -> None:
        source = CsvLoader().load("examples/data_to_kg/sample.csv")[0]
        workflow = SchemaProposalWorkflow()
        proposal = workflow.propose([source], schema_id="sample_people")
        schema = proposal.graph_schema

        person = next(node for node in schema.node_types if node.name == "Person")
        reports_to = next(
            relationship
            for relationship in schema.relationship_types
            if relationship.name == "REPORTS_TO"
        )

        self.assertEqual(schema.status, "proposed")
        self.assertEqual(person.primary_keys, ["id"])
        self.assertEqual(person.source_table, "sample")
        self.assertEqual(reports_to.source_types, ["Person"])
        self.assertEqual(reports_to.target_types, ["Person"])
        self.assertIn("kg:PersonShape a sh:NodeShape", proposal.shacl_turtle)
        self.assertIn("kg:REPORTS_TO a owl:ObjectProperty", proposal.owl_turtle or "")

        approved = workflow.approve(schema)
        promoted = workflow.promote(approved)
        rejected = workflow.reject(schema)

        self.assertEqual(approved.status, "approved")
        self.assertEqual(promoted.status, "promoted")
        self.assertEqual(rejected.status, "rejected")

        batch = DirectTableMapper(
            DirectTableMappingConfig(
                table_name="people",
                node_type="Person",
                id_columns=["id"],
                name_column="name",
                foreign_keys=[
                    ForeignKeyMappingConfig(
                        source_column="manager_id",
                        target_column="id",
                        relationship_type="REPORTS_TO",
                    )
                ],
            )
        ).map_source(source)
        validation = SchemaRegistryValidator(SchemaRegistry(promoted)).validate(batch)

        self.assertTrue(validation.valid)
        self.assertEqual(validation.findings, [])

    def test_schema_proposal_workflow_reports_schema_diffs_across_runs(self) -> None:
        workflow = SchemaProposalWorkflow()
        first_source = CsvLoader().load("examples/data_to_kg/sample.csv")[0]
        first = workflow.propose([first_source], schema_id="sample_people").graph_schema
        second_source = SourceRecord.from_rows(
            [
                {
                    "id": "1",
                    "name": "Ada Lovelace",
                    "email": "ada@example.com",
                    "manager_id": "",
                    "team": "Research",
                    "location": "London",
                }
            ],
            source_type="csv",
            source_name="people.csv",
        )
        second = workflow.propose(
            [second_source],
            schema_id="sample_people",
            previous_schema=first,
        )

        self.assertIsNotNone(second.diff_report)
        self.assertTrue(second.diff_report and second.diff_report.has_changes)
        self.assertIn("Person", second.diff_report.changed_node_types if second.diff_report else [])

    def test_jsonpath_mapping_preserves_source_paths_and_array_edges(self) -> None:
        source = SourceRecord.from_json(
            {
                "service": {
                    "id": "svc-rec-api",
                    "name": "Recommendation API",
                },
                "errors": [
                    {"code": "E1042", "message": "Timeout"},
                    {"code": "E1043", "message": "Retry limit exceeded"},
                ],
            },
            source_name="memory://jsonpath",
        )
        mapper = JsonPathMapper(
            JsonPathMappingConfig(
                node_type="ServicePayload",
                name_path="$.service.name",
                property_paths=[
                    JsonPropertyMapping(property_name="service_name", path="$.service.name"),
                    JsonPropertyMapping(property_name="first_error_code", path="$.errors[0].code"),
                    JsonPropertyMapping(property_name="missing_required", path="$.service.owner", required=True),
                ],
                array_edges=[
                    JsonArrayEdgeMapping(
                        path="$.errors[*]",
                        relationship_type="HAS_ERROR",
                        target_node_type="Error",
                        target_name_path="$.code",
                    )
                ],
            )
        )
        batch = mapper.map_source(source)

        root = next(entity for entity in batch.entities if entity.entity_type == "ServicePayload")
        errors = [entity for entity in batch.entities if entity.entity_type == "Error"]

        self.assertEqual(root.name, "Recommendation API")
        self.assertEqual(root.properties["service_name"], "Recommendation API")
        self.assertEqual(root.properties["first_error_code"], "E1042")
        self.assertEqual(root.properties["_json_property_paths"]["service_name"], "$.service.name")
        self.assertEqual(root.properties["_json_property_paths"]["first_error_code"], "$.errors[0].code")
        self.assertNotIn("missing_required", root.properties)
        self.assertEqual(batch.metadata["missing_paths"], ["$.service.owner"])
        self.assertEqual(len(errors), 2)
        self.assertEqual(
            {entity.properties["_json_path"] for entity in errors},
            {"$.errors[0]", "$.errors[1]"},
        )
        self.assertEqual(
            {relation.properties["source_path"] for relation in batch.relations},
            {"$.errors[0]", "$.errors[1]"},
        )

        report = EvidenceRequiredValidator().validate(batch)
        property_graph = PropertyGraphBuilder().build(batch, validation_report=report)
        root_node = next(
            node
            for node in property_graph.nodes
            if node.properties.get("candidate_id") == root.candidate_id
        )
        first_error_edge = next(
            edge
            for edge in property_graph.edges
            if edge.properties.get("source_path") == "$.errors[0]"
        )

        self.assertIn("$.service.name", root_node.properties["source_paths"])
        self.assertIn("$.errors[0].code", root_node.properties["source_paths"])
        self.assertIn("$.errors[0]", first_error_edge.properties["source_paths"])

    def test_jsonpath_mapping_keeps_repeated_array_objects_distinct(self) -> None:
        source = SourceRecord.from_json(
            {
                "service": {"name": "Recommendation API"},
                "errors": [
                    {"code": "E1042"},
                    {"code": "E1042"},
                ],
            },
            source_name="memory://jsonpath-repeated",
        )
        batch = JsonPathMapper().map_source(source)
        error_entities = [entity for entity in batch.entities if entity.entity_type == "Error"]

        self.assertEqual(len(error_entities), 2)
        self.assertEqual(len({entity.candidate_id for entity in error_entities}), 2)
        self.assertEqual(
            {entity.properties["_json_path"] for entity in error_entities},
            {"$.errors[0]", "$.errors[1]"},
        )

    def test_r2rml_mapping_turtle_generates_evidence_backed_rdf_facts(self) -> None:
        source = SourceRecord.from_rows(
            [
                {"id": "1", "name": "Ada Lovelace", "manager_id": ""},
                {"id": "2", "name": "Grace Hopper", "manager_id": "1"},
            ],
            source_type="sql",
            source_name="sqlite://memory#people",
            metadata={"table": "people"},
        )
        mapping = """
            @prefix rr: <http://www.w3.org/ns/r2rml#> .
            @prefix ex: <http://example.com/> .

            <#PersonMap>
                rr:logicalTable [ rr:tableName "people" ];
                rr:subjectMap [
                    rr:template "http://example.com/person/{id}";
                    rr:class ex:Person
                ];
                rr:predicateObjectMap [
                    rr:predicate ex:name;
                    rr:objectMap [ rr:column "name" ]
                ];
                rr:predicateObjectMap [
                    rr:predicate ex:manager;
                    rr:objectMap [
                        rr:template "http://example.com/person/{manager_id}";
                        rr:termType rr:IRI
                    ]
                ] .
        """

        batch = R2RMLMapper.from_turtle(mapping).map_source(source)
        report = EvidenceRequiredValidator().validate(batch)
        rdf_graph = RDFGraphBuilder().build(batch, validation_report=report)
        triples = {(item.subject, item.predicate, item.object, item.object_type) for item in rdf_graph.triples}

        self.assertEqual(len(batch.entities), 2)
        self.assertTrue(all(fact.evidence for fact in batch.facts))
        self.assertIn(
            (
                "http://example.com/person/1",
                "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
                "http://example.com/Person",
                "iri",
            ),
            triples,
        )
        self.assertIn(
            ("http://example.com/person/2", "http://example.com/manager", "http://example.com/person/1", "iri"),
            triples,
        )
        self.assertIn(
            ("http://example.com/person/1", "http://example.com/name", "Ada Lovelace", "literal"),
            triples,
        )

    def test_rml_mapping_supports_csv_json_and_xml_logical_sources(self) -> None:
        csv_source = SourceRecord.from_rows(
            [{"id": "1", "name": "Ada Lovelace"}],
            source_type="csv",
            source_name="people.csv",
        )
        csv_mapping = """
            @prefix rr: <http://www.w3.org/ns/r2rml#> .
            @prefix rml: <http://semweb.mmlab.be/ns/rml#> .
            @prefix ql: <http://semweb.mmlab.be/ns/ql#> .
            @prefix ex: <http://example.com/> .

            <#PeopleCsv>
                rml:logicalSource [
                    rml:source "people.csv";
                    rml:referenceFormulation ql:CSV
                ];
                rr:subjectMap [
                    rr:template "http://example.com/person/{id}";
                    rr:class ex:Person
                ];
                rr:predicateObjectMap [
                    rr:predicate ex:name;
                    rr:objectMap [ rml:reference "name" ]
                ] .
        """
        csv_batch = RMLMapper.from_turtle(csv_mapping).map_source(csv_source)

        json_source = SourceRecord.from_json(
            {"services": [{"id": "svc-rec", "name": "Recommendation API"}]},
            source_name="services.json",
        )
        json_mapping = """
            @prefix rr: <http://www.w3.org/ns/r2rml#> .
            @prefix rml: <http://semweb.mmlab.be/ns/rml#> .
            @prefix ql: <http://semweb.mmlab.be/ns/ql#> .
            @prefix ex: <http://example.com/> .

            <#ServicesJson>
                rml:logicalSource [
                    rml:source "services.json";
                    rml:iterator "$.services[*]";
                    rml:referenceFormulation ql:JSONPath
                ];
                rr:subjectMap [
                    rr:template "http://example.com/service/{id}";
                    rr:class ex:Service
                ];
                rr:predicateObjectMap [
                    rr:predicate ex:name;
                    rr:objectMap [ rml:reference "name" ]
                ] .
        """
        json_batch = RMLMapper.from_turtle(json_mapping).map_source(json_source)

        xml_source = SourceRecord.from_rows(
            [{"id": "svc-rec", "name": "Recommendation API"}],
            source_type="xml",
            source_name="services.xml",
            metadata={"row_paths": ["/services/service[1]"]},
        )
        xml_mapper = RMLMapper(
            RMLMappingConfig(
                triples_maps=[
                    TriplesMapConfig(
                        mapping_id="ServicesXml",
                        logical_source="services.xml",
                        iterator="/services/service",
                        reference_formulation="xpath",
                        subject_map=TermMapConfig(template="http://example.com/service/{id}"),
                        class_iris=["http://example.com/Service"],
                        predicate_object_maps=[
                            PredicateObjectMapConfig(
                                predicate_map=TermMapConfig(constant="http://example.com/name", term_type="iri"),
                                object_map=TermMapConfig(reference="name", term_type="literal"),
                            )
                        ],
                    )
                ]
            )
        )
        xml_batch = xml_mapper.map_source(xml_source)

        self.assertEqual(csv_batch.facts[1].properties["rdf_object"], "Ada Lovelace")
        self.assertEqual(json_batch.entities[0].properties["rdf_subject_iri"], "http://example.com/service/svc-rec")
        self.assertEqual(xml_batch.facts[1].evidence[0].location["kind"], "cell")

    def test_csvw_mapping_uses_primary_keys_columns_and_foreign_keys(self) -> None:
        source = SourceRecord.from_rows(
            [
                {"id": "1", "name": "Ada Lovelace", "manager_id": ""},
                {"id": "2", "name": "Grace Hopper", "manager_id": "1"},
            ],
            source_type="csv",
            source_name="people.csv",
        )
        metadata = {
            "tables": [
                {
                    "url": "people.csv",
                    "aboutUrl": "http://example.com/person/{id}",
                    "tableSchema": {
                        "columns": [
                            {"name": "id", "datatype": "integer", "suppressOutput": True},
                            {"name": "name", "datatype": "string", "propertyUrl": "http://schema.org/name"},
                            {"name": "manager_id", "suppressOutput": True},
                        ],
                        "primaryKey": "id",
                        "foreignKeys": [
                            {
                                "columnReference": "manager_id",
                                "reference": {
                                    "resource": "people.csv",
                                    "columnReference": "id",
                                },
                            }
                        ],
                    },
                }
            ]
        }

        batch = CSVWMapper(metadata, base_iri="http://example.com/csvw/").map_source(source)
        report = EvidenceRequiredValidator().validate(batch)
        rdf_graph = RDFGraphBuilder().build(batch, validation_report=report)
        triples = {(item.subject, item.predicate, item.object, item.object_type) for item in rdf_graph.triples}

        self.assertEqual(len(batch.entities), 2)
        self.assertEqual(len(batch.relations), 1)
        self.assertIn(
            ("http://example.com/person/1", "http://schema.org/name", "Ada Lovelace", "literal"),
            triples,
        )
        self.assertIn(
            (
                "http://example.com/person/2",
                "http://example.com/csvw/foreignKey/manager_id",
                "http://example.com/person/1",
                "iri",
            ),
            triples,
        )
        self.assertEqual(batch.metadata["primary_keys"], ["id"])

    def test_text_regex_extractor_preserves_span_evidence(self) -> None:
        source = TextLoader().load("examples/data_to_kg/sample.txt")[0]
        batch = RegexEntityExtractor().extract(source)
        labels = {entity.entity_type for entity in batch.entities}

        self.assertIn("EmailAddress", labels)
        self.assertIn("URL", labels)
        self.assertIn("Date", labels)
        self.assertTrue(all(entity.evidence[0].location["kind"] == "span" for entity in batch.entities))

    def test_validation_rejects_candidates_without_evidence(self) -> None:
        entity = EntityCandidate(
            candidate_id="entity_without_evidence",
            entity_type="Person",
            name="Ada",
            extractor_name="unit_test",
            technique="manual",
        )
        report = EvidenceRequiredValidator().validate(ConstructionBatch(entities=[entity]))

        self.assertFalse(report.valid)
        self.assertIn("entity_without_evidence", report.invalid_ids)

    def test_validation_metrics_reports_unsupported_extraction_rate(self) -> None:
        source = SourceRecord.from_text("INCIDENT-1042 affected Recommendation API.")
        supported = EntityCandidate.build(
            entity_type="Identifier",
            name="INCIDENT-1042",
            evidence=[
                EvidenceRef.build(
                    source,
                    location={"kind": "span", "start": 0, "end": 13},
                    text="INCIDENT-1042",
                )
            ],
            extractor_name="regex",
            technique="regex",
        )
        unsupported = EntityCandidate.build(
            entity_type="System",
            name="Recommendation API",
            evidence=[],
            extractor_name="llm",
            technique="ollama",
        )
        batch = ConstructionBatch(entities=[supported, unsupported])

        metrics = ValidationMetrics().evaluate(batch, ValidationReport())

        self.assertEqual(metrics["unsupported_extraction_rate"], 0.5)
        self.assertEqual(metrics["hallucination_risk_proxy"], 0.5)

    def test_entity_resolution_handles_normalized_alias_and_noisy_duplicates_with_audit(self) -> None:
        text = "Recommendation API recommendation api Rec API Recomendation API Feature Store"
        source = SourceRecord.from_text(text, source_name="memory://entity-resolution")

        def entity(name: str) -> EntityCandidate:
            start = text.find(name)
            return EntityCandidate.build(
                entity_type="System",
                name=name,
                evidence=[
                    EvidenceRef.build(
                        source,
                        location={"kind": "span", "start": start, "end": start + len(name)},
                        text=name,
                    )
                ],
                extractor_name="unit_test",
                technique="manual",
            )

        batch = ConstructionBatch(
            source_records=[source],
            entities=[
                entity("Recommendation API"),
                entity("recommendation api"),
                entity("Rec API"),
                entity("Recomendation API"),
                entity("Feature Store"),
            ],
        )
        resolved = NormalizedStringResolver().resolve(batch)
        resolved = RapidFuzzResolver(RapidFuzzResolverConfig(threshold=88, review_threshold=99)).resolve(resolved)
        resolved = AliasTableResolver(
            aliases={"Recommendation API": ["Rec API", "RecommendationAPI"]}
        ).resolve(resolved)

        by_name = {item.name: item for item in resolved.entities}
        recommendation_ids = {
            by_name[name].properties["canonical_entity_id"]
            for name in [
                "Recommendation API",
                "recommendation api",
                "Rec API",
                "Recomendation API",
            ]
        }

        self.assertEqual(len(recommendation_ids), 1)
        self.assertNotEqual(
            by_name["Feature Store"].properties["canonical_entity_id"],
            next(iter(recommendation_ids)),
        )
        noisy = by_name["Recomendation API"]
        self.assertEqual(noisy.properties["canonical_name"], "Recommendation API")
        self.assertTrue(noisy.properties["review_required"])
        self.assertGreater(noisy.properties["resolution_confidence"], 0.9)
        self.assertIn("merged_from", noisy.properties)

        audit = resolved.metadata["entity_resolution_audit"]
        recommendation_audits = [
            row
            for row in audit
            if row["canonical_entity_id"] == next(iter(recommendation_ids))
        ]
        self.assertTrue(recommendation_audits)
        self.assertTrue(any(row["review_required"] for row in recommendation_audits))
        self.assertTrue(
            any(
                {"Recommendation API", "recommendation api", "Rec API", "Recomendation API"}
                <= set(row["merged_names"])
                for row in recommendation_audits
            )
        )

    def test_splink_and_dedupe_adapters_explain_required_configuration(self) -> None:
        batch = ConstructionBatch()

        with self.assertRaises(AdapterNotConfigured):
            SplinkProbabilisticResolver().resolve(batch)
        with self.assertRaises(AdapterNotConfigured):
            DedupeActiveLearningResolver().resolve(batch)

    def test_property_rdf_and_evidence_graphs_include_provenance(self) -> None:
        source = CsvLoader().load("examples/data_to_kg/sample.csv")[0]
        batch = DirectTableMapper(
            DirectTableMappingConfig(node_type="Person", id_columns=["id"], name_column="name")
        ).map_source(source)
        batch = ExactKeyEntityResolver().resolve(batch)
        report = EvidenceRequiredValidator().validate(batch)
        property_graph = PropertyGraphBuilder().build(batch, validation_report=report)
        rdf_graph = RDFGraphBuilder().build(batch, validation_report=report)
        evidence_graph = EvidenceGraphBuilder().build(batch)

        self.assertTrue(property_graph.nodes)
        self.assertTrue(rdf_graph.triples)
        self.assertTrue(evidence_graph.edges)
        first_entity_node = next(node for node in property_graph.nodes if "Entity" in node.labels)
        self.assertIn("evidence_ids", first_entity_node.properties)
        self.assertTrue(RDFExporter().export_rdf_graph(rdf_graph)["turtle"])
        self.assertTrue(Neo4jExporter().export_property_graph(property_graph)[0].startswith("MERGE"))

    def test_rdf_output_validates_against_generated_shacl_shapes(self) -> None:
        source = CsvLoader().load("examples/data_to_kg/sample.csv")[0]
        schema_proposal = SchemaProposalWorkflow().propose([source], schema_id="sample_people")
        batch = DirectTableMapper(
            DirectTableMappingConfig(
                table_name="people",
                node_type="Person",
                id_columns=["id"],
                name_column="name",
                foreign_keys=[
                    ForeignKeyMappingConfig(
                        source_column="manager_id",
                        target_column="id",
                        relationship_type="REPORTS_TO",
                    )
                ],
            )
        ).map_source(source)
        batch = ExactKeyEntityResolver().resolve(batch)
        validation_report = EvidenceRequiredValidator().validate(batch)
        property_graph = PropertyGraphBuilder().build(batch, validation_report=validation_report)
        rdf_graph = RDFGraphBuilder().build(batch, validation_report=validation_report)
        shacl_result = ShaclValidator().validate_rdf_graph(
            rdf_graph,
            graph_schema=schema_proposal.graph_schema,
        )
        rdf_payload = RDFExporter().export_rdf_graph(rdf_graph)

        self.assertTrue(shacl_result.conforms, shacl_result.results_text)
        self.assertIn("kg:PersonShape", shacl_result.shacl_turtle)
        self.assertIn("urn:kg:graph:evidence", rdf_payload["nquads"])
        self.assertIn("urn:kg:graph:data", rdf_payload["nquads"])
        self.assertGreaterEqual(
            len(
                {
                    triple.subject
                    for triple in rdf_graph.triples
                    if triple.predicate.endswith("rdf-syntax-ns#type")
                    and triple.object.endswith("Person")
                }
            ),
            len([node for node in property_graph.nodes if "Person" in node.labels]),
        )

    def test_ollama_provider_retries_invalid_json_and_extractor_returns_candidates(self) -> None:
        responses: Iterator[str | dict[str, Any]] = iter(
            [
                "not valid json",
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "nodes": [
                                    {
                                        "id_hint": "ada",
                                        "type": "Person",
                                        "name": "Ada",
                                        "properties": {},
                                        "evidence_text": "Ada",
                                        "confidence": 0.91,
                                    },
                                    {
                                        "id_hint": "acme",
                                        "type": "Organization",
                                        "name": "Acme",
                                        "properties": {},
                                        "evidence_text": "Acme",
                                        "confidence": 0.9,
                                    },
                                ],
                                "relationships": [
                                    {
                                        "source": "ada",
                                        "type": "WORKS_AT",
                                        "target": "acme",
                                        "properties": {},
                                        "evidence_text": "Ada works at Acme",
                                        "confidence": 0.88,
                                    }
                                ],
                            }
                        )
                    }
                },
            ]
        )

        provider = OllamaProvider(
            OllamaConfig(max_retries=2),
            transport=lambda _body: next(responses),
        )
        source = SourceRecord.from_text("Ada works at Acme.", source_name="memory://ollama")
        batch = OllamaGraphExtractor(
            provider,
            allowed_node_types=["Person", "Organization"],
            allowed_relationship_types=["WORKS_AT"],
        ).extract(source)

        self.assertEqual(len(batch.entities), 2)
        self.assertEqual(len(batch.relations), 1)
        self.assertTrue(batch.relations[0].evidence)

    def test_ollama_label_proposal_plus_gliner_extracts_evidence_spans(self) -> None:
        text = (
            "INCIDENT-1042 affected Recommendation API on 2026-07-09. "
            "Platform Team escalated the issue."
        )

        def transport(body: dict[str, object]) -> dict[str, object]:
            prompt = str(body["messages"][0]["content"])  # type: ignore[index]
            if "Do not extract entities yet" in prompt:
                return {
                    "message": {
                        "content": json.dumps(
                            {
                                "labels": [
                                    {
                                        "name": "Incident",
                                        "description": "Operational incident identifier.",
                                        "rationale": "Incident IDs appear in the text.",
                                        "confidence": 0.95,
                                    },
                                    {
                                        "name": "System",
                                        "description": "Software service or API.",
                                        "rationale": "Recommendation API is a system.",
                                        "confidence": 0.93,
                                    },
                                    {
                                        "name": "Organization",
                                        "description": "Team or business unit.",
                                        "rationale": "Platform Team appears in the text.",
                                        "confidence": 0.9,
                                    },
                                    {
                                        "name": "Date",
                                        "description": "Calendar date.",
                                        "rationale": "A date appears in the text.",
                                        "confidence": 0.88,
                                    },
                                    {
                                        "name": "Thing",
                                        "description": "Too broad.",
                                        "rationale": "Should be filtered.",
                                        "confidence": 0.99,
                                    },
                                ]
                            }
                        )
                    }
                }
            return {
                "message": {
                    "content": json.dumps(
                        {
                            "nodes": [
                                {
                                    "id_hint": "incident",
                                    "type": "Incident",
                                    "name": "INCIDENT-1042",
                                    "properties": {},
                                    "evidence_text": "INCIDENT-1042",
                                    "confidence": 0.9,
                                },
                                {
                                    "id_hint": "system",
                                    "type": "System",
                                    "name": "Recommendation API",
                                    "properties": {},
                                    "evidence_text": "Recommendation API",
                                    "confidence": 0.9,
                                },
                            ],
                            "relationships": [],
                        }
                    )
                }
            }

        provider = OllamaProvider(OllamaConfig(), transport=transport)
        config = HybridGlinerPipelineConfig.from_yaml("configs/ollama_gliner_label_proposal.example.yaml").model_copy(
            update={"compare_ollama_graph": True}
        )
        result = extract_entities_with_ollama_labels_and_gliner(
            text,
            config=config,
            provider=provider,
            gliner_model=FakeGlinerModel(),
        )

        self.assertEqual(
            result.reviewed_labels,
            ["Incident", "System", "Organization", "Date"],
        )
        self.assertNotIn("Thing", result.reviewed_labels)
        self.assertEqual(len(result.gliner_batch.entities), 4)
        self.assertTrue(all(entity.evidence for entity in result.gliner_batch.entities))
        self.assertEqual(
            {
                entity.name: (
                    entity.evidence[0].location["start"],
                    entity.evidence[0].location["end"],
                )
                for entity in result.gliner_batch.entities
            },
            {
                "INCIDENT-1042": (0, 13),
                "Recommendation API": (23, 41),
                "2026-07-09": (45, 55),
                "Platform Team": (57, 70),
            },
        )
        comparison_names = {row.technique for row in result.comparisons}
        self.assertTrue(
            {
                "ollama_labels_plus_gliner",
                "regex_entity_extractor",
                "gazetteer_entity_extractor",
                "ollama_graph_extractor",
            }
            <= comparison_names
        )

    def test_registry_covers_required_categories_and_techniques(self) -> None:
        registry = default_registry()
        required_categories = {
            "loader",
            "mapper",
            "extractor",
            "resolution",
            "schema",
            "validation",
            "graph_builder",
            "graph_store",
            "llm",
            "evaluation",
            "hybrid",
        }
        required_names = {
            "csv_loader",
            "excel_loader",
            "sql_database_loader",
            "json_loader",
            "pdf_loader",
            "direct_table_mapper",
            "r2rml_mapper",
            "rml_mapper",
            "csvw_mapper",
            "obda_virtual_graph_mapper",
            "jsonpath_mapper",
            "regex_entity_extractor",
            "gazetteer_entity_extractor",
            "gliner_entity_extractor",
            "openie_relation_extractor",
            "evidence_first_unstructured_pipeline",
            "ollama_graph_extractor",
            "exact_key_entity_resolver",
            "embedding_resolver",
            "wikidata_linker",
            "schema_registry_validator",
            "shacl_validator",
            "property_graph_builder",
            "rdf_graph_builder",
            "evidence_graph_builder",
            "neo4j_exporter",
            "neo4j_ingestion_adapter",
            "rdf_exporter",
            "rdflib_exporter",
        }

        self.assertTrue(required_categories <= registry.categories())
        self.assertTrue(required_names <= registry.names())
        self.assertNotIn("benchmark_runner", registry.names())
        self.assertIn("benchmark_runner", research_registry().names())

    def test_planner_prefers_deterministic_before_llm_and_human_review(self) -> None:
        registry = research_registry()
        planned = TechniquePlanner().plan(
            [
                registry.get("human_review_resolver"),
                registry.get("ollama_graph_extractor"),
                registry.get("direct_table_mapper"),
                registry.get("regex_entity_extractor"),
            ]
        )

        self.assertEqual([item.name for item in planned][0], "direct_table_mapper")
        self.assertEqual([item.name for item in planned][-1], "human_review_resolver")

    def test_config_loads_complete_example(self) -> None:
        config = DataToKGConfig.from_yaml("configs/data_to_kg_complete.example.yaml")

        self.assertEqual(config.llm.provider, "ollama")
        self.assertIn("csv", config.enabled_techniques()["loaders"])

    def test_cli_constructs_without_a_suite_and_always_writes_a_graph(self) -> None:
        # A missing evaluation suite no longer ends the conversion: the CLI
        # always writes the graph and its metrics.
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "graph.json"
            exit_code = construct_main(
                [
                    "--input",
                    "examples/data_to_kg/sample.txt",
                    "--source-type",
                    "text",
                    "--output",
                    str(output),
                ]
            )
            self.assertEqual(exit_code, 0)
            self.assertTrue(output.exists())


if __name__ == "__main__":
    unittest.main()
