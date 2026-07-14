from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from kg_construction.application.graph_store_benchmark import generate_graph_store_benchmark_fixture
from kg_construction.application.registry import default_registry
from kg_construction.domain.graph_objects import RDFGraph, RDFTriple
from kg_construction.domain.models import (
    ConstructionBatch,
    EntityCandidate,
    EvidenceRef,
    GraphFact,
    RelationCandidate,
    SourceRecord,
)
from kg_construction.infrastructure.enrichment import (
    GraphRetrievalBenchmark,
    GraphSageLinkPredictionTrainer,
    GraphTrainValidationSplitter,
    KnowledgeGraphEmbeddingAdapter,
    LargeGraphAnalyticsBackend,
    Node2VecEmbeddingAdapter,
)
from kg_construction.infrastructure.evaluation import GoldBenchmarkEvaluator, GoldBenchmarkFixture
from kg_construction.infrastructure.graph_stores import (
    KuzuIngestionAdapter,
    KuzuIngestionConfig,
    Neo4jIngestionAdapter,
    Neo4jIngestionConfig,
)
from kg_construction.infrastructure.loaders import EventPayloadLoader, LogLoader
from kg_construction.infrastructure.mapping import (
    CSVWColumnConfig,
    CSVWDialectConfig,
    CSVWDialectParser,
    CSVWMapper,
    CSVWMappingConfig,
    CSVWTableConfig,
    CSVWTableSchemaConfig,
    EventGraphMapper,
    JoinConditionConfig,
    JSONLDMapper,
    OBDAPredicateMapping,
    OBDAVirtualGraphMapper,
    PredicateObjectMapConfig,
    R2RMLMapper,
    R2RMLMappingConfig,
    TermMapConfig,
    TriplesMapConfig,
)
from kg_construction.infrastructure.mapping.semi_structured import LogEventMapper


class FakeResult:
    def consume(self) -> None:
        return None


class FakeTx:
    def __init__(self) -> None:
        self.runs: list[tuple[str, dict[str, object]]] = []

    def run(self, query: str, **params: object) -> FakeResult:
        self.runs.append((query, params))
        return FakeResult()


class FakeSession:
    def __init__(self, driver: FakeNeo4jDriver) -> None:
        self.driver = driver

    def __enter__(self) -> FakeSession:
        return self

    def __exit__(self, _exc_type: object, exc: object, _tb: object) -> None:
        return None

    def execute_write(self, callback: object, *args: object) -> object:
        tx = FakeTx()
        result = callback(tx, *args)  # type: ignore[misc]
        self.driver.runs.extend(tx.runs)
        return result


class FakeNeo4jDriver:
    def __init__(self) -> None:
        self.runs: list[tuple[str, dict[str, object]]] = []

    def session(self, **kwargs: object) -> FakeSession:
        _ = kwargs
        return FakeSession(self)


class FakeKuzuConnection:
    def __init__(self) -> None:
        self.executions: list[tuple[str, dict[str, object] | None]] = []

    def execute(self, query: str, parameters: dict[str, object] | None = None) -> FakeResult:
        self.executions.append((query, parameters))
        return FakeResult()


class NextStageUpgradeTests(unittest.TestCase):
    def test_jsonld_mapper_generates_rdf_facts_and_expansion_metadata(self) -> None:
        source = SourceRecord.from_json(
            {
                "@context": {"name": "http://schema.org/name"},
                "@id": "urn:svc:recommendation",
                "name": "Recommendation API",
            },
            source_name="memory://jsonld",
        ).model_copy(update={"source_type": "jsonld"})

        batch = JSONLDMapper().map_source(source)

        self.assertTrue(batch.facts)
        self.assertEqual(batch.metadata["standard"], "JSON-LD")
        self.assertIn("expanded", batch.metadata)

    def test_r2rml_referencing_object_map_cross_source_join(self) -> None:
        people = SourceRecord.from_rows(
            [{"id": "1", "name": "Ada"}, {"id": "2", "name": "Grace", "manager_id": "1"}],
            source_type="sql",
            source_name="memory://db#people",
            metadata={"table": "people"},
        )
        managers = SourceRecord.from_rows(
            [{"id": "1", "name": "Ada"}],
            source_type="sql",
            source_name="memory://db#managers",
            metadata={"table": "managers"},
        )
        config = R2RMLMappingConfig(
            triples_maps=[
                TriplesMapConfig(
                    mapping_id="people_map",
                    logical_table="people",
                    subject_map=TermMapConfig(template="urn:person:{id}"),
                    predicate_object_maps=[
                        PredicateObjectMapConfig(
                            predicate_map=TermMapConfig(constant="urn:kg:reportsTo"),
                            parent_triples_map="manager_map",
                            join_conditions=[JoinConditionConfig(child="manager_id", parent="id")],
                        )
                    ],
                ),
                TriplesMapConfig(
                    mapping_id="manager_map",
                    logical_table="managers",
                    subject_map=TermMapConfig(template="urn:person:{id}"),
                ),
            ]
        )
        batch = R2RMLMapper(config).map_sources([people, managers])

        self.assertTrue(any(fact.properties.get("referencing_object_map") for fact in batch.facts))
        self.assertTrue(any(fact.object == "urn:person:1" for fact in batch.facts))

    def test_csvw_dialect_parser_and_validation(self) -> None:
        rows = CSVWDialectParser(
            CSVWDialectConfig(delimiter=";", commentPrefix="#", trim=True)
        ).parse_text("# comment\nid;name\n1; Ada \n2;Grace")
        table = CSVWTableConfig(
            url="people.csv",
            tableSchema=CSVWTableSchemaConfig(
                primaryKey="id",
                columns=[
                    CSVWColumnConfig(name="id", required=True),
                    CSVWColumnConfig(name="name", required=True),
                ],
            ),
        )
        source = SourceRecord.from_rows(rows, source_type="csv", source_name="people.csv")
        batch = CSVWMapper(CSVWMappingConfig(tables=[table])).map_source(source)

        self.assertEqual(rows[0]["name"], "Ada")
        self.assertTrue(batch.metadata["csvw_validation"]["valid"])

    def test_obda_rewrites_basic_triple_pattern(self) -> None:
        mapper = OBDAVirtualGraphMapper(
            [
                OBDAPredicateMapping(
                    predicate_iri="urn:kg:name",
                    table="people",
                    subject_template="urn:person:{id}",
                    object_column="name",
                )
            ]
        )
        plans = mapper.rewrite_select("SELECT ?s ?o WHERE { ?s <urn:kg:name> ?o . }")

        self.assertEqual(len(plans), 1)
        self.assertIn("FROM people", plans[0].sql)

    def test_event_payload_loader_mapper_trace_span_and_log_templates(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            event_path = Path(tmpdir) / "events.json"
            event_path.write_text(
                json.dumps(
                    {
                        "events": [
                            {"event_id": "e1", "event_type": "request", "service": "Recommendation API", "trace_id": "t1", "span_id": "s1"},
                            {"event_id": "e2", "event_type": "db", "service": "Recommendation API", "trace_id": "t1", "span_id": "s2", "parent_span_id": "s1", "caused_by": "e1"},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            log_path = Path(tmpdir) / "events.log"
            log_path.write_text(
                "2026-07-09 10:00:00 ERROR app INCIDENT-1042 failed with status 503\n"
                "2026-07-09 10:01:00 ERROR app INCIDENT-1043 failed with status 500\n",
                encoding="utf-8",
            )
            event_source = EventPayloadLoader().load(event_path)[0]
            event_batch = EventGraphMapper().map_source(event_source)
            log_source = LogLoader().load(log_path)[0]
            log_batch = LogEventMapper().map_source(log_source)

        self.assertTrue(any(rel.relation_type == "PARENT_OF_SPAN" for rel in event_batch.relations))
        self.assertTrue(any(rel.relation_type == "CAUSED" for rel in event_batch.relations))
        self.assertEqual(log_source.records[0]["template"], log_source.records[1]["template"])
        self.assertTrue(any(entity.entity_type == "LogTemplate" for entity in log_batch.entities))

    def test_advanced_graph_ml_and_retrieval_benchmark(self) -> None:
        graph = generate_graph_store_benchmark_fixture(node_count=12, edge_factor=2)
        split = GraphTrainValidationSplitter().split(graph)
        graph = Node2VecEmbeddingAdapter(dimensions=4).enrich(graph)
        graph = KnowledgeGraphEmbeddingAdapter(dimensions=4).enrich(graph)
        graph = LargeGraphAnalyticsBackend().pagerank(graph)
        rows = GraphRetrievalBenchmark().compare(graph, k=3)
        trainer_payload = GraphSageLinkPredictionTrainer().prepare(graph)

        self.assertTrue(split.validation_edges)
        self.assertIn("node2vec", graph.metadata)
        self.assertIn("knowledge_graph_embeddings", graph.metadata)
        self.assertEqual({row.strategy for row in rows}, {"text_only", "vector", "graphrag"})
        self.assertIn("split", trainer_payload)

    def test_neosemantics_and_kuzu_bulk_paths_use_expected_queries(self) -> None:
        rdf_graph = RDFGraph(
            triples=[RDFTriple(subject="urn:s", predicate="urn:p", object="value")]
        )
        driver = FakeNeo4jDriver()
        neo4j = Neo4jIngestionAdapter(
            Neo4jIngestionConfig(enable_neosemantics=True),
            driver=driver,
        )
        report = neo4j.write_rdf_graph(rdf_graph)
        graph = generate_graph_store_benchmark_fixture(node_count=3, edge_factor=1)
        connection = FakeKuzuConnection()
        kuzu = KuzuIngestionAdapter(KuzuIngestionConfig(database_path="memory"), connection=connection)
        with tempfile.TemporaryDirectory() as tmpdir:
            bulk_report = kuzu.bulk_import_property_graph(graph, tmpdir)

        self.assertEqual(report.graph_model, "rdf_graph")
        self.assertTrue(any("n10s.rdf.import.inline" in query for query, _ in driver.runs))
        self.assertTrue(any(query.startswith("COPY KGNode") for query, _ in connection.executions))
        self.assertTrue(bulk_report.metadata["bulk_import"])

    def test_gold_benchmark_fixtures_evaluate_spans_triples_and_relations(self) -> None:
        evidence_source = SourceRecord.from_text("INCIDENT-1042 affected Recommendation API.", source_name="memory://gold")
        evidence = EvidenceRef.build(evidence_source, text="INCIDENT-1042")
        batch = ConstructionBatch(
            entities=[
                EntityCandidate.build(entity_type="Incident", name="INCIDENT-1042", evidence=[evidence], extractor_name="unit", technique="gold"),
                EntityCandidate.build(entity_type="System", name="Recommendation API", evidence=[evidence], extractor_name="unit", technique="gold"),
                EntityCandidate.build(entity_type="Date", name="2026-07-09", evidence=[evidence], extractor_name="unit", technique="gold"),
            ],
            facts=[
                GraphFact.build(subject="urn:kg:incident:INCIDENT-1042", predicate="http://www.w3.org/1999/02/22-rdf-syntax-ns#type", object="urn:kg:schema:Incident", evidence=[evidence], extractor_name="unit", technique="gold"),
                GraphFact.build(subject="urn:kg:incident:INCIDENT-1042", predicate="urn:kg:schema:affectedSystem", object="urn:kg:system:recommendation-api", evidence=[evidence], extractor_name="unit", technique="gold"),
            ],
            relations=[
                RelationCandidate.build(source="entity_incident_1042", relation_type="AFFECTED", target="entity_recommendation_api", evidence=[evidence], extractor_name="unit", technique="gold"),
                RelationCandidate.build(source="entity_incident_1042", relation_type="INVESTIGATED_BY", target="entity_platform_team", evidence=[evidence], extractor_name="unit", technique="gold"),
            ],
        )
        evaluator = GoldBenchmarkEvaluator()
        ner_gold = GoldBenchmarkFixture.from_json_file("tests/fixtures/gold_standards/ner_spans.json")
        triple_gold = GoldBenchmarkFixture.from_json_file("tests/fixtures/gold_standards/gold_triples.json")
        relation_gold = GoldBenchmarkFixture.from_json_file("tests/fixtures/gold_standards/relation_sample.json")

        self.assertEqual(evaluator.evaluate_entities(batch, ner_gold).f1, 1.0)
        self.assertEqual(evaluator.evaluate_triples(batch, triple_gold).f1, 1.0)
        self.assertEqual(evaluator.evaluate_relations(batch, relation_gold).f1, 1.0)

    def test_registry_marks_next_stage_capabilities_as_implemented(self) -> None:
        registry = default_registry()
        for name in {
            "jsonld_loader",
            "jsonld_mapper",
            "obda_virtual_graph_mapper",
            "event_payload_loader",
            "event_graph_mapper",
            "graph_train_validation_splitter",
            "node2vec_embedding_adapter",
            "knowledge_graph_embedding_adapter",
            "large_graph_analytics_backend",
            "graph_retrieval_benchmark",
        }:
            self.assertEqual(registry.get(name).status, "implemented")


if __name__ == "__main__":
    unittest.main()
