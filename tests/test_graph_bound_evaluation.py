from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from kg_construction import load_sources
from kg_construction.construction.config import DataToKGConfig
from kg_construction.construction.workflow import DraftConstructionWorkflow
from kg_construction.domain.graph_objects import (
    PropertyEdge,
    PropertyGraph,
    PropertyNode,
)
from kg_construction.domain.models import SourceRecord
from kg_construction.evaluation import (
    AccuracyMetrics,
    GraphBoundIntendedUseEvaluator,
    Neo4jVectorWorkloadBackend,
    VersionedGraphEvaluationSuite,
    predicted_quality_mappings,
    property_graph_sha256,
)
from kg_construction.infrastructure.production.pii import PIIRedactionPolicy
from kg_construction.interfaces.cli.construct import main as construct_main
from kg_construction.research.workflow import (
    draft_workflow_for_source_type,
)
from kg_construction.unstructured.pipeline import UnstructuredPipeline


def _graph() -> PropertyGraph:
    source_ids = ["source-1"]
    return PropertyGraph(
        nodes=[
            PropertyNode(
                node_id="user",
                labels=["Entity", "User"],
                properties={"name": "User", "source_ids": source_ids},
            ),
            PropertyNode(
                node_id="middle",
                labels=["Entity", "Item"],
                properties={"name": "Middle", "source_ids": source_ids},
            ),
            PropertyNode(
                node_id="item-1",
                labels=["Entity", "Item"],
                properties={"name": "Item One", "source_ids": source_ids},
            ),
            PropertyNode(
                node_id="item-2",
                labels=["Entity", "Item"],
                properties={"name": "Item Two", "source_ids": source_ids},
            ),
        ],
        edges=[
            PropertyEdge(
                edge_id="likes",
                source_id="user",
                target_id="item-1",
                type="LIKES",
            ),
            PropertyEdge(
                edge_id="likes-middle",
                source_id="middle",
                target_id="item-2",
                type="LIKES",
            ),
            PropertyEdge(
                edge_id="likes-item-two",
                source_id="item-2",
                target_id="middle",
                type="LIKES",
            ),
            PropertyEdge(
                edge_id="knows-1",
                source_id="user",
                target_id="middle",
                type="KNOWS",
            ),
            PropertyEdge(
                edge_id="knows-2",
                source_id="middle",
                target_id="item-1",
                type="KNOWS",
            ),
        ],
    )


def _suite_payload() -> dict[str, object]:
    user = {"node_ids": ["user"]}
    item_one = {"node_ids": ["item-1"]}
    items = {"labels": ["Item"]}
    likes_path = {
        "start": user,
        "end": item_one,
        "relationship_types": ["LIKES"],
        "max_depth": 1,
    }
    ranking_queries = [
        {
            "source": source,
            "relationship_type": "LIKES",
            "candidates": items,
            "k": 3,
        }
        for source in (
            user,
            {"node_ids": ["middle"]},
            {"node_ids": ["item-2"]},
        )
    ]
    link_truths = ["item-1", "item-2", "middle"]
    return {
        "version": "kg-graph-evaluation-suite-v1",
        "construction_profile": "structured",
        "fixture_id": "unit-graph-v1",
        "quality": {
            "structured_formation": {
                "record_label": "User",
                "expected_record_count": 1,
                "expected_node_counts": {"User": 1, "Item": 3},
                "expected_relationship_counts": {"LIKES": 3, "KNOWS": 2},
                "required_properties": {
                    "User": ["name", "source_ids"],
                    "Item": ["name", "source_ids"],
                },
                "forbidden_properties": {
                    "User": ["raw_manager_id"],
                    "Item": ["raw_category"],
                },
                "identity_keys": {"User": ["name"], "Item": ["name"]},
            }
        },
        "sql_like": {
            "cases": [
                {
                    "case_id": "sql",
                    "query": {
                        "selector": user,
                        "projections": {"name": "$property.name"},
                        "aggregations": {"count": "count"},
                    },
                    "expected_rows": [{"name": "User"}],
                    "expected_aggregations": {"count": 1},
                }
            ],
            "max_p95_latency_ms": 1000,
        },
        "nosql": {
            "cases": [
                {
                    "case_id": "nosql",
                    "query": {
                        "start": user,
                        "relationship_types": ["LIKES"],
                    },
                    "expected_node_ids": ["item-1", "user"],
                    "expected_paths": [["user", "LIKES", "item-1"]],
                }
            ],
            "max_p95_latency_ms": 1000,
        },
        "graphrag": {
            "cases": [
                {
                    "case_id": "rag",
                    "query": {
                        "query_text": "Item One",
                        "candidates": items,
                        "k": 1,
                        "path_query": likes_path,
                        "answer_query": likes_path,
                    },
                    "context_relevance": {"item-1": 3},
                    "expected_paths": [["user", "LIKES", "item-1"]],
                    "expected_answer_node_ids": ["item-1"],
                    "expected_citation_ids": ["source-1"],
                    "requires_multi_hop": True,
                }
            ]
        },
        "vector": {
            "cases": [
                {
                    "case_id": "vector",
                    "query": {
                        "query_text": "Item One",
                        "candidates": items,
                        "allowed": items,
                        "k": 1,
                    },
                    "relevance": {"item-1": 2},
                }
            ],
            "max_p95_latency_ms": 1000,
            "max_peak_memory_bytes": 1_000_000,
        },
        "link_prediction": {
            "cases": [
                {
                    "case_id": f"link-{index}",
                    "query": ranking_queries[index],
                    "true_entity_id": link_truths[index],
                }
                for index in range(3)
            ]
        },
        "recommendation": {
            "cases": [
                {
                    "case_id": f"recommendation-{index}",
                    "query": ranking_queries[index],
                    "relevant_item_ids": [link_truths[index]],
                }
                for index in range(3)
            ],
            "catalog": items,
            "feature_properties": ["name"],
        },
        "reasoning": {
            "cases": [
                {
                    "case_id": "reasoning",
                    "query": {
                        "start": user,
                        "relationship_type_sequence": ["KNOWS", "KNOWS"],
                        "inferred_predicate": "CONNECTED_TO",
                    },
                    "expected_answers": ["item-1"],
                    "expected_inferred_facts": [
                        {
                            "subject": "user",
                            "predicate": "CONNECTED_TO",
                            "object": "item-1",
                        }
                    ],
                }
            ],
            "max_p95_runtime_ms": 1000,
        },
    }


def _suite() -> VersionedGraphEvaluationSuite:
    return VersionedGraphEvaluationSuite.seal(_suite_payload())


def test_graph_bound_suite_computes_predictions_and_binds_all_evidence() -> None:
    graph = _graph()
    report, execution = GraphBoundIntendedUseEvaluator().evaluate(graph, _suite())

    assert report.passed
    assert report.graph_bound
    assert report.graph_sha256 == property_graph_sha256(graph)
    assert report.fixture_sha256 == _suite().fixture_sha256
    assert not execution.certifiable_backend
    assert execution.training_graph_sha256 != execution.graph_sha256
    assert set(execution.held_out_edge_ids) == {
        "likes",
        "likes-middle",
        "likes-item-two",
    }
    assert set(execution.derived_popularity_strata.values()) == {
        "head",
        "torso",
        "tail",
    }
    # Every direct LIKES gold edge is absent from the ranking graph: all cases
    # receive the same deterministic no-direct-edge ranking rather than having
    # their own held-out label promoted to rank one.
    assert [
        case.ranked_entity_ids
        for case in execution.inputs.link_prediction.cases
    ] == [["item-1", "item-2", "middle"]] * 3
    assert [
        case.known_true_entity_ids
        for case in execution.inputs.link_prediction.cases
    ] == [["item-1"], ["item-2"], ["middle"]]
    assert report.formation_gold_executed
    assert report.formation_gold_passed
    assert report.formation_gold_metrics["identity_key_uniqueness"] == 1.0
    for family in report.families.values():
        assert family.workload_support == family.evaluated_cases
        assert len(family.query_sha256) == family.evaluated_cases
        assert len(family.observed_latency_ms) == family.evaluated_cases
        assert family.graph_sha256 == report.graph_sha256
        assert family.fixture_sha256 == report.fixture_sha256


def test_live_vector_backend_waits_for_the_complete_frozen_snapshot() -> None:
    class DelayedStore:
        config = SimpleNamespace(index_name="delayed-test-index")

        def __init__(self) -> None:
            self.query_count = 0
            self.upserted = ()

        def upsert(self, records: object) -> None:
            self.upserted = tuple(records)  # type: ignore[arg-type]

        def query(
            self,
            _vector: object,
            *,
            top_k: int,
            filters: object,
        ) -> list[SimpleNamespace]:
            del top_k, filters
            self.query_count += 1
            if self.query_count == 1:
                return []
            return [SimpleNamespace(vector_id="item-1")]

        def count(self, *, filters: object) -> int:
            del filters
            return len(self.upserted)

    class LockedEmbedding:
        @staticmethod
        def encode(texts: list[str]) -> list[list[float]]:
            return [[1.0, *([0.0] * 63)] for _ in texts]

    store = DelayedStore()
    backend = Neo4jVectorWorkloadBackend(
        store=store,
        embedding_model=LockedEmbedding(),
        model_id="locked-test-model",
        refresh_attempts=2,
        refresh_interval_seconds=0.0,
    )

    observation = backend.query(_graph(), _suite().vector.cases[0].query)

    assert observation.ranking == ["item-1"]
    assert store.query_count == 2
    assert len(store.upserted) == 3


def test_suite_rejects_predictions_and_tampered_fixture_hash() -> None:
    payload = _suite_payload()
    sql = payload["sql_like"]
    assert isinstance(sql, dict)
    cases = sql["cases"]
    assert isinstance(cases, list)
    first = cases[0]
    assert isinstance(first, dict)
    first["predicted_rows"] = [{"name": "User"}]
    with pytest.raises(ValidationError, match="predicted_rows"):
        VersionedGraphEvaluationSuite.seal(payload)

    payload = _suite_payload()
    sql = payload["sql_like"]
    assert isinstance(sql, dict)
    cases = sql["cases"]
    assert isinstance(cases, list)
    first = cases[0]
    assert isinstance(first, dict)
    query = first["query"]
    assert isinstance(query, dict)
    query["aggregations"] = {"different_name": "count"}
    with pytest.raises(ValidationError, match="aggregation names"):
        VersionedGraphEvaluationSuite.seal(payload)

    sealed = _suite().model_dump(mode="json")
    sealed["fixture_id"] = "tampered"
    with pytest.raises(ValidationError, match="fixture_sha256 mismatch"):
        VersionedGraphEvaluationSuite.model_validate(sealed)


def test_suite_rejects_caller_supplied_link_labels_and_missing_formation_gold() -> None:
    payload = _suite_payload()
    link_prediction = payload["link_prediction"]
    assert isinstance(link_prediction, dict)
    cases = link_prediction["cases"]
    assert isinstance(cases, list)
    first = cases[0]
    assert isinstance(first, dict)
    first["known_true_entity_ids"] = [first["true_entity_id"]]
    first["popularity_stratum"] = "head"
    with pytest.raises(ValidationError, match="known_true_entity_ids"):
        VersionedGraphEvaluationSuite.seal(payload)

    payload = _suite_payload()
    payload["quality"] = {}
    with pytest.raises(ValidationError, match="structured-formation gold"):
        VersionedGraphEvaluationSuite.seal(payload)


def test_structured_formation_failure_is_conjunctive() -> None:
    graph = _graph()
    graph.nodes[-1].properties["name"] = "Item One"

    report, execution = GraphBoundIntendedUseEvaluator().evaluate(graph, _suite())

    assert execution.formation_gold_executed
    assert not execution.formation_gold_passed
    assert not execution.formation_gold_assertions["identity_unique:Item"]
    assert not report.passed


def test_cli_draft_writes_outputs_even_when_release_gates_fail(
    tmp_path: Path,
) -> None:
    draft = DraftConstructionWorkflow().run(
        [SourceRecord.from_text("draft", source_name="memory://draft")]
    )
    output = tmp_path / "graph.json"
    with patch(
        "kg_construction.interfaces.cli.construct.construct_kg",
        return_value=draft,
    ):
        result = construct_main(
            [
                "--input",
                "examples/data_to_kg/sample.txt",
                "--source-type",
                "text",
                "--output",
                str(output),
            ]
        )

    assert result == 0
    assert output.exists()
    assert (tmp_path / "graph.quality.json").exists()
    assert (tmp_path / "graph.intended-use.json").exists()


def test_repository_sample_suites_are_sealed_and_dataset_specific() -> None:
    csv_suite = VersionedGraphEvaluationSuite.from_path(
        "configs/graph_evaluation_suite.sample_csv.yaml"
    )
    text_suite = VersionedGraphEvaluationSuite.from_path(
        "configs/graph_evaluation_suite.sample_text.yaml"
    )

    assert csv_suite.construction_profile == "structured"
    assert text_suite.construction_profile == "unstructured"
    assert csv_suite.fixture_sha256 != text_suite.fixture_sha256
    assert text_suite.quality.expected_batch.entities
    assert text_suite.quality.expected_batch.relations


def test_sample_csv_suite_executes_substantive_structured_formation_gold() -> None:
    suite = VersionedGraphEvaluationSuite.from_path(
        "configs/graph_evaluation_suite.sample_csv.yaml"
    )
    config = DataToKGConfig.from_yaml(
        "configs/data_to_kg_complete.example.yaml"
    )
    graph = draft_workflow_for_source_type("csv", config).run(
        load_sources("examples/data_to_kg/sample.csv", source_type="csv")
    ).property_graph

    report, execution = GraphBoundIntendedUseEvaluator().evaluate(graph, suite)

    assert report.passed
    assert execution.formation_gold_executed
    assert execution.formation_gold_passed
    assert execution.formation_gold_metrics["record_count_exact"] == 1.0
    assert execution.formation_gold_metrics["required_property_coverage"] == 1.0
    assert execution.formation_gold_metrics["identity_key_uniqueness"] == 1.0
    assert set(execution.held_out_edge_ids) == {
        "edge_024a4cb1dd36f4832383",
        "edge_94034278dcc3250c4d53",
        "edge_e1dd07dfae5157fcabd4",
    }


def test_sample_text_quality_gold_is_bound_to_the_actual_redacted_snapshot() -> None:
    suite = VersionedGraphEvaluationSuite.from_path(
        "configs/graph_evaluation_suite.sample_text.yaml"
    )
    original = load_sources(
        "examples/data_to_kg/sample.txt",
        source_type="text",
    )[0]
    redacted = PIIRedactionPolicy().redact_source_record(original)

    assert redacted.source_id == "source_7cbdc1a5851256d53232"
    assert redacted.source_id != original.source_id
    assert (
        redacted.metadata["_source_lineage"]["parent_document_version"]
        == original.source_id
    )
    gold_evidence = [
        evidence
        for item in (
            *suite.quality.expected_batch.entities,
            *suite.quality.expected_batch.relations,
        )
        for evidence in item.evidence
    ]
    assert {evidence.source_id for evidence in gold_evidence} == {
        redacted.source_id
    }
    assert {
        evidence.metadata["document_version"] for evidence in gold_evidence
    } == {redacted.document_version}

    pipeline = UnstructuredPipeline()
    batch = pipeline.extract(redacted)
    accuracy = AccuracyMetrics().evaluate(batch, suite.quality.expected_batch)
    predicted_coreference, predicted_links = predicted_quality_mappings(batch)

    assert accuracy["entity_f1"] == 1.0
    assert accuracy["span_f1"] == 1.0
    assert accuracy["relation_f1"] == 1.0
    assert set(predicted_coreference) == set(suite.quality.coreference_clusters)
    predicted_partition = {
        (left, right)
        for left in predicted_coreference
        for right in predicted_coreference
        if predicted_coreference[left] == predicted_coreference[right]
    }
    gold_partition = {
        (left, right)
        for left in suite.quality.coreference_clusters
        for right in suite.quality.coreference_clusters
        if (
            suite.quality.coreference_clusters[left]
            == suite.quality.coreference_clusters[right]
        )
    }
    assert predicted_partition == gold_partition
    assert predicted_links == suite.quality.entity_links


def test_sample_text_workloads_are_bound_only_to_the_mandatory_profile() -> None:
    suite = VersionedGraphEvaluationSuite.from_path(
        "configs/graph_evaluation_suite.sample_text.yaml"
    )
    original = load_sources(
        "examples/data_to_kg/sample.txt",
        source_type="text",
    )[0]
    redacted = PIIRedactionPolicy().redact_source_record(original)
    deterministic_batch = UnstructuredPipeline().extract(redacted)
    deterministic_ids = {
        entity.candidate_id for entity in deterministic_batch.entities
    }
    mandatory_workload_ids = {
        node_id
        for case in suite.link_prediction.cases
        for node_id in (
            *case.query.source.node_ids,
            *case.query.candidates.node_ids,
        )
    }

    assert suite.fixture_id == "sample-text-publication-v3"
    assert mandatory_workload_ids
    assert mandatory_workload_ids.isdisjoint(deterministic_ids)
