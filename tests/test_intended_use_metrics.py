from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from kg_construction.evaluation import (
    GraphRAGCase,
    GraphRAGEvaluationInput,
    IntendedUseEvaluationInputs,
    IntendedUseMetricReport,
    IntendedUseMetricsEvaluator,
    LinkPredictionCase,
    LinkPredictionEvaluationInput,
    NoSQLEvaluationInput,
    NoSQLTraversalCase,
    ReasoningCase,
    ReasoningEvaluationInput,
    ReasoningFact,
    RecommendationCase,
    RecommendationEvaluationInput,
    SQLLikeEvaluationInput,
    SQLLikeQueryCase,
    VectorIndexCase,
    VectorIndexEvaluationInput,
)


def _complete_inputs() -> IntendedUseEvaluationInputs:
    fact = ReasoningFact(subject="alice", predicate="works_for", object="acme")
    return IntendedUseEvaluationInputs(
        sql_like=SQLLikeEvaluationInput(
            cases=[
                SQLLikeQueryCase(
                    case_id="sql-1",
                    predicted_rows=[{"department": "Engineering", "employees": 2}],
                    expected_rows=[{"employees": 2, "department": "Engineering"}],
                    predicted_aggregations={"employee_count": 2},
                    expected_aggregations={"employee_count": 2},
                    latency_ms=4.0,
                )
            ],
            max_p95_latency_ms=10.0,
        ),
        nosql=NoSQLEvaluationInput(
            cases=[
                NoSQLTraversalCase(
                    case_id="traversal-1",
                    visited_node_ids=["alice", "acme"],
                    expected_node_ids=["acme", "alice"],
                    returned_paths=[["alice", "works_for", "acme"]],
                    expected_paths=[["alice", "works_for", "acme"]],
                    latency_ms=3.0,
                )
            ],
            max_p95_latency_ms=10.0,
        ),
        graphrag=GraphRAGEvaluationInput(
            cases=[
                GraphRAGCase(
                    case_id="rag-1",
                    retrieved_context_ids=["ctx-1"],
                    context_relevance={"ctx-1": 3.0},
                    retrieved_paths=[["alice", "works_for", "acme"]],
                    expected_paths=[["alice", "works_for", "acme"]],
                    requires_multi_hop=True,
                    multi_hop_answer_correct=True,
                    answer_faithfulness=1.0,
                    predicted_citation_ids=["source-1"],
                    expected_citation_ids=["source-1"],
                )
            ],
            k=1,
        ),
        vector=VectorIndexEvaluationInput(
            cases=[
                VectorIndexCase(
                    case_id="vector-1",
                    retrieved_ids=["ctx-1"],
                    relevance={"ctx-1": 2.0},
                    filter_allowed_ids=["ctx-1"],
                    latency_ms=2.0,
                )
            ],
            eligible_item_count=1,
            indexed_item_count=1,
            peak_memory_bytes=1024,
            k=1,
            max_p95_latency_ms=10.0,
            max_peak_memory_bytes=2048,
        ),
        link_prediction=LinkPredictionEvaluationInput(
            cases=[
                LinkPredictionCase(
                    case_id="lp-head",
                    ranked_entity_ids=["also-true", "head-answer"],
                    true_entity_id="head-answer",
                    known_true_entity_ids=["also-true", "head-answer"],
                    popularity_stratum="head",
                ),
                LinkPredictionCase(
                    case_id="lp-torso",
                    ranked_entity_ids=["torso-answer"],
                    true_entity_id="torso-answer",
                    known_true_entity_ids=["torso-answer"],
                    popularity_stratum="torso",
                ),
                LinkPredictionCase(
                    case_id="lp-tail",
                    ranked_entity_ids=["tail-answer"],
                    true_entity_id="tail-answer",
                    known_true_entity_ids=["tail-answer"],
                    popularity_stratum="tail",
                ),
            ]
        ),
        recommendation=RecommendationEvaluationInput(
            cases=[
                RecommendationCase(
                    case_id="rec-1",
                    user_id="user-1",
                    recommended_item_ids=["item-1", "item-2"],
                    relevant_item_ids=["item-1", "item-2"],
                )
            ],
            catalog_item_ids=["item-1", "item-2"],
            item_popularity_probability={"item-1": 0.5, "item-2": 0.25},
            item_features={"item-1": ["graph"], "item-2": ["database"]},
            k=2,
        ),
        reasoning=ReasoningEvaluationInput(
            cases=[
                ReasoningCase(
                    case_id="reasoning-1",
                    predicted_answers=["acme"],
                    expected_answers=["acme"],
                    predicted_inferred_facts=[fact],
                    expected_inferred_facts=[fact],
                    runtime_ms=5.0,
                )
            ],
            max_p95_runtime_ms=10.0,
        ),
    )


def test_all_intended_use_families_execute_with_complete_judgments() -> None:
    report = IntendedUseMetricsEvaluator().evaluate(_complete_inputs())

    assert report.passed
    assert report.missing_uses == []
    assert len(report.executed_uses) == 7
    assert all(family.executed and family.passed for family in report.families.values())
    assert report.families["sql_like"].metrics["aggregation_accuracy"] == 1.0
    assert report.families["nosql"].metrics["path_f1"] == 1.0
    assert report.families["graphrag"].metrics["answer_faithfulness"] == 1.0
    assert report.families["vector"].metrics["index_coverage"] == 1.0
    assert report.families["link_prediction"].popularity_strata["tail"].mrr == 1.0
    assert report.families["recommendation"].metrics["novelty"] == 1.5
    assert report.families["recommendation"].metrics["intra_list_diversity"] == 1.0
    assert report.families["reasoning"].metrics["inferred_fact_f1"] == 1.0


def test_absent_judgments_are_not_executed_and_never_pass() -> None:
    report = IntendedUseMetricsEvaluator().evaluate()

    assert not report.passed
    assert set(report.missing_uses) == {
        "sql_like",
        "nosql",
        "graphrag",
        "vector",
        "link_prediction",
        "recommendation",
        "reasoning",
    }
    for family in report.families.values():
        assert not family.executed
        assert not family.passed
        assert family.evaluated_cases == 0
        assert family.metrics == {}
        assert family.thresholds == []
        assert "requires" in family.reason


def test_incomplete_family_judgments_return_explicit_non_execution() -> None:
    evaluator = IntendedUseMetricsEvaluator()
    sql = evaluator.evaluate_sql_like(
        SQLLikeEvaluationInput(
            cases=[
                SQLLikeQueryCase(
                    case_id="no-aggregation-gold",
                    predicted_rows=[],
                    expected_rows=[],
                    latency_ms=1.0,
                )
            ],
            max_p95_latency_ms=10.0,
        )
    )
    graph_rag = evaluator.evaluate_graphrag(
        GraphRAGEvaluationInput(
            cases=[
                GraphRAGCase(
                    case_id="no-multihop",
                    retrieved_context_ids=["context"],
                    context_relevance={"context": 1.0},
                    retrieved_paths=[["a", "p", "b"]],
                    expected_paths=[["a", "p", "b"]],
                    requires_multi_hop=False,
                    answer_faithfulness=1.0,
                    predicted_citation_ids=["source"],
                    expected_citation_ids=["source"],
                )
            ]
        )
    )
    link_prediction = evaluator.evaluate_link_prediction(
        LinkPredictionEvaluationInput(
            cases=[
                LinkPredictionCase(
                    case_id="head-only",
                    ranked_entity_ids=["answer"],
                    true_entity_id="answer",
                    known_true_entity_ids=["answer"],
                    popularity_stratum="head",
                )
            ]
        )
    )

    assert not sql.executed and not sql.passed and sql.metrics == {}
    assert not graph_rag.executed and not graph_rag.passed
    assert "multi-hop" in graph_rag.reason
    assert not link_prediction.executed and not link_prediction.passed
    assert "tail" in link_prediction.reason


def test_metrics_use_filtered_ranking_and_fail_real_thresholds() -> None:
    evaluator = IntendedUseMetricsEvaluator()
    report = evaluator.evaluate_link_prediction(
        LinkPredictionEvaluationInput(
            cases=[
                LinkPredictionCase(
                    case_id="head",
                    ranked_entity_ids=["known-other", "head-answer"],
                    true_entity_id="head-answer",
                    known_true_entity_ids=["head-answer", "known-other"],
                    popularity_stratum="head",
                ),
                LinkPredictionCase(
                    case_id="torso",
                    ranked_entity_ids=["wrong", "torso-answer"],
                    true_entity_id="torso-answer",
                    known_true_entity_ids=["torso-answer"],
                    popularity_stratum="torso",
                ),
                LinkPredictionCase(
                    case_id="tail",
                    ranked_entity_ids=["wrong"],
                    true_entity_id="tail-answer",
                    known_true_entity_ids=["tail-answer"],
                    popularity_stratum="tail",
                ),
            ],
            min_filtered_mrr=0.6,
            min_filtered_hits_at_k={1: 0.5, 3: 0.8, 10: 0.8},
        )
    )

    assert report.executed
    assert not report.passed
    assert report.metrics["filtered_mrr"] == 0.5
    assert report.metrics["filtered_hits_at_1"] == pytest.approx(1 / 3)
    assert report.metrics["filtered_hits_at_3"] == pytest.approx(2 / 3)
    assert report.popularity_strata["head"].mrr == 1.0
    assert report.popularity_strata["torso"].mrr == 0.5
    assert report.popularity_strata["tail"].mrr == 0.0


def test_ranking_filter_resource_and_latency_metrics_are_computed() -> None:
    evaluator = IntendedUseMetricsEvaluator()
    graph_rag = evaluator.evaluate_graphrag(
        GraphRAGEvaluationInput(
            cases=[
                GraphRAGCase(
                    case_id="ranked",
                    retrieved_context_ids=["irrelevant", "relevant"],
                    context_relevance={"relevant": 1.0},
                    retrieved_paths=[["wrong"]],
                    expected_paths=[["a", "p", "b"]],
                    requires_multi_hop=True,
                    multi_hop_answer_correct=False,
                    answer_faithfulness=0.25,
                    predicted_citation_ids=["wrong-source"],
                    expected_citation_ids=["source"],
                )
            ],
            k=2,
        )
    )
    vector = evaluator.evaluate_vector(
        VectorIndexEvaluationInput(
            cases=[
                VectorIndexCase(
                    case_id="filtered",
                    retrieved_ids=["relevant", "filter-violation"],
                    relevance={"relevant": 1.0},
                    filter_allowed_ids=["relevant"],
                    latency_ms=20.0,
                )
            ],
            eligible_item_count=4,
            indexed_item_count=2,
            peak_memory_bytes=4096,
            k=2,
            max_p95_latency_ms=10.0,
            max_peak_memory_bytes=2048,
        )
    )

    assert graph_rag.executed and not graph_rag.passed
    assert graph_rag.metrics["context_recall_at_k"] == 1.0
    assert graph_rag.metrics["mrr"] == 0.5
    assert graph_rag.metrics["ndcg_at_k"] == pytest.approx(1 / math.log2(3))
    assert graph_rag.metrics["multi_hop_success_rate"] == 0.0
    assert graph_rag.metrics["citation_f1"] == 0.0

    assert vector.executed and not vector.passed
    assert vector.metrics["index_coverage"] == 0.5
    assert vector.metrics["filter_correctness"] == 0.5
    assert vector.metrics["latency_p95_ms"] == 20.0
    assert vector.metrics["peak_memory_bytes"] == 4096.0


def test_query_traversal_recommendation_and_reasoning_metrics_penalize_errors() -> None:
    evaluator = IntendedUseMetricsEvaluator()
    sql = evaluator.evaluate_sql_like(
        SQLLikeEvaluationInput(
            cases=[
                SQLLikeQueryCase(
                    case_id="sql-errors",
                    predicted_rows=[{"id": 1}, {"id": 3}],
                    expected_rows=[{"id": 1}, {"id": 2}],
                    predicted_aggregations={"count": 9},
                    expected_aggregations={"count": 10},
                    latency_ms=20.0,
                )
            ],
            max_p95_latency_ms=10.0,
        )
    )
    nosql = evaluator.evaluate_nosql(
        NoSQLEvaluationInput(
            cases=[
                NoSQLTraversalCase(
                    case_id="traversal-errors",
                    visited_node_ids=["a", "x"],
                    expected_node_ids=["a", "b"],
                    returned_paths=[["a", "wrong", "x"]],
                    expected_paths=[["a", "edge", "b"]],
                    latency_ms=20.0,
                )
            ],
            max_p95_latency_ms=10.0,
        )
    )
    recommendation = evaluator.evaluate_recommendation(
        RecommendationEvaluationInput(
            cases=[
                RecommendationCase(
                    case_id="recommendation-errors",
                    user_id="user",
                    recommended_item_ids=["a", "c"],
                    relevant_item_ids=["a", "b"],
                )
            ],
            catalog_item_ids=["a", "b", "c"],
            item_popularity_probability={"a": 0.5, "b": 0.25, "c": 0.25},
            item_features={"a": ["same"], "b": ["other"], "c": ["same"]},
            k=2,
        )
    )
    gold_fact = ReasoningFact(subject="a", predicate="implies", object="b")
    reasoning = evaluator.evaluate_reasoning(
        ReasoningEvaluationInput(
            cases=[
                ReasoningCase(
                    case_id="reasoning-errors",
                    predicted_answers=["wrong"],
                    expected_answers=["b"],
                    predicted_inferred_facts=[],
                    expected_inferred_facts=[gold_fact],
                    runtime_ms=20.0,
                )
            ],
            max_p95_runtime_ms=10.0,
        )
    )

    assert sql.executed and not sql.passed
    assert sql.metrics["row_precision"] == 0.5
    assert sql.metrics["row_recall"] == 0.5
    assert sql.metrics["aggregation_accuracy"] == 0.0
    assert sql.metrics["latency_p95_ms"] == 20.0

    assert nosql.executed and not nosql.passed
    assert nosql.metrics["traversal_precision"] == 0.5
    assert nosql.metrics["traversal_coverage"] == 0.5
    assert nosql.metrics["path_f1"] == 0.0

    assert recommendation.executed and not recommendation.passed
    assert recommendation.metrics["precision_at_k"] == 0.5
    assert recommendation.metrics["recall_at_k"] == 0.5
    assert recommendation.metrics["catalog_coverage"] == pytest.approx(2 / 3)
    assert recommendation.metrics["novelty"] == 1.5
    assert recommendation.metrics["intra_list_diversity"] == 0.0

    assert reasoning.executed and not reasoning.passed
    assert reasoning.metrics["competency_question_accuracy"] == 0.0
    assert reasoning.metrics["inferred_fact_f1"] == 0.0
    assert reasoning.metrics["runtime_p95_ms"] == 20.0


def test_unexecuted_report_contract_rejects_false_success() -> None:
    with pytest.raises(ValidationError, match="unexecuted benchmark cannot pass"):
        IntendedUseMetricReport(
            intended_use="vector",
            executed=False,
            passed=True,
            reason="missing",
        )
