"""Product evaluation API.

The implementations live in this package. Imports are explicit and lazy so a
formation-metric consumer does not initialize graph workloads, Neo4j adapters,
or unstructured model evaluation. Weighted technique comparison remains in
``kg_construction.research``.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORT_GROUPS: dict[str, tuple[str, ...]] = {
    "accuracy_metrics": (
        "AccuracyMetrics",
        "entity_key",
        "relation_key",
    ),
    "claim_evaluations": (
        "ClaimEvaluationSuite",
        "EvalScore",
        "HardCoverageReport",
        "HardCoverageThresholds",
    ),
    "coverage_metrics": ("CoverageMetrics",),
    "entity_resolution_metrics": (
        "EntityResolutionMetrics",
        "evaluation_record_id",
        "evaluation_record_id_map",
        "predicted_merge_pairs",
    ),
    "extraction_coverage": (
        "CoverageEvaluator",
        "CoverageReport",
        "IntentJudge",
    ),
    "gold_standards": (
        "GoldBenchmarkEvaluator",
        "GoldBenchmarkFixture",
        "GoldBenchmarkScore",
        "GoldEntityResolutionFixture",
        "GoldEntityResolutionRecord",
        "GoldEntitySpan",
        "GoldRelation",
        "GoldRetrievalCase",
        "GoldRetrievalContext",
        "GoldRetrievalEdge",
        "GoldRetrievalFixture",
        "GoldTriple",
    ),
    "graph_bound_suite": (
        "GRAPH_EVALUATION_SUITE_VERSION",
        "BoundGraphRAGCase",
        "BoundGraphRAGSuite",
        "BoundLinkPredictionCase",
        "BoundLinkPredictionSuite",
        "BoundNoSQLCase",
        "BoundNoSQLSuite",
        "BoundReasoningCase",
        "BoundReasoningSuite",
        "BoundRecommendationCase",
        "BoundRecommendationSuite",
        "BoundSQLLikeCase",
        "BoundSQLLikeSuite",
        "BoundVectorCase",
        "BoundVectorSuite",
        "DeterministicVectorWorkloadBackend",
        "EvidenceGroundedPathAnswerBackend",
        "GraphAnswerBackend",
        "GraphAnswerObservation",
        "GraphBoundExecution",
        "GraphBoundIntendedUseEvaluator",
        "GraphBoundQualityGold",
        "GraphNodeSelector",
        "GraphTraversalQuery",
        "GraphWorkloadBackend",
        "GraphWorkloadRunner",
        "InMemoryGraphWorkloadBackend",
        "LexicalRetrievalQuery",
        "NeighborRankingQuery",
        "Neo4jVectorWorkloadBackend",
        "PathQuery",
        "ReasoningPathQuery",
        "SQLNodeQuery",
        "StructuredFormationGold",
        "VectorSearchQuery",
        "VectorWorkloadBackend",
        "VectorWorkloadObservation",
        "VersionedGraphEvaluationSuite",
        "evaluation_fixture_sha256",
        "predicted_quality_mappings",
        "property_graph_sha256",
    ),
    "intended_use_metrics": (
        "GraphRAGCase",
        "GraphRAGEvaluationInput",
        "IntendedUseEvaluationInputs",
        "IntendedUseEvaluationReport",
        "IntendedUseMetricReport",
        "IntendedUseMetricsEvaluator",
        "LinkPredictionCase",
        "LinkPredictionEvaluationInput",
        "MetricThresholdResult",
        "NoSQLEvaluationInput",
        "NoSQLTraversalCase",
        "PopularityStratumReport",
        "ReasoningCase",
        "ReasoningEvaluationInput",
        "ReasoningFact",
        "RecommendationCase",
        "RecommendationEvaluationInput",
        "SQLLikeEvaluationInput",
        "SQLLikeQueryCase",
        "VectorIndexCase",
        "VectorIndexEvaluationInput",
    ),
    "metrics": (
        "ConstructionMetricsEvaluator",
        "precision_recall_f1",
    ),
    "modelling_correctness": (
        "ModellingCase",
        "evaluate_structured_case",
        "evaluate_unstructured_case",
    ),
    "neo4j_workload_backend": (
        "Neo4jGraphWorkloadBackend",
        "Neo4jGraphWorkloadConfig",
    ),
    "operational_metrics": (
        "OperationalMetrics",
        "OperationalMetricsRecorder",
        "RetryObservation",
        "RuntimeReport",
    ),
    "quality_gates": (
        "QUALITY_THRESHOLDS_VERSION",
        "ConstructionReplayEvidence",
        "KnowledgeGraphQualityEvaluator",
        "KnowledgeGraphQualityReport",
        "MetricFamilyReport",
        "QualityEvaluationInputs",
        "QualityGateResult",
        "QualityThreshold",
        "VersionedQualityThresholds",
    ),
    "retrieval_metrics": ("RetrievalMetrics",),
    "schema_quality": (
        "SchemaQualityEvaluator",
        "SchemaQualityGate",
        "SchemaQualityGateSettings",
        "SchemaQualityReport",
    ),
    "statistical_confidence": (
        "bootstrap_mean_interval",
        "wilson_score_interval",
    ),
    "validation_metrics": (
        "ValidationMetrics",
        "validation_failure_metrics",
    ),
}

_EXPORTS = {
    name: (f"kg_construction.evaluation.{module_name}", name)
    for module_name, names in _EXPORT_GROUPS.items()
    for name in names
}


def __getattr__(name: str) -> Any:
    """Resolve one explicit evaluation export from its owning module."""

    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = target
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value


__all__ = sorted(_EXPORTS)
