"""Compatibility exports for the canonical intended-use evaluation API.

New code should import these names from
``kg_construction.evaluation.intended_use_metrics``.
"""

from __future__ import annotations

from kg_construction.evaluation.intended_use_metrics import (
    GraphRAGCase,
    GraphRAGEvaluationInput,
    IntendedUseEvaluationInputs,
    IntendedUseEvaluationReport,
    IntendedUseMetricReport,
    IntendedUseMetricsEvaluator,
    LinkPredictionCase,
    LinkPredictionEvaluationInput,
    MetricThresholdResult,
    NoSQLEvaluationInput,
    NoSQLTraversalCase,
    PopularityStratumReport,
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

__all__ = [
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
]
