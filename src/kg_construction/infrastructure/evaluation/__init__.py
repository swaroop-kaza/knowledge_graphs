"""Evaluation modules."""

from __future__ import annotations

from kg_construction.infrastructure.evaluation.accuracy_metrics import AccuracyMetrics
from kg_construction.infrastructure.evaluation.benchmark import BenchmarkResult, BenchmarkRunner
from kg_construction.infrastructure.evaluation.coverage_metrics import CoverageMetrics
from kg_construction.infrastructure.evaluation.entity_resolution_metrics import (
    EntityResolutionMetrics,
)
from kg_construction.infrastructure.evaluation.gold_standards import (
    GoldBenchmarkEvaluator,
    GoldBenchmarkFixture,
    GoldBenchmarkScore,
    GoldEntityResolutionFixture,
    GoldEntityResolutionRecord,
    GoldEntitySpan,
    GoldRelation,
    GoldRetrievalCase,
    GoldRetrievalContext,
    GoldRetrievalEdge,
    GoldRetrievalFixture,
    GoldTriple,
)
from kg_construction.infrastructure.evaluation.metrics import (
    ConstructionMetricsEvaluator,
    precision_recall_f1,
)
from kg_construction.infrastructure.evaluation.operational_metrics import (
    OperationalMetrics,
    OperationalMetricsRecorder,
    RuntimeReport,
)
from kg_construction.infrastructure.evaluation.retrieval_metrics import RetrievalMetrics
from kg_construction.infrastructure.evaluation.statistical_confidence import (
    bootstrap_mean_interval,
    wilson_score_interval,
)
from kg_construction.infrastructure.evaluation.validation_metrics import ValidationMetrics

__all__ = [
    "AccuracyMetrics",
    "BenchmarkResult",
    "BenchmarkRunner",
    "ConstructionMetricsEvaluator",
    "CoverageMetrics",
    "EntityResolutionMetrics",
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
    "OperationalMetrics",
    "OperationalMetricsRecorder",
    "RetrievalMetrics",
    "RuntimeReport",
    "ValidationMetrics",
    "bootstrap_mean_interval",
    "precision_recall_f1",
    "wilson_score_interval",
]
