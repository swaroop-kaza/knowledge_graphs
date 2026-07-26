"""Compatibility imports for canonical knowledge-graph quality gates.

New source code should import these contracts from
``kg_construction.evaluation.quality_gates``.
"""

from kg_construction.evaluation.quality_gates import (
    QUALITY_THRESHOLDS_VERSION,
    ConstructionReplayEvidence,
    KnowledgeGraphQualityEvaluator,
    KnowledgeGraphQualityReport,
    MetricFamilyReport,
    QualityEvaluationInputs,
    QualityGateResult,
    QualityThreshold,
    VersionedQualityThresholds,
)

__all__ = [
    "QUALITY_THRESHOLDS_VERSION",
    "ConstructionReplayEvidence",
    "KnowledgeGraphQualityEvaluator",
    "KnowledgeGraphQualityReport",
    "MetricFamilyReport",
    "QualityEvaluationInputs",
    "QualityGateResult",
    "QualityThreshold",
    "VersionedQualityThresholds",
]
