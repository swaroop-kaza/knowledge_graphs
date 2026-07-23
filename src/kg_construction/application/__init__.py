"""Application services for Data-to-KG construction."""

from __future__ import annotations

from kg_construction.application.benchmark_runner import (
    BenchmarkHardGateSettings,
    BenchmarkRunMetadata,
    CandidateGraphResult,
    EvaluationSettings,
    GoldDatasetManifest,
    GoldEvaluationSummary,
    HardGateResult,
    TechniqueEvaluationConfig,
    TechniqueEvaluationReport,
    TechniqueEvaluationRunner,
)
from kg_construction.application.config import (
    DataToKGConfig,
    ErrorQuarantineSettings,
    HumanReviewQueueSettings,
    IncrementalSyncSettings,
    OllamaSettings,
    PIIRedactionSettings,
    PipelineConfig,
    ProductionSettings,
    RunManifestSettings,
    StructuredLoggingSettings,
    TechniqueConfig,
)
from kg_construction.application.dataset_profiler import DatasetProfile, DatasetProfiler
from kg_construction.application.graph_ml import (
    GraphMLWorkflow,
    GraphMLWorkflowConfig,
    graph_ml_workflow_config_from_dict,
    run_graph_ml_capabilities,
)
from kg_construction.application.hybrid_gliner import (
    ExtractionComparisonRow,
    HybridGlinerExtractionResult,
    HybridGlinerPipelineConfig,
    extract_entities_with_ollama_labels_and_gliner,
)
from kg_construction.application.modeling_policy import (
    CanonicalizationRule,
    ColumnPolicy,
    ModelingPolicy,
    PromotionThresholds,
    RelationshipDeclaration,
    default_modeling_policy,
)
from kg_construction.application.planner import TechniquePlanner
from kg_construction.application.production import ProductionServices, build_production_services
from kg_construction.application.recommendation_engine import (
    RecommendationDecision,
    RecommendationEngine,
    RecommendationSummary,
)
from kg_construction.application.registry import (
    SUPPORT_LEVEL_DEFINITIONS,
    SUPPORT_LEVEL_LABELS,
    PluginRegistry,
    SupportLevel,
    TechniqueDescriptor,
    TechniqueStatus,
    default_registry,
)
from kg_construction.application.scoring import (
    PenaltyWeights,
    ProductionThresholds,
    ScoringEngine,
    ScoringWeights,
    TechniqueScore,
)
from kg_construction.application.semantic_profiler import (
    ColumnFeatures,
    ColumnRoleAssignment,
    ForeignKeyLink,
    ModelingPlan,
    SemanticColumnProfiler,
)
from kg_construction.application.semantic_table_mapper import SemanticTableMapper
from kg_construction.application.technique_selector import ApplicabilityResult, TechniqueSelector
from kg_construction.application.workflow import ConstructionResult, ConstructionWorkflow

__all__ = [
    "ApplicabilityResult",
    "BenchmarkHardGateSettings",
    "BenchmarkRunMetadata",
    "CandidateGraphResult",
    "CanonicalizationRule",
    "ColumnFeatures",
    "ColumnPolicy",
    "ColumnRoleAssignment",
    "ConstructionResult",
    "ConstructionWorkflow",
    "DataToKGConfig",
    "DatasetProfile",
    "DatasetProfiler",
    "ErrorQuarantineSettings",
    "EvaluationSettings",
    "ExtractionComparisonRow",
    "ForeignKeyLink",
    "ModelingPlan",
    "ModelingPolicy",
    "PromotionThresholds",
    "RelationshipDeclaration",
    "SemanticColumnProfiler",
    "SemanticTableMapper",
    "default_modeling_policy",
    "GraphMLWorkflow",
    "GraphMLWorkflowConfig",
    "GoldDatasetManifest",
    "GoldEvaluationSummary",
    "HardGateResult",
    "HybridGlinerExtractionResult",
    "HybridGlinerPipelineConfig",
    "HumanReviewQueueSettings",
    "IncrementalSyncSettings",
    "OllamaSettings",
    "PIIRedactionSettings",
    "PenaltyWeights",
    "PipelineConfig",
    "PluginRegistry",
    "ProductionServices",
    "ProductionSettings",
    "ProductionThresholds",
    "RecommendationDecision",
    "RecommendationEngine",
    "RecommendationSummary",
    "RunManifestSettings",
    "SUPPORT_LEVEL_DEFINITIONS",
    "SUPPORT_LEVEL_LABELS",
    "ScoringEngine",
    "ScoringWeights",
    "StructuredLoggingSettings",
    "SupportLevel",
    "TechniqueConfig",
    "TechniqueDescriptor",
    "TechniqueEvaluationConfig",
    "TechniqueEvaluationReport",
    "TechniqueEvaluationRunner",
    "TechniquePlanner",
    "TechniqueScore",
    "TechniqueSelector",
    "TechniqueStatus",
    "build_production_services",
    "default_registry",
    "extract_entities_with_ollama_labels_and_gliner",
    "graph_ml_workflow_config_from_dict",
    "run_graph_ml_capabilities",
]
