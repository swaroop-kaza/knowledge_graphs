"""Production-oriented Data-to-Knowledge Graph construction framework."""

from __future__ import annotations

from kg_construction.application.benchmark_runner import (
    TechniqueEvaluationConfig,
    TechniqueEvaluationReport,
)
from kg_construction.application.config import DataToKGConfig
from kg_construction.application.dataset_profiler import DatasetProfile
from kg_construction.application.public_api import (
    construct_kg,
    default_workflow_for_source_type,
    evaluate_techniques,
    load_sources,
    profile_dataset,
)
from kg_construction.application.publication_contract import (
    PublicationContractAssessment,
    PublicationContractRegistry,
    TrustedPublicationBinding,
    UntrustedPublicationContractError,
)
from kg_construction.application.registry import (
    SUPPORT_LEVEL_DEFINITIONS,
    SUPPORT_LEVEL_LABELS,
    PluginRegistry,
    SupportLevel,
    default_registry,
    research_registry,
)
from kg_construction.application.workflow import (
    ConstructionResult,
    ConstructionWorkflow,
    PublicationGateError,
    PublicationGateReport,
)
from kg_construction.domain.models import SourceRecord
from kg_construction.infrastructure.evaluation import (
    IntendedUseEvaluationInputs,
    IntendedUseEvaluationReport,
    IntendedUseMetricsEvaluator,
    KnowledgeGraphQualityEvaluator,
    KnowledgeGraphQualityReport,
    QualityEvaluationInputs,
    VersionedGraphEvaluationSuite,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "ConstructionResult",
    "ConstructionWorkflow",
    "DataToKGConfig",
    "DatasetProfile",
    "IntendedUseEvaluationInputs",
    "IntendedUseEvaluationReport",
    "IntendedUseMetricsEvaluator",
    "KnowledgeGraphQualityEvaluator",
    "KnowledgeGraphQualityReport",
    "PluginRegistry",
    "PublicationGateError",
    "PublicationGateReport",
    "PublicationContractAssessment",
    "PublicationContractRegistry",
    "SUPPORT_LEVEL_DEFINITIONS",
    "SUPPORT_LEVEL_LABELS",
    "SourceRecord",
    "SupportLevel",
    "TechniqueEvaluationConfig",
    "TechniqueEvaluationReport",
    "TrustedPublicationBinding",
    "UntrustedPublicationContractError",
    "VersionedGraphEvaluationSuite",
    "QualityEvaluationInputs",
    "construct_kg",
    "default_registry",
    "default_workflow_for_source_type",
    "evaluate_techniques",
    "load_sources",
    "profile_dataset",
    "research_registry",
]
