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
from kg_construction.application.registry import (
    SUPPORT_LEVEL_DEFINITIONS,
    SUPPORT_LEVEL_LABELS,
    PluginRegistry,
    SupportLevel,
    default_registry,
)
from kg_construction.application.workflow import ConstructionResult, ConstructionWorkflow
from kg_construction.domain.models import SourceRecord

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "ConstructionResult",
    "ConstructionWorkflow",
    "DataToKGConfig",
    "DatasetProfile",
    "PluginRegistry",
    "SUPPORT_LEVEL_DEFINITIONS",
    "SUPPORT_LEVEL_LABELS",
    "SourceRecord",
    "SupportLevel",
    "TechniqueEvaluationConfig",
    "TechniqueEvaluationReport",
    "construct_kg",
    "default_registry",
    "default_workflow_for_source_type",
    "evaluate_techniques",
    "load_sources",
    "profile_dataset",
]
