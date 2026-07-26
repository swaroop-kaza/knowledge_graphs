"""Evidence-grounded data-to-knowledge-graph construction.

The root package exposes the construction and evaluation surface only. Technique
comparison is deliberately namespaced under :mod:`kg_construction.research`, and
release authorization is namespaced under :mod:`kg_construction.certification`.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

from kg_construction.construction.api import (
    construct_kg,
    convert,
    default_workflow_for_source_type,
    load_sources,
    profile_dataset,
)
from kg_construction.construction.config import DataToKGConfig
from kg_construction.construction.profiling import DatasetProfile
from kg_construction.construction.workflow import (
    ConstructionResult,
    ConstructionWorkflow,
    PublicationGateError,
    PublicationGateReport,
)
from kg_construction.domain.models import SourceRecord
from kg_construction.evaluation import (
    IntendedUseEvaluationInputs,
    IntendedUseEvaluationReport,
    IntendedUseMetricsEvaluator,
    KnowledgeGraphQualityEvaluator,
    KnowledgeGraphQualityReport,
    QualityEvaluationInputs,
    VersionedGraphEvaluationSuite,
)

__version__ = "0.1.0"

# One-cycle lazy compatibility for names that used to make a normal package
# import initialize the entire research and certification stacks.
_LEGACY_EXPORTS: dict[str, tuple[str, str]] = {
    "PluginRegistry": ("kg_construction.certification.registry", "PluginRegistry"),
    "PublicationContractAssessment": (
        "kg_construction.certification.publication_contract",
        "PublicationContractAssessment",
    ),
    "PublicationContractRegistry": (
        "kg_construction.certification.publication_contract",
        "PublicationContractRegistry",
    ),
    "SUPPORT_LEVEL_DEFINITIONS": (
        "kg_construction.certification.registry",
        "SUPPORT_LEVEL_DEFINITIONS",
    ),
    "SUPPORT_LEVEL_LABELS": (
        "kg_construction.certification.registry",
        "SUPPORT_LEVEL_LABELS",
    ),
    "SupportLevel": ("kg_construction.certification.registry", "SupportLevel"),
    "TechniqueEvaluationConfig": (
        "kg_construction.research",
        "TechniqueEvaluationConfig",
    ),
    "TechniqueEvaluationReport": (
        "kg_construction.research",
        "TechniqueEvaluationReport",
    ),
    "TrustedPublicationBinding": (
        "kg_construction.certification.publication_contract",
        "TrustedPublicationBinding",
    ),
    "UntrustedPublicationContractError": (
        "kg_construction.certification.publication_contract",
        "UntrustedPublicationContractError",
    ),
    "default_registry": (
        "kg_construction.certification.registry",
        "default_registry",
    ),
    "evaluate_techniques": ("kg_construction.research", "evaluate_techniques"),
}


def __getattr__(name: str) -> Any:
    """Resolve legacy cross-boundary exports lazily."""

    target = _LEGACY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = target
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value


__all__ = [
    "ConstructionResult",
    "ConstructionWorkflow",
    "DataToKGConfig",
    "DatasetProfile",
    "IntendedUseEvaluationInputs",
    "IntendedUseEvaluationReport",
    "IntendedUseMetricsEvaluator",
    "KnowledgeGraphQualityEvaluator",
    "KnowledgeGraphQualityReport",
    "PublicationGateError",
    "PublicationGateReport",
    "QualityEvaluationInputs",
    "SourceRecord",
    "VersionedGraphEvaluationSuite",
    "__version__",
    "construct_kg",
    "convert",
    "default_workflow_for_source_type",
    "load_sources",
    "profile_dataset",
]
