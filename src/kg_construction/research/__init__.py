"""Comparative research tools for knowledge-graph construction.

Production construction and graph-quality evaluation live outside this
package. This namespace owns technique selection, weighted scoring,
recommendations, comparative benchmarks, and experimental enrichment
workflows. Exports are explicit and lazy so importing one research submodule
does not initialize the certification registry or every research dependency.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORT_GROUPS: dict[str, tuple[str, ...]] = {
    "api": ("evaluate_techniques",),
    "benchmark_runner": (
        "BenchmarkHardGateSettings",
        "BenchmarkRunMetadata",
        "CandidateGraphResult",
        "EvaluationSettings",
        "GoldDatasetManifest",
        "GoldEvaluationSummary",
        "HardGateResult",
        "TechniqueEvaluationConfig",
        "TechniqueEvaluationReport",
        "TechniqueEvaluationRunner",
    ),
    "graph_ml": (
        "GraphMLWorkflow",
        "GraphMLWorkflowConfig",
        "graph_ml_workflow_config_from_dict",
        "run_graph_ml_capabilities",
    ),
    "graph_store_benchmark": ("generate_graph_store_benchmark_fixture",),
    "hybrid_gliner": (
        "ExtractionComparisonRow",
        "HybridGlinerExtractionResult",
        "HybridGlinerPipelineConfig",
        "LabelReviewSettings",
        "apply_manual_label_review",
        "compare_entity_extraction_strategies",
        "comparison_row",
        "extract_entities_with_ollama_labels_and_gliner",
    ),
    "planner": ("TechniquePlanner",),
    "recommendation_engine": (
        "RecommendationDecision",
        "RecommendationEngine",
        "RecommendationSummary",
    ),
    "scoring": (
        "PenaltyWeights",
        "ProductionThresholds",
        "ResearchScreeningThresholds",
        "ScoringEngine",
        "ScoringWeights",
        "TechniqueScore",
        "bounded_metric",
        "inverse_unbounded",
    ),
    "simple_benchmark": (
        "BenchmarkResult",
        "BenchmarkRunner",
        "TechniqueRunner",
    ),
    "technique_selector": (
        "ApplicabilityResult",
        "ApplicabilityWeights",
        "TechniqueSelector",
    ),
}

_EXPORTS = {
    name: (f"kg_construction.research.{module_name}", name)
    for module_name, names in _EXPORT_GROUPS.items()
    for name in names
}


def __getattr__(name: str) -> Any:
    """Resolve one explicit research export from its owning module."""

    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = target
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value


__all__ = sorted(_EXPORTS)
