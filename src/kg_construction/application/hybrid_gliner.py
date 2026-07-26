"""Compatibility imports for :mod:`kg_construction.research.hybrid_gliner`."""

from kg_construction.research.hybrid_gliner import (
    ExtractionComparisonRow,
    HybridGlinerExtractionResult,
    HybridGlinerPipelineConfig,
    LabelReviewSettings,
    apply_manual_label_review,
    compare_entity_extraction_strategies,
    comparison_row,
    extract_entities_with_ollama_labels_and_gliner,
)

__all__ = [
    "ExtractionComparisonRow",
    "HybridGlinerExtractionResult",
    "HybridGlinerPipelineConfig",
    "LabelReviewSettings",
    "apply_manual_label_review",
    "compare_entity_extraction_strategies",
    "comparison_row",
    "extract_entities_with_ollama_labels_and_gliner",
]
