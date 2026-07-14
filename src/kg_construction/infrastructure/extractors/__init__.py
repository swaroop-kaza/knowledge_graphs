"""Entity and relation extractors."""

from __future__ import annotations

from kg_construction.infrastructure.extractors.gazetteer import GazetteerEntityExtractor
from kg_construction.infrastructure.extractors.gliner import GlinerConfig, GlinerEntityExtractor
from kg_construction.infrastructure.extractors.ollama_graph import OllamaGraphExtractor
from kg_construction.infrastructure.extractors.regex import RegexEntityExtractor
from kg_construction.infrastructure.extractors.relations import (
    DependencyPatternRelationExtractor,
    DependencyRelationPattern,
    GlinerOllamaRelationTypingExtractor,
    OpenIEConfig,
    OpenIERelationExtractor,
    RebelConfig,
    RebelRelationExtractor,
    RelationTypingConfig,
    TransformerRelationExtractor,
    default_dependency_relation_patterns,
)
from kg_construction.infrastructure.extractors.stubs import (
    CoreferenceAdapter,
    JointEntityRelationExtractor,
    PairwiseRelationClassifierAdapter,
    PosTaggingAdapter,
    SentenceSplitterAdapter,
    SpacyEntityRulerExtractor,
    SpacyNerExtractor,
    SupervisedRelationExtractor,
    TransformerNerExtractor,
)

__all__ = [
    "CoreferenceAdapter",
    "DependencyPatternRelationExtractor",
    "DependencyRelationPattern",
    "GazetteerEntityExtractor",
    "GlinerConfig",
    "GlinerEntityExtractor",
    "GlinerOllamaRelationTypingExtractor",
    "JointEntityRelationExtractor",
    "OllamaGraphExtractor",
    "OpenIEConfig",
    "OpenIERelationExtractor",
    "PairwiseRelationClassifierAdapter",
    "PosTaggingAdapter",
    "RebelConfig",
    "RebelRelationExtractor",
    "RegexEntityExtractor",
    "RelationTypingConfig",
    "SentenceSplitterAdapter",
    "SpacyEntityRulerExtractor",
    "SpacyNerExtractor",
    "SupervisedRelationExtractor",
    "TransformerRelationExtractor",
    "TransformerNerExtractor",
    "default_dependency_relation_patterns",
]
