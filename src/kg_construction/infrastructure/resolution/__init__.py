"""Entity resolution and linking adapters."""

from __future__ import annotations

from kg_construction.infrastructure.resolution.common import reconcile_entity_references
from kg_construction.infrastructure.resolution.entity_resolution import (
    AliasTableResolver,
    AliasTableResolverConfig,
    DedupeActiveLearningResolver,
    DedupeResolverConfig,
    NormalizedStringResolver,
    NormalizedStringResolverConfig,
    RapidFuzzResolver,
    RapidFuzzResolverConfig,
    SplinkProbabilisticResolver,
    SplinkResolverConfig,
)
from kg_construction.infrastructure.resolution.exact import ExactKeyEntityResolver
from kg_construction.infrastructure.resolution.linking import (
    CrossEncoderModelProtocol,
    CrossEncoderResolver,
    CrossEncoderResolverConfig,
    DBpediaLinker,
    DBpediaLinkerConfig,
    EmbeddingModelProtocol,
    EmbeddingResolver,
    EmbeddingResolverConfig,
    OpenRefineReconciler,
    OpenRefineReconcilerConfig,
    WikidataLinker,
    WikidataLinkerConfig,
)
from kg_construction.infrastructure.resolution.stubs import (
    BlockingResolver,
    GraphNeighborhoodResolver,
    HumanReviewResolver,
)

FuzzyStringResolver = RapidFuzzResolver
ProbabilisticLinkageResolver = SplinkProbabilisticResolver

__all__ = [
    "AliasTableResolver",
    "AliasTableResolverConfig",
    "BlockingResolver",
    "CrossEncoderModelProtocol",
    "CrossEncoderResolver",
    "CrossEncoderResolverConfig",
    "DBpediaLinker",
    "DBpediaLinkerConfig",
    "DedupeActiveLearningResolver",
    "DedupeResolverConfig",
    "EmbeddingModelProtocol",
    "EmbeddingResolver",
    "EmbeddingResolverConfig",
    "ExactKeyEntityResolver",
    "FuzzyStringResolver",
    "GraphNeighborhoodResolver",
    "HumanReviewResolver",
    "NormalizedStringResolver",
    "NormalizedStringResolverConfig",
    "OpenRefineReconciler",
    "OpenRefineReconcilerConfig",
    "ProbabilisticLinkageResolver",
    "RapidFuzzResolver",
    "RapidFuzzResolverConfig",
    "reconcile_entity_references",
    "SplinkProbabilisticResolver",
    "SplinkResolverConfig",
    "WikidataLinker",
    "WikidataLinkerConfig",
]
