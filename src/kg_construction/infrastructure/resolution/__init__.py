"""Entity resolution and linking adapters."""

from __future__ import annotations

from kg_construction.infrastructure.resolution.canonicalization import (
    EntityLinker,
    LinkResult,
    UnstructuredCanonicalizer,
)
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
from kg_construction.infrastructure.resolution.mandatory_ensemble import (
    ExternalIdentityDecision,
    ExternalIdentityPromotionPolicy,
    GovernedExternalIdentityPromoter,
    MandatoryScopedResolutionEnsemble,
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
    "EntityLinker",
    "LinkResult",
    "UnstructuredCanonicalizer",
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
    "ExternalIdentityDecision",
    "ExternalIdentityPromotionPolicy",
    "ExactKeyEntityResolver",
    "FuzzyStringResolver",
    "GraphNeighborhoodResolver",
    "GovernedExternalIdentityPromoter",
    "HumanReviewResolver",
    "MandatoryScopedResolutionEnsemble",
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
