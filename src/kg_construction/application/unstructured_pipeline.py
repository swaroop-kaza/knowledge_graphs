"""Compatibility imports for the canonical unstructured pipeline package.

New source code should import these contracts from
``kg_construction.unstructured.pipeline``. This module remains importable for
downstream callers that used the former application-layer location.
"""

from kg_construction.unstructured.pipeline import (
    AttributeExtractor,
    ClaimExtractor,
    ConservativeCoreferenceResolver,
    CoreferenceResolver,
    EntityLink,
    EntityLinkProvider,
    ExplicitNilObservedResolver,
    IdentityResolution,
    IdentityResolver,
    KeepUnresolvedResolver,
    MentionDetector,
    PatternClaimExtractor,
    PipelineResult,
    ProviderContractError,
    ProviderIdentityResolver,
    RuleAttributeExtractor,
    RuleMentionDetector,
    RuleSegmenter,
    RuntimeProfile,
    Segmenter,
    StringCoreference,
    UnstructuredPipeline,
    UnstructuredPipelineConfig,
)

__all__ = [
    "AttributeExtractor",
    "ClaimExtractor",
    "ConservativeCoreferenceResolver",
    "CoreferenceResolver",
    "EntityLink",
    "EntityLinkProvider",
    "ExplicitNilObservedResolver",
    "IdentityResolution",
    "IdentityResolver",
    "KeepUnresolvedResolver",
    "MentionDetector",
    "PatternClaimExtractor",
    "PipelineResult",
    "ProviderContractError",
    "ProviderIdentityResolver",
    "RuleAttributeExtractor",
    "RuleMentionDetector",
    "RuleSegmenter",
    "RuntimeProfile",
    "Segmenter",
    "StringCoreference",
    "UnstructuredPipeline",
    "UnstructuredPipelineConfig",
]
