"""Unstructured-data knowledge-graph construction.

The pillar preserves mentions, evidence, semantic assertions (``Claim``),
relations, coreference decisions, and unresolved identities before graph
materialization.
"""

from importlib import import_module
from typing import Any

from kg_construction.domain.claim import (
    Claim,
    DocumentRef,
    Evidence,
    Mention,
    TextSpan,
)
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

_RUNTIME_EXPORTS = {
    "HeldOutClaimCase",
    "HeldOutClaimCaseReport",
    "HeldOutClaimDataset",
    "HeldOutClaimEvaluationReport",
    "MandatoryUnstructuredRuntime",
}


def __getattr__(name: str) -> Any:
    """Load the heavy locked runtime only when a runtime symbol is requested."""

    if name not in _RUNTIME_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module("kg_construction.unstructured.runtime"), name)
    globals()[name] = value
    return value

__all__ = [
    "AttributeExtractor",
    "Claim",
    "ClaimExtractor",
    "ConservativeCoreferenceResolver",
    "CoreferenceResolver",
    "DocumentRef",
    "EntityLink",
    "EntityLinkProvider",
    "Evidence",
    "ExplicitNilObservedResolver",
    "HeldOutClaimCase",
    "HeldOutClaimCaseReport",
    "HeldOutClaimDataset",
    "HeldOutClaimEvaluationReport",
    "IdentityResolution",
    "IdentityResolver",
    "KeepUnresolvedResolver",
    "MandatoryUnstructuredRuntime",
    "Mention",
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
    "TextSpan",
    "UnstructuredPipeline",
    "UnstructuredPipelineConfig",
]
