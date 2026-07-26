"""Structured modelling policy and resolved-plan contracts."""

from kg_construction.structured.modeling_policy import (
    CanonicalizationRule,
    ColumnPolicy,
    ModelingPolicy,
    PromotionThresholds,
    RelationshipDeclaration,
    default_modeling_policy,
)
from kg_construction.structured.resolved_plan import (
    ClassKind,
    EvaluationExpectations,
    ForeignKeyResolution,
    IdentityRole,
    IdentityScope,
    Materialization,
    OnMissingTarget,
    ResolvedAttribute,
    ResolvedClass,
    ResolvedModelingPlan,
    ResolvedProperty,
    ResolvedRelationship,
    ReviewState,
    SchemaDefinitions,
)

__all__ = [
    "CanonicalizationRule",
    "ClassKind",
    "ColumnPolicy",
    "EvaluationExpectations",
    "ForeignKeyResolution",
    "IdentityRole",
    "IdentityScope",
    "Materialization",
    "ModelingPolicy",
    "OnMissingTarget",
    "PromotionThresholds",
    "RelationshipDeclaration",
    "ResolvedAttribute",
    "ResolvedClass",
    "ResolvedModelingPlan",
    "ResolvedProperty",
    "ResolvedRelationship",
    "ReviewState",
    "SchemaDefinitions",
    "default_modeling_policy",
]
