"""Structured-data knowledge-graph construction.

The pillar owns semantic profiling, explicit identity and relationship
declarations, duplicate-resistant key handling, and schema projections.
"""

from kg_construction.structured.catalog import (
    CatalogTableMapper,
    DeclaredColumnMapping,
    DeclaredForeignKey,
    SourceCatalog,
    TableDeclaration,
)
from kg_construction.structured.contracts import (
    CanonicalizationRule,
    ClassKind,
    ColumnPolicy,
    EvaluationExpectations,
    ForeignKeyResolution,
    IdentityRole,
    IdentityScope,
    Materialization,
    ModelingPolicy,
    OnMissingTarget,
    PromotionThresholds,
    RelationshipDeclaration,
    ResolvedAttribute,
    ResolvedClass,
    ResolvedModelingPlan,
    ResolvedProperty,
    ResolvedRelationship,
    ReviewState,
    SchemaDefinitions,
    default_modeling_policy,
)
from kg_construction.structured.identity import (
    TargetKeyIndex,
    TargetKeyLookup,
    canonical_key,
    normalize_key_value,
    validate_unique_key,
)
from kg_construction.structured.mapping import SemanticTableMapper
from kg_construction.structured.planning import resolve_modeling_plan
from kg_construction.structured.profiling import (
    ColumnFeatures,
    ColumnRoleAssignment,
    ForeignKeyLink,
    KeyCandidate,
    ModelingPlan,
    SemanticColumnProfiler,
)
from kg_construction.structured.schema_projection import (
    ProjectionBundle,
    assert_projections_agree,
    project_all,
    projection_disagreements,
)

__all__ = [
    "CanonicalizationRule",
    "CatalogTableMapper",
    "ClassKind",
    "ColumnFeatures",
    "ColumnPolicy",
    "ColumnRoleAssignment",
    "DeclaredColumnMapping",
    "DeclaredForeignKey",
    "EvaluationExpectations",
    "ForeignKeyLink",
    "ForeignKeyResolution",
    "IdentityRole",
    "IdentityScope",
    "KeyCandidate",
    "Materialization",
    "ModelingPlan",
    "ModelingPolicy",
    "OnMissingTarget",
    "ProjectionBundle",
    "PromotionThresholds",
    "RelationshipDeclaration",
    "ResolvedAttribute",
    "ResolvedClass",
    "ResolvedModelingPlan",
    "ResolvedProperty",
    "ResolvedRelationship",
    "ReviewState",
    "SchemaDefinitions",
    "SemanticColumnProfiler",
    "SemanticTableMapper",
    "SourceCatalog",
    "TableDeclaration",
    "TargetKeyIndex",
    "TargetKeyLookup",
    "assert_projections_agree",
    "canonical_key",
    "default_modeling_policy",
    "normalize_key_value",
    "project_all",
    "projection_disagreements",
    "resolve_modeling_plan",
    "validate_unique_key",
]
