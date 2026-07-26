"""Consistent projections of one resolved schema contract."""

from kg_construction.structured.schema_projections import (
    ProjectionBundle,
    assert_projections_agree,
    project_all,
    projection_disagreements,
)

__all__ = [
    "ProjectionBundle",
    "assert_projections_agree",
    "project_all",
    "projection_disagreements",
]
