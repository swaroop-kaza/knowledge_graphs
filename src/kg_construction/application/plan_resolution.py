"""Resolve a proposed modelling plan into the single resolved contract.

Profiling proposes a :class:`~kg_construction.application.semantic_profiler.ModelingPlan`;
this turns it into a :class:`~kg_construction.application.resolved_plan.ResolvedModelingPlan`,
the only input accepted for promoted-graph construction.

Resolution keeps semantic role and physical materialization separate: a
categorical dimension materialised as a node becomes a governed concept plus an
object relationship, while the same role materialised as a property becomes a
scalar-enumeration literal — never both. Foreign keys resolve to relationships
whose missing targets are skipped, not fabricated as placeholder entities.
"""

from __future__ import annotations

import logging

from kg_construction.application.modeling_policy import CanonicalizationRule
from kg_construction.application.resolved_plan import (
    ClassKind,
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
)
from kg_construction.application.semantic_profiler import ColumnRoleAssignment, ModelingPlan
from kg_construction.domain.modeling import Cardinality, ColumnRole
from kg_construction.domain.schema import JsonType

logger = logging.getLogger(__name__)

_DTYPE_JSON: dict[str, JsonType] = {
    "integer": "integer",
    "number": "number",
    "boolean": "boolean",
    "string": "string",
    "date": "date",
    "datetime": "datetime",
}
_ROLE_DATATYPE: dict[ColumnRole, JsonType] = {
    ColumnRole.MEASURE: "number",
    ColumnRole.TEMPORAL: "datetime",
    ColumnRole.FREE_TEXT: "string",
    ColumnRole.IDENTIFIER: "string",
    ColumnRole.CATEGORICAL_DIMENSION: "string",
}


def _datatype(assignment: ColumnRoleAssignment) -> JsonType:
    if assignment.datatype is not None:
        return assignment.datatype
    inferred = _DTYPE_JSON.get(assignment.features.dtype)
    if inferred is not None:
        return inferred
    return _ROLE_DATATYPE.get(assignment.role, "string")


def _cardinality(null_rate: float, *, identity: bool = False) -> Cardinality:
    if identity or null_rate == 0.0:
        return Cardinality(min_count=1, max_count=1)
    return Cardinality.zero_or_one()


def _relationship_name(assignment: ColumnRoleAssignment, reference_type: str) -> str:
    if assignment.relationship_type:
        return assignment.relationship_type
    return f"has_{reference_type.lower()}"


def resolve_modeling_plan(
    plan: ModelingPlan,
    *,
    source_name: str,
    identity_scope: IdentityScope = IdentityScope.SOURCE,
    version: str = "v1",
) -> ResolvedModelingPlan:
    """Resolve a proposed plan into the construction-ready contract."""

    record = plan.record_type
    classes: dict[str, ResolvedClass] = {
        record: ResolvedClass(
            name=record,
            kind=ClassKind.ENTITY,
            identity_keys=list(plan.identity_columns),
            identity_scope=identity_scope,
            source_table=plan.table_name,
        )
    }
    properties: list[ResolvedProperty] = []
    relationships: list[ResolvedRelationship] = []
    foreign_keys: list[ForeignKeyResolution] = []
    attributes: list[ResolvedAttribute] = []

    for assignment in plan.assignments:
        column = assignment.column
        materialization = Materialization(assignment.materialization)
        null_rate = assignment.features.null_rate
        canonicalization: CanonicalizationRule | None = None
        identity_role = IdentityRole.NONE
        target: str | None = None
        datatype: JsonType | None = None

        if materialization is Materialization.EXCLUDED:
            pass
        elif materialization is Materialization.IDENTITY:
            identity_role = IdentityRole.PRIMARY
            datatype = _datatype(assignment)
            target = column
            properties.append(
                ResolvedProperty(
                    class_name=record,
                    name=column,
                    datatype=datatype,
                    nullable=False,
                    cardinality=_cardinality(null_rate, identity=True),
                    source_column=column,
                )
            )
        elif materialization is Materialization.PROPERTY:
            datatype = _datatype(assignment)
            target = column
            properties.append(
                ResolvedProperty(
                    class_name=record,
                    name=column,
                    datatype=datatype,
                    nullable=null_rate > 0.0,
                    cardinality=_cardinality(null_rate),
                    source_column=column,
                )
            )
        elif materialization is Materialization.NODE:
            # A categorical dimension promoted to a shared node: a governed concept.
            reference_type = assignment.reference_type or column
            target = reference_type
            canonicalization = CanonicalizationRule()
            classes.setdefault(
                reference_type,
                ResolvedClass(
                    name=reference_type,
                    kind=ClassKind.CONCEPT,
                    concept_scheme=f"urn:kg:scheme:{plan.table_name}:{column}",
                    identity_scope=identity_scope,
                ),
            )
            relationships.append(
                ResolvedRelationship(
                    name=_relationship_name(assignment, reference_type),
                    domain=record,
                    range=reference_type,
                    cardinality=_cardinality(null_rate),
                    via="categorical",
                    source_columns=[column],
                    on_missing_target=OnMissingTarget.SKIP,
                )
            )
        elif materialization is Materialization.EDGE:
            # A foreign key to a business entity.
            reference_type = assignment.reference_type or column
            target = reference_type
            identity_role = IdentityRole.FOREIGN
            canonicalization = CanonicalizationRule()
            is_self_reference = reference_type == record
            if not is_self_reference:
                classes.setdefault(
                    reference_type,
                    ResolvedClass(
                        name=reference_type,
                        kind=ClassKind.ENTITY,
                        identity_scope=identity_scope,
                    ),
                )
            target_key = list(plan.identity_columns) if is_self_reference else []
            relationship_name = _relationship_name(assignment, reference_type)
            relationships.append(
                ResolvedRelationship(
                    name=relationship_name,
                    domain=record,
                    range=reference_type,
                    cardinality=_cardinality(null_rate),
                    via="foreign_key",
                    source_columns=[column],
                    on_missing_target=OnMissingTarget.SKIP,
                )
            )
            foreign_keys.append(
                ForeignKeyResolution(
                    columns=[column],
                    source_class=record,
                    target_class=reference_type,
                    target_key=target_key,
                    relationship_name=relationship_name,
                    cardinality=_cardinality(null_rate),
                    on_missing=OnMissingTarget.SKIP,
                    self_referential=is_self_reference,
                    target_available=is_self_reference,
                )
            )

        attributes.append(
            ResolvedAttribute(
                column=column,
                semantic_role=assignment.role,
                materialization=materialization,
                target=target,
                datatype=datatype,
                nullable=null_rate > 0.0 and materialization is not Materialization.IDENTITY,
                cardinality=_cardinality(null_rate, identity=materialization is Materialization.IDENTITY),
                canonicalization=canonicalization,
                identity_role=identity_role,
                confidence=assignment.confidence,
            )
        )

    resolved = ResolvedModelingPlan(
        source_name=source_name,
        table_name=plan.table_name,
        record_class=record,
        version=version,
        classes=sorted(classes.values(), key=lambda item: item.name),
        properties=properties,
        relationships=relationships,
        foreign_keys=foreign_keys,
        attributes=attributes,
    )
    logger.info(
        "Resolved %s into %d classes, %d properties, %d relationships (%d foreign keys)",
        record,
        len(resolved.classes),
        len(resolved.properties),
        len(resolved.relationships),
        len(resolved.foreign_keys),
    )
    return resolved


__all__ = ["resolve_modeling_plan"]
