"""A source catalog that resolves modelling plans from declarations or profiling.

The catalog accepts declared structure — database constraints, CSVW metadata,
R2RML mappings, or an explicit user policy — and falls back to the profiler only
when no declaration is available. It resolves foreign keys across the tables it
holds (so an edge links to a real node when the target table is present, and is
skipped otherwise), and exposes the resolved plan the semantic mapper executes.

``CatalogTableMapper`` is the workflow-facing mapper: it resolves each source's
plan from the catalog on demand, so it plugs into ``ConstructionWorkflow`` like
any other mapper while remaining declaration-driven.
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from kg_construction.application.plan_resolution import resolve_modeling_plan
from kg_construction.application.resolved_plan import (
    IdentityScope,
    ResolvedModelingPlan,
)
from kg_construction.application.semantic_profiler import SemanticColumnProfiler
from kg_construction.domain.modeling import ControlledVocabulary
from kg_construction.domain.models import ConstructionBatch, SourceRecord
from kg_construction.domain.schema import JsonType

JsonDict = dict[str, Any]

logger = logging.getLogger(__name__)


class DeclaredForeignKey(BaseModel):
    """A declared foreign key from one table to another."""

    model_config = ConfigDict(extra="forbid")

    columns: list[str]
    target_table: str
    target_key: list[str] = Field(default_factory=list)
    relationship_name: str | None = None


class TableDeclaration(BaseModel):
    """Declared structure for one table, from any authoritative source."""

    model_config = ConfigDict(extra="forbid")

    table_name: str
    record_type: str
    primary_key: list[str] = Field(default_factory=list)
    foreign_keys: list[DeclaredForeignKey] = Field(default_factory=list)
    datatypes: dict[str, JsonType] = Field(default_factory=dict)
    identity_scope: IdentityScope = IdentityScope.SOURCE
    controlled_vocabularies: list[ControlledVocabulary] = Field(default_factory=list)
    origin: str = "user"  # user | csvw | r2rml | database | profiler

    @property
    def declared_key_count(self) -> int:
        return (1 if self.primary_key else 0) + len(self.foreign_keys)


class SourceCatalog:
    """Hold table declarations and resolve modelling plans from them."""

    def __init__(self, *, profiler: SemanticColumnProfiler | None = None) -> None:
        self.declarations: dict[str, TableDeclaration] = {}
        self.profiler = profiler or SemanticColumnProfiler()

    # -- Declaration intake -----------------------------------------------------

    def declare(self, declaration: TableDeclaration) -> SourceCatalog:
        self.declarations[declaration.table_name] = declaration
        return self

    @staticmethod
    def from_database_metadata(metadata: JsonDict) -> TableDeclaration:
        """Build a declaration from database catalog metadata and constraints."""

        foreign_keys = [
            DeclaredForeignKey(
                columns=list(fk["columns"]),
                target_table=fk["target_table"],
                target_key=list(fk.get("target_key", [])),
                relationship_name=fk.get("relationship_name"),
            )
            for fk in metadata.get("foreign_keys", [])
        ]
        return TableDeclaration(
            table_name=metadata["table_name"],
            record_type=metadata.get("record_type", metadata["table_name"].title()),
            primary_key=list(metadata.get("primary_key", [])),
            foreign_keys=foreign_keys,
            datatypes=dict(metadata.get("datatypes", {})),
            identity_scope=IdentityScope(metadata.get("identity_scope", "source")),
            origin="database",
        )

    @staticmethod
    def from_csvw(metadata: JsonDict) -> TableDeclaration:
        """Build a declaration from CSVW tabular metadata (a ``tableSchema``)."""

        table_schema = metadata.get("tableSchema", metadata)
        columns = table_schema.get("columns", [])
        datatypes: dict[str, JsonType] = {}
        for column in columns:
            name = column.get("name") or column.get("titles")
            datatype = column.get("datatype")
            if name and isinstance(datatype, str):
                datatypes[name] = _CSVW_DATATYPES.get(datatype, "string")
        primary_key = table_schema.get("primaryKey", [])
        if isinstance(primary_key, str):
            primary_key = [primary_key]
        foreign_keys = [
            DeclaredForeignKey(
                columns=_as_list(fk["columnReference"]),
                target_table=fk["reference"]["resource"],
                target_key=_as_list(fk["reference"].get("columnReference", [])),
            )
            for fk in table_schema.get("foreignKeys", [])
        ]
        name = metadata.get("table_name") or metadata.get("url", "records")
        return TableDeclaration(
            table_name=name,
            record_type=metadata.get("record_type", str(name).split(".")[0].title()),
            primary_key=list(primary_key),
            foreign_keys=foreign_keys,
            datatypes=datatypes,
            origin="csvw",
        )

    @staticmethod
    def from_r2rml(mapping: JsonDict) -> TableDeclaration:
        """Build a declaration from an R2RML-style logical-table mapping."""

        subject = mapping.get("subjectMap", {})
        primary_key = _as_list(subject.get("template_columns", subject.get("column", [])))
        foreign_keys = [
            DeclaredForeignKey(
                columns=_as_list(ref["child"]),
                target_table=ref["parentTriplesMap"],
                target_key=_as_list(ref.get("parent", [])),
                relationship_name=ref.get("predicate"),
            )
            for ref in mapping.get("referencingObjectMaps", [])
        ]
        return TableDeclaration(
            table_name=mapping["logicalTable"],
            record_type=subject.get("class", mapping["logicalTable"].title()),
            primary_key=primary_key,
            foreign_keys=foreign_keys,
            identity_scope=IdentityScope(mapping.get("identity_scope", "source")),
            origin="r2rml",
        )

    # -- Resolution -------------------------------------------------------------

    def resolved_plan(self, source: SourceRecord) -> ResolvedModelingPlan:
        """Resolve the plan for ``source`` from a declaration or by profiling."""

        records = source.records
        declaration = self.declarations.get(source.source_name)
        modeling_plan = self.profiler.profile(
            records,
            table_name=source.source_name,
            record_type=declaration.record_type if declaration else _record_type(source.source_name),
        )
        if declaration is not None:
            modeling_plan = _apply_declaration(modeling_plan, declaration)
            scope = declaration.identity_scope
            vocabularies = declaration.controlled_vocabularies
        else:
            scope = IdentityScope.SOURCE
            vocabularies = []

        plan = resolve_modeling_plan(modeling_plan, source_name=source.source_name, identity_scope=scope)
        if vocabularies:
            plan = plan.model_copy(update={"controlled_vocabularies": list(vocabularies)})
        return self._link_foreign_keys(plan)

    def _link_foreign_keys(self, plan: ResolvedModelingPlan) -> ResolvedModelingPlan:
        """Mark foreign keys whose target table is present in the catalog."""

        present = {declaration.record_type for declaration in self.declarations.values()}
        if not present:
            return plan
        updated = [
            fk.model_copy(update={"target_available": True})
            if not fk.self_referential and fk.target_class in present
            else fk
            for fk in plan.foreign_keys
        ]
        return plan.model_copy(update={"foreign_keys": updated})

    def declared_key_mapping_coverage(self, plans: dict[str, ResolvedModelingPlan]) -> float:
        """Return the fraction of declared keys and foreign keys mapped into plans."""

        declared = 0
        mapped = 0
        for declaration in self.declarations.values():
            plan = plans.get(declaration.record_type)
            if declaration.primary_key:
                declared += 1
                if plan is not None and plan.clazz(declaration.record_type).identity_keys == declaration.primary_key:
                    mapped += 1
            for foreign_key in declaration.foreign_keys:
                declared += 1
                if plan is not None and any(
                    fk.columns == foreign_key.columns for fk in plan.foreign_keys
                ):
                    mapped += 1
        return 1.0 if declared == 0 else mapped / declared


class CatalogTableMapper:
    """Workflow mapper that resolves each source's plan from the catalog."""

    name = "catalog_table_mapper"
    technique = "policy_driven_semantic_mapping"

    def __init__(self, catalog: SourceCatalog | None = None) -> None:
        self.catalog = catalog or SourceCatalog()

    def map_source(self, source: SourceRecord) -> ConstructionBatch:
        from kg_construction.application.semantic_table_mapper import SemanticTableMapper

        plan = self.catalog.resolved_plan(source)
        return SemanticTableMapper(plan).map_source(source)


_CSVW_DATATYPES: dict[str, JsonType] = {
    "integer": "integer",
    "int": "integer",
    "long": "integer",
    "decimal": "number",
    "double": "number",
    "float": "number",
    "number": "number",
    "boolean": "boolean",
    "date": "date",
    "dateTime": "datetime",
    "datetime": "datetime",
    "string": "string",
}


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return list(value)


def _record_type(source_name: str) -> str:
    stem = source_name.rsplit("/", 1)[-1].split(".")[0]
    tokens = [token for token in stem.replace("-", "_").split("_") if token]
    singular = _singularize(tokens[-1]) if tokens else "Record"
    tokens = tokens[:-1] + [singular] if tokens else ["Record"]
    return "".join(token[:1].upper() + token[1:] for token in tokens) or "Record"


def _singularize(token: str) -> str:
    lowered = token.lower()
    if lowered.endswith(("ss", "us", "is")):
        return token
    if lowered.endswith("ies") and len(lowered) > 3:
        return token[:-3] + "y"
    if lowered.endswith("s") and len(lowered) > 3:
        return token[:-1]
    return token


def _apply_declaration(modeling_plan: Any, declaration: TableDeclaration) -> Any:
    """Override profiler proposals with declared keys, foreign keys, and datatypes."""

    from kg_construction.application.semantic_profiler import ForeignKeyLink
    from kg_construction.domain.modeling import ColumnRole

    assignments = [assignment.model_copy(deep=True) for assignment in modeling_plan.assignments]
    fk_columns = {column for fk in declaration.foreign_keys for column in fk.columns}
    fk_target = {
        fk.columns[0]: fk for fk in declaration.foreign_keys if fk.columns
    }

    for assignment in assignments:
        column = assignment.column
        if declaration.primary_key and column in declaration.primary_key:
            assignment.role = ColumnRole.IDENTIFIER
            assignment.materialization = "identity"
            assignment.reference_type = None
        elif column in fk_columns:
            fk = fk_target.get(column)
            assignment.role = ColumnRole.ENTITY_REFERENCE
            assignment.materialization = "edge"
            assignment.reference_type = _record_type(fk.target_table) if fk else assignment.reference_type
        if column in declaration.datatypes:
            assignment.datatype = declaration.datatypes[column]

    identity_columns = declaration.primary_key or modeling_plan.identity_columns
    foreign_keys = [
        ForeignKeyLink(source_column=fk.columns[0], target_column=(fk.target_key[0] if fk.target_key else None))
        for fk in declaration.foreign_keys
        if fk.columns
    ] or modeling_plan.foreign_keys

    return modeling_plan.model_copy(
        update={
            "identity_columns": list(identity_columns),
            "foreign_keys": foreign_keys,
            "assignments": assignments,
        }
    )


__all__ = [
    "CatalogTableMapper",
    "DeclaredForeignKey",
    "SourceCatalog",
    "TableDeclaration",
]
