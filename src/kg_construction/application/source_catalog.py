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
import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from kg_construction.application.modeling_policy import CanonicalizationRule
from kg_construction.application.plan_resolution import resolve_modeling_plan
from kg_construction.application.resolved_plan import (
    ClassKind,
    ForeignKeyResolution,
    IdentityRole,
    IdentityScope,
    Materialization,
    ResolvedClass,
    ResolvedModelingPlan,
    ResolvedRelationship,
)
from kg_construction.application.semantic_profiler import SemanticColumnProfiler
from kg_construction.application.structured_identity import (
    TargetKeyIndex,
    validate_unique_key,
)
from kg_construction.domain.evidence import row_evidence
from kg_construction.domain.modeling import Cardinality, ColumnRole, ControlledVocabulary
from kg_construction.domain.models import (
    ConstructionBatch,
    RelationCandidate,
    SourceRecord,
)
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

    @model_validator(mode="after")
    def _valid_key_arity(self) -> DeclaredForeignKey:
        if not self.columns:
            raise ValueError("a foreign key must declare at least one source column")
        if self.target_key and len(self.columns) != len(self.target_key):
            raise ValueError(
                f"foreign-key columns {self.columns!r} and target key "
                f"{self.target_key!r} must have equal arity"
            )
        return self


class DeclaredColumnMapping(BaseModel):
    """Authoritative semantic/materialisation decision for one source column."""

    model_config = ConfigDict(extra="forbid")

    role: ColumnRole | None = None
    materialization: Materialization | None = None
    target: str | None = None
    relationship_name: str | None = None
    datatype: JsonType | None = None
    controlled_vocabulary: str | None = None
    canonicalization: CanonicalizationRule | None = None
    concept_scheme: str | None = None

    @model_validator(mode="after")
    def _mapping_is_coherent(self) -> DeclaredColumnMapping:
        if (
            self.materialization in {Materialization.NODE, Materialization.EDGE}
            and not self.target
        ):
            raise ValueError("node/edge column mappings must declare a target class")
        if self.concept_scheme and self.materialization is not Materialization.NODE:
            raise ValueError("concept_scheme is valid only for node materialization")
        return self


class TableDeclaration(BaseModel):
    """Declared structure for one table, from any authoritative source."""

    model_config = ConfigDict(extra="forbid")

    table_name: str
    record_type: str
    primary_key: list[str] = Field(default_factory=list)
    alternate_keys: list[list[str]] = Field(default_factory=list)
    foreign_keys: list[DeclaredForeignKey] = Field(default_factory=list)
    datatypes: dict[str, JsonType] = Field(default_factory=dict)
    column_mappings: dict[str, DeclaredColumnMapping] = Field(default_factory=dict)
    identity_scope: IdentityScope = IdentityScope.SOURCE
    controlled_vocabularies: list[ControlledVocabulary] = Field(default_factory=list)
    table_kind: Literal["auto", "entity", "association", "event", "edge"] = "auto"
    association_relationship_name: str | None = None
    origin: str = "user"  # user | csvw | r2rml | database | profiler

    @property
    def declared_key_count(self) -> int:
        return (
            (1 if self.primary_key else 0)
            + len(self.alternate_keys)
            + len(self.foreign_keys)
        )

    @model_validator(mode="after")
    def _keys_are_well_formed(self) -> TableDeclaration:
        keys = [self.primary_key, *self.alternate_keys]
        if any(key and len(key) != len(set(key)) for key in keys):
            raise ValueError("declared keys must not repeat a column")
        if any(not key for key in self.alternate_keys):
            raise ValueError("alternate keys must not be empty")
        signatures = [tuple(key) for key in keys if key]
        if len(signatures) != len(set(signatures)):
            raise ValueError("primary and alternate key declarations must be distinct")
        vocabulary_names = {
            vocabulary.name for vocabulary in self.controlled_vocabularies
        }
        unknown_vocabularies = sorted(
            {
                mapping.controlled_vocabulary
                for mapping in self.column_mappings.values()
                if mapping.controlled_vocabulary
                and mapping.controlled_vocabulary not in vocabulary_names
            }
        )
        if unknown_vocabularies:
            raise ValueError(
                "column mappings reference undeclared controlled vocabularies: "
                f"{unknown_vocabularies}"
            )
        non_lexical_vocabulary_columns = sorted(
            column
            for column, mapping in self.column_mappings.items()
            if mapping.controlled_vocabulary
            and (
                mapping.datatype
                or self.datatypes.get(column)
                or "string"
            )
            != "string"
        )
        if non_lexical_vocabulary_columns:
            raise ValueError(
                "controlled-vocabulary codes must use lexical string datatypes "
                f"for columns {non_lexical_vocabulary_columns}"
            )
        return self


class SourceCatalog:
    """Hold table declarations and resolve modelling plans from them."""

    def __init__(self, *, profiler: SemanticColumnProfiler | None = None) -> None:
        self.declarations: dict[str, TableDeclaration] = {}
        self.profiler = profiler or SemanticColumnProfiler()

    # -- Declaration intake -----------------------------------------------------

    def declare(self, declaration: TableDeclaration) -> SourceCatalog:
        self.declarations[declaration.table_name] = declaration
        return self

    def declaration_for(self, source_or_table_name: str) -> TableDeclaration | None:
        """Return one unambiguous declaration by full name, basename, or stem."""

        exact = self.declarations.get(source_or_table_name)
        if exact is not None:
            return exact
        normalized = _catalog_name(source_or_table_name)
        matches = [
            declaration
            for name, declaration in self.declarations.items()
            if _catalog_name(name) == normalized
        ]
        return matches[0] if len(matches) == 1 else None

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
            alternate_keys=_metadata_alternate_keys(metadata),
            foreign_keys=foreign_keys,
            datatypes=dict(metadata.get("datatypes", {})),
            column_mappings=dict(metadata.get("column_mappings", {})),
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
            alternate_keys=[
                _as_list(key)
                for key in table_schema.get("candidateKeys", [])
            ],
            foreign_keys=foreign_keys,
            datatypes=datatypes,
            column_mappings=dict(metadata.get("column_mappings", {})),
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
            alternate_keys=[
                _as_list(key) for key in mapping.get("alternateKeys", [])
            ],
            foreign_keys=foreign_keys,
            column_mappings=dict(mapping.get("column_mappings", {})),
            identity_scope=IdentityScope(mapping.get("identity_scope", "source")),
            origin="r2rml",
        )

    # -- Resolution -------------------------------------------------------------

    def resolved_plan(
        self,
        source: SourceRecord,
        *,
        require_explicit_vocabulary_bindings: bool = False,
    ) -> ResolvedModelingPlan:
        """Resolve the plan for ``source`` from a declaration or by profiling."""

        records = source.records
        declaration = self.declaration_for(source.source_name)
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
        if declaration is not None:
            plan = self._apply_declared_keys(plan, declaration)
        if vocabularies:
            vocabulary_names = {item.name for item in vocabularies}
            explicit_bindings = {
                column: mapping.controlled_vocabulary
                for column, mapping in (
                    declaration.column_mappings.items()
                    if declaration is not None
                    else ()
                )
                if mapping.controlled_vocabulary
            }
            if require_explicit_vocabulary_bindings:
                unbound = sorted(
                    vocabulary_names - set(explicit_bindings.values())
                )
                if unbound:
                    raise ValueError(
                        "strict structured declarations require an explicit "
                        "column_mappings.<column>.controlled_vocabulary binding "
                        f"for {unbound}"
                    )
            attributes = [
                attribute.model_copy(
                    update={
                        "controlled_vocabulary": (
                            explicit_bindings.get(attribute.column)
                            or (
                                attribute.target
                                if not require_explicit_vocabulary_bindings
                                and attribute.target in vocabulary_names
                                else None
                            )
                        )
                    }
                )
                if (
                    attribute.column in explicit_bindings
                    or (
                        not require_explicit_vocabulary_bindings
                        and attribute.target in vocabulary_names
                    )
                )
                else attribute
                for attribute in plan.attributes
            ]
            controlled_by_column = {
                attribute.column: attribute.controlled_vocabulary
                for attribute in attributes
                if attribute.controlled_vocabulary
            }
            properties = [
                prop.model_copy(
                    update={
                        "controlled_vocabulary": controlled_by_column.get(
                            prop.source_column or ""
                        )
                    }
                )
                if prop.source_column in controlled_by_column
                else prop
                for prop in plan.properties
            ]
            relationships = [
                relationship.model_copy(
                    update={
                        "controlled_vocabulary": next(
                            (
                                controlled_by_column[column]
                                for column in relationship.source_columns
                                if column in controlled_by_column
                            ),
                            None,
                        )
                    }
                )
                if any(
                    column in controlled_by_column
                    for column in relationship.source_columns
                )
                else relationship
                for relationship in plan.relationships
            ]
            plan = plan.model_copy(
                update={
                    "attributes": attributes,
                    "properties": properties,
                    "relationships": relationships,
                    "controlled_vocabularies": list(vocabularies),
                }
            )
        if declaration is not None:
            plan = self._apply_declared_foreign_keys(plan, declaration)
            plan = self._apply_table_kind(plan, declaration, records)
        return self._link_foreign_keys(plan)

    @staticmethod
    def _apply_declared_keys(
        plan: ResolvedModelingPlan,
        declaration: TableDeclaration,
    ) -> ResolvedModelingPlan:
        """Attach authoritative primary/alternate identity metadata to the plan."""

        record = plan.clazz(plan.record_class).model_copy(
            update={
                "identity_keys": list(declaration.primary_key),
                "alternate_keys": [
                    list(key) for key in declaration.alternate_keys
                ],
                "identity_scope": declaration.identity_scope,
            }
        )
        alternate_columns = {
            column for key in declaration.alternate_keys for column in key
        }
        attributes = []
        for attribute in plan.attributes:
            updates: dict[str, object] = {}
            mapping = declaration.column_mappings.get(attribute.column)
            if (
                attribute.column in alternate_columns
                and attribute.identity_role is IdentityRole.NONE
            ):
                updates["identity_role"] = IdentityRole.ALTERNATE
            if mapping is not None and mapping.canonicalization is not None:
                updates["canonicalization"] = mapping.canonicalization
            attributes.append(
                attribute.model_copy(update=updates) if updates else attribute
            )
        classes = [
            record if item.name == plan.record_class else item
            for item in plan.classes
        ]
        concept_schemes = {
            mapping.target: mapping.concept_scheme
            for mapping in declaration.column_mappings.values()
            if mapping.target and mapping.concept_scheme
        }
        classes = [
            item.model_copy(
                update={"concept_scheme": concept_schemes[item.name]}
            )
            if item.name in concept_schemes
            else item
            for item in classes
        ]
        return plan.model_copy(
            update={
                "classes": classes,
                "attributes": attributes,
            }
        )

    @staticmethod
    def _apply_table_kind(
        plan: ResolvedModelingPlan,
        declaration: TableDeclaration,
        records: list[JsonDict],
    ) -> ResolvedModelingPlan:
        kind = declaration.table_kind
        if kind == "auto":
            foreign_columns = {
                column for foreign_key in declaration.foreign_keys for column in foreign_key.columns
            }
            observed_columns = {column for row in records for column in row}
            qualifiers = observed_columns - foreign_columns - set(declaration.primary_key)
            kind = (
                "association"
                if len(declaration.foreign_keys) >= 2 and qualifiers
                else "entity"
            )
        if kind in {"edge", "entity"}:
            return plan
        record = plan.clazz(plan.record_class).model_copy(
            update={
                "kind": (
                    ClassKind.EVENT if kind == "event" else ClassKind.ASSOCIATION
                )
            }
        )
        return plan.model_copy(
            update={
                "classes": [
                    record if item.name == plan.record_class else item
                    for item in plan.classes
                ]
            }
        )

    def _apply_declared_foreign_keys(
        self,
        plan: ResolvedModelingPlan,
        declaration: TableDeclaration,
    ) -> ResolvedModelingPlan:
        """Coalesce declared, including composite, foreign keys into one relationship."""

        if not declaration.foreign_keys:
            return plan
        relationships = list(plan.relationships)
        foreign_keys = list(plan.foreign_keys)
        attributes = list(plan.attributes)
        classes = {item.name: item for item in plan.classes}

        for declared in declaration.foreign_keys:
            columns = set(declared.columns)
            target_declaration = self.declaration_for(declared.target_table)
            if target_declaration is not None:
                if not target_declaration.primary_key:
                    raise ValueError(
                        f"foreign-key target {target_declaration.table_name!r} "
                        "must declare a primary key"
                    )
                target_key = list(
                    declared.target_key or target_declaration.primary_key
                )
                declared_target_keys = {
                    tuple(target_declaration.primary_key),
                    *(
                        tuple(key)
                        for key in target_declaration.alternate_keys
                    ),
                }
                if tuple(target_key) not in declared_target_keys:
                    raise ValueError(
                        f"foreign key {declared.columns!r} targets "
                        f"{target_key!r}, which is not a declared primary or "
                        f"alternate key of {target_declaration.table_name!r}"
                    )
                target_key_datatypes: list[JsonType] = [
                    target_declaration.datatypes.get(column, "string")
                    for column in target_key
                ]
            else:
                target_key = list(declared.target_key)
                target_key_datatypes = []
            target_class = (
                target_declaration.record_type
                if target_declaration is not None
                else _record_type(declared.target_table)
            )
            target_source = (
                target_declaration.table_name
                if target_declaration is not None
                else declared.target_table
            )
            target_scope = (
                target_declaration.identity_scope
                if target_declaration is not None
                else IdentityScope.SOURCE
            )
            self_referential = target_declaration is declaration or (
                _catalog_name(declared.target_table) == _catalog_name(declaration.table_name)
            )
            if self_referential:
                target_class = declaration.record_type
                target_source = plan.source_name
                target_scope = declaration.identity_scope

            existing_relationship = next(
                (
                    relationship
                    for relationship in relationships
                    if set(relationship.source_columns) & columns
                ),
                None,
            )
            relationship_name = (
                declared.relationship_name
                or (
                    existing_relationship.name
                    if existing_relationship is not None
                    else f"has_{target_class.lower()}"
                )
            )
            cardinality = (
                existing_relationship.cardinality
                if existing_relationship is not None
                else Cardinality.zero_or_one()
            )

            relationships = [
                relationship
                for relationship in relationships
                if not (set(relationship.source_columns) & columns)
            ]
            foreign_keys = [
                foreign_key
                for foreign_key in foreign_keys
                if not (set(foreign_key.columns) & columns)
            ]
            relationships.append(
                ResolvedRelationship(
                    name=relationship_name,
                    domain=plan.record_class,
                    range=target_class,
                    cardinality=cardinality,
                    via="foreign_key",
                    source_columns=list(declared.columns),
                    target_columns=target_key,
                )
            )
            foreign_keys.append(
                ForeignKeyResolution(
                    columns=list(declared.columns),
                    source_class=plan.record_class,
                    target_class=target_class,
                    target_key=target_key,
                    target_key_datatypes=target_key_datatypes,
                    target_source_name=target_source,
                    target_identity_scope=target_scope,
                    relationship_name=relationship_name,
                    cardinality=cardinality,
                    self_referential=self_referential,
                    target_available=self_referential or target_declaration is not None,
                )
            )
            attributes = [
                attribute.model_copy(
                    update={
                        "semantic_role": ColumnRole.ENTITY_REFERENCE,
                        "materialization": Materialization.EDGE,
                        "target": target_class,
                        "identity_role": IdentityRole.FOREIGN,
                    }
                )
                if attribute.column in columns
                else attribute
                for attribute in attributes
            ]
            target_model = ResolvedClass(
                name=target_class,
                kind=ClassKind.ENTITY,
                identity_keys=list(
                    target_declaration.primary_key
                    if target_declaration is not None
                    else target_key
                ),
                alternate_keys=(
                    [
                        list(key)
                        for key in target_declaration.alternate_keys
                    ]
                    if target_declaration is not None
                    else []
                ),
                identity_scope=target_scope,
                source_table=target_source,
            )
            if target_class != plan.record_class:
                classes[target_class] = target_model

        return plan.model_copy(
            update={
                "classes": sorted(
                    (
                        item
                        for name, item in classes.items()
                        if name
                        in (
                            {plan.record_class}
                            | {prop.class_name for prop in plan.properties}
                            | {
                                endpoint
                                for relationship in relationships
                                for endpoint in (relationship.domain, relationship.range)
                            }
                        )
                    ),
                    key=lambda item: item.name,
                ),
                "relationships": relationships,
                "foreign_keys": foreign_keys,
                "attributes": attributes,
            }
        )

    def _link_foreign_keys(self, plan: ResolvedModelingPlan) -> ResolvedModelingPlan:
        """Mark foreign keys whose target table is present in the catalog."""

        by_type = {declaration.record_type: declaration for declaration in self.declarations.values()}
        updated = []
        for foreign_key in plan.foreign_keys:
            target = by_type.get(foreign_key.target_class)
            if target is None:
                updated.append(foreign_key)
                continue
            updated.append(
                foreign_key.model_copy(
                    update={
                        "target_available": True,
                        "target_source_name": target.table_name,
                        "target_identity_scope": target.identity_scope,
                        "target_key": foreign_key.target_key or target.primary_key,
                        "target_key_datatypes": [
                            target.datatypes.get(column, "string")
                            for column in (
                                foreign_key.target_key or target.primary_key
                            )
                        ],
                    }
                )
            )
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
            for alternate_key in declaration.alternate_keys:
                declared += 1
                if (
                    plan is not None
                    and alternate_key
                    in plan.clazz(declaration.record_type).alternate_keys
                ):
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

    def __init__(
        self,
        catalog: SourceCatalog | None = None,
        *,
        require_declarations: bool = False,
    ) -> None:
        self.catalog = catalog or SourceCatalog()
        self.require_declarations = require_declarations
        self._prepared_plans: dict[str, ResolvedModelingPlan] = {}
        self._target_key_index: TargetKeyIndex = {}

    def prepare_sources(self, sources: list[SourceRecord]) -> None:
        """Resolve every source before mapping so cross-table targets are known."""

        if self.require_declarations:
            missing = [
                source.source_name
                for source in sources
                if self.catalog.declaration_for(source.source_name) is None
            ]
            if missing:
                raise ValueError(
                    f"structured sources have no authoritative declaration: {sorted(missing)}"
                )
        self._prepared_plans = {
            source.source_id: self.catalog.resolved_plan(
                source,
                require_explicit_vocabulary_bindings=self.require_declarations,
            )
            for source in sources
        }
        self._target_key_index = {}
        from kg_construction.application.semantic_table_mapper import SemanticTableMapper

        for source in sources:
            plan = self._prepared_plans[source.source_id]
            record_class = plan.clazz(plan.record_class)
            attributes = {attribute.column: attribute for attribute in plan.attributes}
            if not record_class.identity_keys:
                continue

            primary_types = _key_datatypes(
                attributes,
                record_class.identity_keys,
                source_name=source.source_name,
            )
            primary_rows = validate_unique_key(
                source.records,
                columns=record_class.identity_keys,
                datatypes=primary_types,
                key_name=(
                    f"primary key {plan.record_class}"
                    f"({', '.join(record_class.identity_keys)})"
                ),
            )
            primary_id_by_row = {
                row_index: SemanticTableMapper._entity_id(
                    entity_class=plan.record_class,
                    identity_scope=record_class.identity_scope,
                    source_name=source.source_name,
                    business_value=primary_value,
                )
                for primary_value, row_index in primary_rows.items()
            }
            declaration = self.catalog.declaration_for(source.source_name)
            source_aliases = {source.source_name}
            if declaration is not None:
                source_aliases.add(declaration.table_name)

            for key_columns in [
                record_class.identity_keys,
                *record_class.alternate_keys,
            ]:
                key_types = _key_datatypes(
                    attributes,
                    key_columns,
                    source_name=source.source_name,
                )
                rows_by_key = validate_unique_key(
                    source.records,
                    columns=key_columns,
                    datatypes=key_types,
                    key_name=(
                        f"{'primary' if key_columns == record_class.identity_keys else 'alternate'} "
                        f"key {plan.record_class}({', '.join(key_columns)})"
                    ),
                )
                for key_value, row_index in rows_by_key.items():
                    target_id = primary_id_by_row[row_index]
                    for source_alias in source_aliases:
                        lookup = (
                            source_alias,
                            plan.record_class,
                            tuple(key_columns),
                            key_value,
                        )
                        previous = self._target_key_index.get(lookup)
                        if previous is not None and previous != target_id:
                            raise ValueError(
                                f"target key collision for {lookup[:3]!r}: "
                                f"{key_value} resolves to both {previous!r} and "
                                f"{target_id!r}"
                            )
                        self._target_key_index[lookup] = target_id

    def map_source(self, source: SourceRecord) -> ConstructionBatch:
        from kg_construction.application.semantic_table_mapper import SemanticTableMapper

        if (
            self.require_declarations
            and self.catalog.declaration_for(source.source_name) is None
        ):
            raise ValueError(
                f"structured source {source.source_name!r} has no authoritative declaration"
            )
        plan = self._prepared_plans.get(source.source_id) or self.catalog.resolved_plan(
            source,
            require_explicit_vocabulary_bindings=self.require_declarations,
        )
        declaration = self.catalog.declaration_for(source.source_name)
        if declaration is not None and self._is_pure_join(declaration, source):
            return self._map_pure_join(plan, declaration, source)
        return SemanticTableMapper(
            plan,
            target_key_index=self._target_key_index,
        ).map_source(source)

    @staticmethod
    def _is_pure_join(
        declaration: TableDeclaration,
        source: SourceRecord,
    ) -> bool:
        if declaration.table_kind == "edge":
            return True
        if declaration.table_kind != "auto" or len(declaration.foreign_keys) != 2:
            return False
        foreign_columns = {
            column for foreign_key in declaration.foreign_keys for column in foreign_key.columns
        }
        observed_columns = {column for row in source.records for column in row}
        return observed_columns <= foreign_columns | set(declaration.primary_key)

    def _map_pure_join(
        self,
        plan: ResolvedModelingPlan,
        declaration: TableDeclaration,
        source: SourceRecord,
    ) -> ConstructionBatch:
        from kg_construction.application.semantic_table_mapper import SemanticTableMapper

        if len(plan.foreign_keys) != 2:
            raise ValueError(
                f"pure join table {declaration.table_name!r} must resolve exactly two foreign keys"
            )
        target_resolver = SemanticTableMapper(
            plan,
            target_key_index=self._target_key_index,
        )
        relations: list[RelationCandidate] = []
        missing_targets = 0
        for row_index, row in enumerate(source.records):
            endpoint_ids: list[str] = []
            for foreign_key in plan.foreign_keys:
                if any(row.get(column) in (None, "") for column in foreign_key.columns):
                    endpoint_ids = []
                    break
                raw_values = [
                    row[column] for column in foreign_key.columns
                ]
                target_id = target_resolver._foreign_key_target(
                    foreign_key,
                    raw_values,
                    source,
                )
                if target_id is None:
                    endpoint_ids = []
                    missing_targets += 1
                    break
                endpoint_ids.append(target_id)
            if len(endpoint_ids) != 2:
                continue
            relations.append(
                RelationCandidate.build(
                    source=endpoint_ids[0],
                    relation_type=(
                        declaration.association_relationship_name
                        or _association_relationship_name(declaration.table_name)
                    ),
                    target=endpoint_ids[1],
                    evidence=[row_evidence(source, row_index, row)],
                    extractor_name=self.name,
                    technique="pure_join_table_to_relationship",
                    properties={"source_table": declaration.table_name},
                )
            )
        return ConstructionBatch(
            source_records=[source],
            relations=relations,
            metadata={
                "mapper": self.name,
                "table_name": declaration.table_name,
                "table_kind": "edge",
                "missing_join_targets": missing_targets,
                "resolved_modeling_plans": {
                    plan.plan_id: plan.model_dump(mode="json")
                },
            },
        )


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


def _key_datatypes(
    attributes: dict[str, Any],
    columns: list[str],
    *,
    source_name: str,
) -> list[JsonType]:
    missing = [column for column in columns if column not in attributes]
    if missing:
        raise ValueError(
            f"declared key for {source_name!r} references unknown column(s) "
            f"{missing!r}"
        )
    return [attributes[column].datatype or "string" for column in columns]


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return list(value)


def _metadata_alternate_keys(metadata: JsonDict) -> list[list[str]]:
    """Read alternate/unique keys from common database-catalog payload shapes."""

    values = metadata.get(
        "alternate_keys",
        metadata.get("unique_constraints", []),
    ) or []
    result: list[list[str]] = []
    for value in values:
        if isinstance(value, dict):
            columns = value.get(
                "columns",
                value.get("column_names", value.get("columnReference", [])),
            )
            result.append(_as_list(columns))
        else:
            result.append(_as_list(value))
    return result


def _record_type(source_name: str) -> str:
    stem = source_name.rsplit("/", 1)[-1].split(".")[0]
    tokens = [token for token in stem.replace("-", "_").split("_") if token]
    singular = _singularize(tokens[-1]) if tokens else "Record"
    tokens = tokens[:-1] + [singular] if tokens else ["Record"]
    return "".join(token[:1].upper() + token[1:] for token in tokens) or "Record"


def _catalog_name(source_name: str) -> str:
    stem = source_name.replace("\\", "/").rsplit("/", 1)[-1].rsplit(".", 1)[0]
    return "".join(character for character in stem.casefold() if character.isalnum())


def _association_relationship_name(table_name: str) -> str:
    stem = table_name.replace("\\", "/").rsplit("/", 1)[-1].rsplit(".", 1)[0]
    value = re.sub(r"[^a-z0-9]+", "_", stem.casefold()).strip("_")
    if value.endswith("ies"):
        value = value[:-3] + "y"
    elif value.endswith("s") and not value.endswith("ss"):
        value = value[:-1]
    return value or "associated_with"


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
        column: fk
        for fk in declaration.foreign_keys
        for column in fk.columns
    }
    alternate_columns = {
        column for key in declaration.alternate_keys for column in key
    }
    identifier_columns = set(declaration.primary_key) | alternate_columns

    for assignment in assignments:
        column = assignment.column
        mapping = declaration.column_mappings.get(column)
        if mapping is not None:
            if mapping.role is not None:
                assignment.role = mapping.role
            if mapping.materialization is not None:
                assignment.materialization = mapping.materialization.value
            if mapping.target is not None:
                assignment.reference_type = mapping.target
            if mapping.relationship_name is not None:
                assignment.relationship_type = mapping.relationship_name
            if mapping.datatype is not None:
                assignment.datatype = mapping.datatype

        if declaration.primary_key and column in declaration.primary_key:
            assignment.role = ColumnRole.IDENTIFIER
            assignment.materialization = "identity"
            assignment.reference_type = None
        elif (
            declaration.primary_key
            and assignment.materialization == "identity"
            and column not in declaration.primary_key
        ):
            # A profiler-proposed identity cannot compete with an authoritative
            # declared primary key.  Retain it as a lexical property.
            assignment.materialization = "property"
            assignment.datatype = declaration.datatypes.get(column, "string")
        elif column in fk_columns:
            fk = fk_target.get(column)
            assignment.role = ColumnRole.ENTITY_REFERENCE
            assignment.materialization = "edge"
            assignment.reference_type = _record_type(fk.target_table) if fk else assignment.reference_type
            if fk is not None and fk.relationship_name:
                assignment.relationship_type = fk.relationship_name
        if column in declaration.datatypes:
            assignment.datatype = declaration.datatypes[column]
        elif column in identifier_columns and (
            mapping is None or mapping.datatype is None
        ):
            # Keys are lexical by default, even when every observed value looks
            # numeric; only an explicit datatype may coerce them.
            assignment.datatype = "string"

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
    "DeclaredColumnMapping",
    "DeclaredForeignKey",
    "SourceCatalog",
    "TableDeclaration",
]
