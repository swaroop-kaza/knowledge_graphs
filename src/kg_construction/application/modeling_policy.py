"""Declarative modelling policy that drives structured graph construction.

A ``ModelingPolicy`` is the per-source contract the profiler proposes and the
mapper executes. It binds each column to a
:class:`~kg_construction.domain.modeling.ColumnRole`, declares how categorical
values are canonicalised and which controlled vocabularies apply, sets the
thresholds that decide node-vs-property promotion (including a supernode guard),
and declares the relationships that must be formed.

It lowers into an :class:`~kg_construction.domain.modeling.OntologySchema` (and
from there into the existing ``GraphSchema``) via
:meth:`ModelingPolicy.to_ontology_schema`, so a single policy drives mapping,
ontology, and validation.
"""

from __future__ import annotations

import re
import unicodedata

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from kg_construction.domain.modeling import (
    Cardinality,
    ColumnRole,
    ControlledVocabulary,
    OntologyClass,
    OntologyPredicate,
    OntologySchema,
)
from kg_construction.domain.schema import JsonType

_WHITESPACE_RE = re.compile(r"\s+")

_ROLE_DEFAULT_DATATYPE: dict[ColumnRole, JsonType] = {
    ColumnRole.MEASURE: "number",
    ColumnRole.TEMPORAL: "datetime",
    ColumnRole.FREE_TEXT: "string",
    ColumnRole.IDENTIFIER: "string",
}


class PromotionThresholds(BaseModel):
    """Rules that decide whether a value becomes a shared node or stays a property.

    Defaults are conservative and grounded in table-profiling practice: a column
    is categorical-like when its distinct/row ratio is low and its absolute
    distinct count is bounded; very low-cardinality dense values (e.g. booleans)
    stay properties to avoid supernodes; long strings are free text, not
    categories.
    """

    model_config = ConfigDict(extra="forbid")

    promote_categoricals_to_nodes: bool = True
    categorical_max_distinct_ratio: float = Field(default=0.5, ge=0.0, le=1.0)
    categorical_max_distinct_absolute: int = Field(default=1000, ge=1)
    min_rows_for_ratio: int = Field(default=20, ge=1)
    keep_as_property_max_distinct: int = Field(default=2, ge=0)
    free_text_min_avg_length: int = Field(default=40, ge=1)
    supernode_fanin_warning: int = Field(default=10000, ge=1)
    supernode_policy: str = Field(default="warn")  # warn | keep_property | allow

    @field_validator("supernode_policy")
    @classmethod
    def _known_policy(cls, value: str) -> str:
        allowed = {"warn", "keep_property", "allow"}
        if value not in allowed:
            raise ValueError(f"supernode_policy must be one of {sorted(allowed)}, got {value!r}")
        return value


class CanonicalizationRule(BaseModel):
    """How a categorical/reference surface value is normalised for node identity.

    Applying the rule to two surface forms of the same real value must yield the
    same string, so both collapse into one :class:`ReferenceNode`.
    """

    model_config = ConfigDict(extra="forbid")

    unicode_nfkc: bool = True
    trim: bool = True
    collapse_whitespace: bool = True
    lowercase: bool = False
    casefold: bool = False
    strip_characters: str | None = None
    aliases: dict[str, str] = Field(default_factory=dict)

    def apply(self, value: str) -> str:
        """Return the canonical form of ``value`` under this rule."""

        result = value
        if self.unicode_nfkc:
            result = unicodedata.normalize("NFKC", result)
        if self.strip_characters:
            result = result.strip(self.strip_characters)
        if self.collapse_whitespace:
            result = _WHITESPACE_RE.sub(" ", result)
        if self.trim:
            result = result.strip()
        if self.casefold:
            result = result.casefold()
        elif self.lowercase:
            result = result.lower()
        return self.aliases.get(result, result)


class ColumnPolicy(BaseModel):
    """The declared modelling decision for a single source column."""

    model_config = ConfigDict(extra="forbid")

    column: str
    role: ColumnRole
    reference_type: str | None = None
    relationship_type: str | None = None
    property_name: str | None = None
    datatype: JsonType | None = None
    cardinality: Cardinality | None = None
    controlled_vocabulary: str | None = None
    canonicalization: CanonicalizationRule | None = None
    description: str | None = None

    @model_validator(mode="after")
    def _check_role_contract(self) -> ColumnPolicy:
        if self.role.becomes_node and not self.reference_type:
            raise ValueError(
                f"column {self.column!r} has role {self.role.value!r} and must declare a reference_type"
            )
        if not self.role.becomes_node and self.reference_type is not None:
            raise ValueError(
                f"column {self.column!r} has role {self.role.value!r} and must not declare a reference_type"
            )
        return self

    def target_relationship(self) -> str:
        """Return the edge label for a node-promoted column."""

        if self.relationship_type:
            return self.relationship_type
        reference = self.reference_type or self.column
        return f"has_{reference.lower()}"

    def target_property(self) -> str:
        """Return the property name for a property-valued column."""

        return self.property_name or self.column


class RelationshipDeclaration(BaseModel):
    """An explicitly declared relationship (R2RML referencing-object-map analogue)."""

    model_config = ConfigDict(extra="forbid")

    name: str
    source_type: str
    target_type: str
    cardinality: Cardinality = Field(default_factory=Cardinality.zero_or_many)
    source_columns: list[str] = Field(default_factory=list)
    target_columns: list[str] = Field(default_factory=list)
    via_reference: bool = False
    description: str | None = None


class ModelingPolicy(BaseModel):
    """The per-table contract for turning structured records into a graph."""

    model_config = ConfigDict(extra="forbid")

    table_name: str = "records"
    record_type: str = "Record"
    id_columns: list[str] = Field(default_factory=list)
    columns: list[ColumnPolicy] = Field(default_factory=list)
    relationships: list[RelationshipDeclaration] = Field(default_factory=list)
    controlled_vocabularies: list[ControlledVocabulary] = Field(default_factory=list)
    thresholds: PromotionThresholds = Field(default_factory=PromotionThresholds)
    default_canonicalization: CanonicalizationRule = Field(default_factory=CanonicalizationRule)
    schema_version: str = "v1"

    @model_validator(mode="after")
    def _unique_columns(self) -> ModelingPolicy:
        seen = [policy.column for policy in self.columns]
        duplicates = sorted({column for column in seen if seen.count(column) > 1})
        if duplicates:
            raise ValueError(f"duplicate column policies for {duplicates}")
        vocab_names = [item.name for item in self.controlled_vocabularies]
        vocab_duplicates = sorted({name for name in vocab_names if vocab_names.count(name) > 1})
        if vocab_duplicates:
            raise ValueError(f"duplicate controlled vocabularies for {vocab_duplicates}")
        return self

    def column_policy(self, column: str) -> ColumnPolicy | None:
        """Return the declared policy for ``column`` if present."""

        return next((policy for policy in self.columns if policy.column == column), None)

    def vocabulary(self, name: str) -> ControlledVocabulary | None:
        """Return the controlled vocabulary named ``name`` if present."""

        return next((item for item in self.controlled_vocabularies if item.name == name), None)

    def reference_types(self) -> set[str]:
        """Return the set of reference/dimension classes this policy introduces."""

        return {
            policy.reference_type
            for policy in self.columns
            if policy.reference_type is not None
        }

    def canonicalization_for(self, column: str) -> CanonicalizationRule:
        """Return the effective canonicalisation rule for ``column``."""

        policy = self.column_policy(column)
        if policy is not None and policy.canonicalization is not None:
            return policy.canonicalization
        return self.default_canonicalization

    def to_ontology_schema(self) -> OntologySchema:
        """Derive the ontology (classes, predicates, cardinality) from this policy."""

        classes: dict[str, OntologyClass] = {
            self.record_type: OntologyClass(
                name=self.record_type,
                role="record",
                key_properties=list(self.id_columns),
                source_table=self.table_name,
            )
        }
        predicates: list[OntologyPredicate] = []
        for policy in self.columns:
            if policy.role.becomes_node and policy.reference_type is not None:
                classes.setdefault(
                    policy.reference_type,
                    OntologyClass(name=policy.reference_type, role="reference"),
                )
                predicates.append(
                    OntologyPredicate(
                        name=policy.target_relationship(),
                        domain=self.record_type,
                        range=policy.reference_type,
                        kind="object",
                        cardinality=policy.cardinality or Cardinality.zero_or_one(),
                        source_columns=[policy.column],
                        description=policy.description,
                    )
                )
            elif policy.role.becomes_property:
                datatype: JsonType = policy.datatype or _ROLE_DEFAULT_DATATYPE.get(
                    policy.role, "string"
                )
                predicates.append(
                    OntologyPredicate(
                        name=policy.target_property(),
                        domain=self.record_type,
                        range=datatype,
                        kind="datatype",
                        cardinality=policy.cardinality or Cardinality.zero_or_one(),
                        source_columns=[policy.column],
                        description=policy.description,
                    )
                )
        for declaration in self.relationships:
            classes.setdefault(
                declaration.target_type,
                OntologyClass(name=declaration.target_type, role="reference" if declaration.via_reference else "record"),
            )
            predicates.append(
                OntologyPredicate(
                    name=declaration.name,
                    domain=declaration.source_type,
                    range=declaration.target_type,
                    kind="object",
                    cardinality=declaration.cardinality,
                    source_columns=list(declaration.source_columns),
                    target_columns=list(declaration.target_columns),
                    description=declaration.description,
                )
            )
        return OntologySchema(
            schema_id=self.table_name,
            version=self.schema_version,
            classes=sorted(classes.values(), key=lambda item: item.name),
            predicates=predicates,
            controlled_vocabularies=list(self.controlled_vocabularies),
        )


def default_modeling_policy(
    *,
    table_name: str = "records",
    record_type: str = "Record",
) -> ModelingPolicy:
    """Return a conservative, empty-column policy with default thresholds."""

    return ModelingPolicy(table_name=table_name, record_type=record_type)


__all__ = [
    "CanonicalizationRule",
    "ColumnPolicy",
    "ModelingPolicy",
    "PromotionThresholds",
    "RelationshipDeclaration",
    "default_modeling_policy",
]
