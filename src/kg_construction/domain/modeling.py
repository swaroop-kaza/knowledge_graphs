"""Core modelling vocabulary: roles, reference nodes, and ontology schema.

Defines the concepts the modelling pipeline is built on:

- ``ColumnRole`` -- the semantic role a source column plays in the graph.
- ``Cardinality`` -- explicit min/max multiplicity for a predicate.
- ``ControlledVocabulary`` -- a closed or open set of permissible categorical values.
- ``ReferenceNode`` -- a shared, deterministically-identified node that a
  categorical or reference value collapses into, so a value is stored once rather
  than repeated as a literal on every row.
- ``OntologyClass`` / ``OntologyPredicate`` / ``OntologySchema`` -- the declared
  classes and predicates with domain, range, and cardinality.

``OntologySchema`` lowers into the existing ``GraphSchema`` via
:meth:`OntologySchema.to_graph_schema`, so the shared SHACL/OWL emitters and
validators are reused. See ``docs/knowledge_graph_modelling.md``.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from kg_construction.domain.models import normalize_text, stable_id
from kg_construction.domain.schema import (
    GraphSchema,
    JsonType,
    NodeTypeDefinition,
    PropertyDefinition,
    RelationshipTypeDefinition,
)

_JSON_TYPES: frozenset[str] = frozenset(
    {"string", "number", "integer", "boolean", "object", "array", "date", "datetime"}
)


class ColumnRole(StrEnum):
    """Semantic role a source column plays when it becomes part of the graph.

    The role is the semantic decision; physical materialization (node vs literal)
    is resolved separately in the ``ResolvedModelingPlan``. See
    ``docs/knowledge_graph_modelling.md``.
    """

    IDENTIFIER = "identifier"
    """Uniquely identifies the row's entity (primary key). Becomes node identity."""

    ENTITY_REFERENCE = "entity_reference"
    """A foreign reference to another entity. Becomes a typed relationship edge."""

    CATEGORICAL_DIMENSION = "categorical_dimension"
    """A low-cardinality dimension value. Promoted to a shared ``ReferenceNode``."""

    MEASURE = "measure"
    """A numeric quantity/measurement. Stays a node property (a literal value)."""

    TEMPORAL = "temporal"
    """A date/time/interval. Stays a property (or an occurrence qualifier)."""

    FREE_TEXT = "free_text"
    """Unbounded natural-language text. Stays a property; may seed extraction."""

    IGNORE = "ignore"
    """Excluded from the graph (noise, internal columns, prohibited data)."""

    @property
    def becomes_node(self) -> bool:
        """True when a value in this role collapses into a shared node."""

        return self in {ColumnRole.CATEGORICAL_DIMENSION, ColumnRole.ENTITY_REFERENCE}

    @property
    def becomes_property(self) -> bool:
        """True when a value in this role stays a literal property."""

        return self in {ColumnRole.MEASURE, ColumnRole.TEMPORAL, ColumnRole.FREE_TEXT}


class Cardinality(BaseModel):
    """Explicit multiplicity for a predicate (SHACL ``sh:minCount``/``sh:maxCount``).

    ``max_count=None`` means unbounded (``*``).
    """

    model_config = ConfigDict(extra="forbid")

    min_count: int = Field(default=0, ge=0)
    max_count: int | None = Field(default=1, ge=1)

    @model_validator(mode="after")
    def _check_bounds(self) -> Cardinality:
        if self.max_count is not None and self.min_count > self.max_count:
            raise ValueError(
                f"min_count ({self.min_count}) cannot exceed max_count ({self.max_count})"
            )
        return self

    @property
    def required(self) -> bool:
        """True when at least one value must be present."""

        return self.min_count >= 1

    @property
    def functional(self) -> bool:
        """True when at most one value is allowed (a single-valued predicate)."""

        return self.max_count == 1

    @property
    def label(self) -> str:
        """Human-readable multiplicity label."""

        if self.min_count == 1 and self.max_count == 1:
            return "exactly_one"
        if self.min_count == 0 and self.max_count == 1:
            return "zero_or_one"
        if self.min_count == 1 and self.max_count is None:
            return "one_or_many"
        if self.min_count == 0 and self.max_count is None:
            return "zero_or_many"
        upper = "*" if self.max_count is None else str(self.max_count)
        return f"{self.min_count}..{upper}"

    @classmethod
    def exactly_one(cls) -> Cardinality:
        return cls(min_count=1, max_count=1)

    @classmethod
    def zero_or_one(cls) -> Cardinality:
        return cls(min_count=0, max_count=1)

    @classmethod
    def zero_or_many(cls) -> Cardinality:
        return cls(min_count=0, max_count=None)

    @classmethod
    def one_or_many(cls) -> Cardinality:
        return cls(min_count=1, max_count=None)


class ControlledVocabulary(BaseModel):
    """A named set of permissible categorical values (LinkML enum / SHACL sh:in).

    ``closed=True`` means values outside the set are not permitted and must be
    quarantined or reviewed rather than silently admitted.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    permissible_values: list[str] = Field(default_factory=list)
    closed: bool = True
    aliases: dict[str, str] = Field(default_factory=dict)
    description: str | None = None

    @field_validator("name")
    @classmethod
    def _non_empty_name(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("controlled vocabulary name must not be empty")
        return value

    def canonical_value(self, value: str) -> str | None:
        """Return the permitted canonical value, or ``None`` when not permitted.

        An open vocabulary (``closed=False``) always returns the alias-resolved
        value. A closed vocabulary returns ``None`` for values outside the set.
        """

        resolved = self.aliases.get(value, value)
        if not self.closed:
            return resolved
        return resolved if resolved in self.permissible_values else None

    def is_permitted(self, value: str) -> bool:
        """True when ``value`` (after alias resolution) is admissible."""

        return self.canonical_value(value) is not None


class ReferenceNode(BaseModel):
    """A shared, deduplicated node that a categorical/reference value becomes.

    The node identity is a deterministic function of ``(reference_type,
    canonical value)`` only, so the same real value seen on 10,000 rows resolves
    to *one* node. ``raw_values`` retains the observed surface forms for
    provenance without affecting identity.
    """

    model_config = ConfigDict(extra="forbid")

    reference_type: str
    value: str
    raw_values: list[str] = Field(default_factory=list)

    @property
    def node_id(self) -> str:
        """Deterministic, display-independent node identity."""

        return stable_id("ref", self.reference_type, self.value)

    @property
    def labels(self) -> list[str]:
        """Graph labels for the reference node."""

        base = "Reference"
        return [base] if self.reference_type == base else [base, self.reference_type]

    @classmethod
    def build(
        cls,
        *,
        reference_type: str,
        value: str,
        raw_value: str | None = None,
        normalize: bool = True,
    ) -> ReferenceNode:
        """Build a reference node, normalising ``value`` for stable identity."""

        canonical = normalize_text(value) if normalize else value
        raw_values = sorted({raw_value if raw_value is not None else value})
        return cls(reference_type=reference_type, value=canonical, raw_values=raw_values)


class OntologyClass(BaseModel):
    """A declared node class with a modelling role."""

    model_config = ConfigDict(extra="forbid")

    name: str
    role: str = Field(default="record")  # record | reference | event
    key_properties: list[str] = Field(default_factory=list)
    source_table: str | None = None
    description: str | None = None

    @field_validator("role")
    @classmethod
    def _known_role(cls, value: str) -> str:
        allowed = {"record", "reference", "event"}
        if value not in allowed:
            raise ValueError(f"class role must be one of {sorted(allowed)}, got {value!r}")
        return value


class OntologyPredicate(BaseModel):
    """A declared predicate with domain, range, and cardinality.

    ``kind='object'`` is a relationship between two classes (the ``range`` is a
    class name). ``kind='datatype'`` is a literal-valued property (the ``range``
    is a JSON/XSD datatype name).
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    domain: str
    range: str
    kind: str = Field(default="object")  # object | datatype
    cardinality: Cardinality = Field(default_factory=Cardinality)
    source_columns: list[str] = Field(default_factory=list)
    target_columns: list[str] = Field(default_factory=list)
    description: str | None = None

    @field_validator("kind")
    @classmethod
    def _known_kind(cls, value: str) -> str:
        if value not in {"object", "datatype"}:
            raise ValueError(f"predicate kind must be 'object' or 'datatype', got {value!r}")
        return value

    @model_validator(mode="after")
    def _check_datatype_range(self) -> OntologyPredicate:
        if self.kind == "datatype" and self.range not in _JSON_TYPES:
            raise ValueError(
                f"datatype predicate range must be one of {sorted(_JSON_TYPES)}, got {self.range!r}"
            )
        return self


class OntologySchema(BaseModel):
    """The modelling-intent ontology: classes, predicates, and vocabularies.

    Lowers into the existing :class:`GraphSchema` via :meth:`to_graph_schema`,
    so downstream SHACL/OWL emission and validation are reused unchanged.
    """

    model_config = ConfigDict(extra="forbid")

    schema_id: str = "default"
    version: str = "v1"
    namespace: str = "urn:kg:schema:"
    classes: list[OntologyClass] = Field(default_factory=list)
    predicates: list[OntologyPredicate] = Field(default_factory=list)
    controlled_vocabularies: list[ControlledVocabulary] = Field(default_factory=list)

    def class_names(self) -> set[str]:
        return {item.name for item in self.classes}

    def reference_classes(self) -> list[OntologyClass]:
        """Return classes that represent shared categorical/reference dimensions."""

        return [item for item in self.classes if item.role == "reference"]

    def unresolved_predicate_references(self) -> list[str]:
        """Return object predicates whose domain/range class is not declared.

        A soft consistency check (does not raise); useful for review before promotion.
        """

        known = self.class_names()
        issues: list[str] = []
        for predicate in self.predicates:
            if predicate.domain not in known:
                issues.append(f"{predicate.name}: domain class {predicate.domain!r} is not declared")
            if predicate.kind == "object" and predicate.range not in known:
                issues.append(f"{predicate.name}: range class {predicate.range!r} is not declared")
        return issues

    def to_graph_schema(self) -> GraphSchema:
        """Lower this ontology into the existing GraphSchema registry model.

        Datatype predicates become node properties on their domain class;
        object predicates become relationship types with source/target types.
        Cardinality ``min_count >= 1`` maps to ``required``; single-valued object
        predicates are recorded but full min/max on relationships is retained
        only on this ontology (the legacy SHACL emitter caps property maxCount).
        """

        properties_by_class: dict[str, list[PropertyDefinition]] = {
            item.name: [] for item in self.classes
        }
        relationship_types: list[RelationshipTypeDefinition] = []
        for predicate in self.predicates:
            if predicate.kind == "datatype":
                value_type: JsonType = (
                    predicate.range if predicate.range in _JSON_TYPES else "string"  # type: ignore[assignment]
                )
                properties_by_class.setdefault(predicate.domain, []).append(
                    PropertyDefinition(
                        name=predicate.name,
                        value_type=value_type,
                        required=predicate.cardinality.required,
                        unique=predicate.cardinality.functional and predicate.cardinality.required,
                        source_column=(predicate.source_columns[0] if predicate.source_columns else None),
                        description=predicate.description,
                    )
                )
            else:
                relationship_types.append(
                    RelationshipTypeDefinition(
                        name=predicate.name,
                        source_types=[predicate.domain],
                        target_types=[predicate.range],
                        source_columns=list(predicate.source_columns),
                        target_columns=list(predicate.target_columns),
                        description=predicate.description,
                    )
                )

        node_types = [
            NodeTypeDefinition(
                name=item.name,
                properties=properties_by_class.get(item.name, []),
                primary_keys=list(item.key_properties),
                source_table=item.source_table,
                description=item.description,
            )
            for item in self.classes
        ]
        return GraphSchema(
            schema_id=self.schema_id,
            version=self.version,
            namespace=self.namespace,
            node_types=node_types,
            relationship_types=relationship_types,
            metadata={
                "derived_from": "OntologySchema",
                "reference_classes": sorted(item.name for item in self.reference_classes()),
                "predicate_cardinality": {
                    predicate.name: predicate.cardinality.label for predicate in self.predicates
                },
                "controlled_vocabularies": sorted(item.name for item in self.controlled_vocabularies),
            },
        )


__all__ = [
    "Cardinality",
    "ColumnRole",
    "ControlledVocabulary",
    "OntologyClass",
    "OntologyPredicate",
    "OntologySchema",
    "ReferenceNode",
]
