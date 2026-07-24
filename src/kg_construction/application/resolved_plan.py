"""The single resolved modelling contract that drives graph construction.

A :class:`ResolvedModelingPlan` is the *only* accepted input to promoted-graph
construction. Profiling may propose a plan, but every schema artifact and every
constructed graph is derived from a resolved plan, so all of them agree by
construction.

The plan keeps **semantic role** (what a column means) and **physical
materialization** (whether it becomes a node, an edge, or a literal property)
as separate, explicit decisions — a categorical dimension can be a shared node
*or* a scalar-enumeration property, and that choice, not the role, drives whether
its predicate is an object property or a datatype property.

From one plan it projects six views that must agree on class, property,
relationship, identity, datatype, and cardinality definitions:

* the property-graph schema (via the canonical :class:`OntologySchema`),
* an RDF 1.1 / JSON-LD ``@context``,
* an OWL 2 RL ontology,
* SHACL 1.0 shapes,
* Neo4j uniqueness/existence constraints, and
* evaluation expectations.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from kg_construction.application.modeling_policy import CanonicalizationRule
from kg_construction.domain.modeling import (
    Cardinality,
    ColumnRole,
    ControlledVocabulary,
    OntologyClass,
    OntologyPredicate,
    OntologySchema,
)
from kg_construction.domain.models import stable_id
from kg_construction.domain.schema import GraphSchema, JsonType

# JsonType -> XSD local name, for JSON-LD @type and OWL/SHACL ranges.
_XSD_LOCAL: dict[str, str] = {
    "string": "string",
    "integer": "integer",
    "number": "decimal",
    "boolean": "boolean",
    "date": "date",
    "datetime": "dateTime",
    "object": "string",
    "array": "string",
}
_XSD = "http://www.w3.org/2001/XMLSchema#"


def _xsd_iri(datatype: str) -> str:
    return _XSD + _XSD_LOCAL.get(datatype, "string")


class Materialization(StrEnum):
    """The physical decision: how a resolved column is represented in the graph."""

    IDENTITY = "identity"
    """Becomes the owning record's identity (primary key)."""

    NODE = "node"
    """Collapses into a shared node, joined by an object predicate."""

    EDGE = "edge"
    """Becomes a typed relationship to another entity (foreign key)."""

    PROPERTY = "property"
    """Stays a literal value on its owning node (datatype predicate)."""

    EXCLUDED = "excluded"
    """Not represented in the graph."""

    @property
    def is_object(self) -> bool:
        """True when this materialization produces an object predicate (node/edge)."""

        return self in {Materialization.NODE, Materialization.EDGE}


class IdentityRole(StrEnum):
    """The key role a column plays in identity resolution."""

    PRIMARY = "primary"
    ALTERNATE = "alternate"
    COMPOSITE = "composite"
    FOREIGN = "foreign"
    NONE = "none"


class IdentityScope(StrEnum):
    """How far an identity is unique."""

    SOURCE = "source"
    """Unique within one source/file only (conservative default)."""

    GLOBAL = "global"
    """Unique across all sources (a governed global key)."""

    CROSS_SOURCE = "cross_source"
    """Resolved across sources by an explicit resolver."""


class ClassKind(StrEnum):
    """The doctrine class distinction."""

    ENTITY = "entity"
    """A business entity: exists independently, has attributes/relationships."""

    CONCEPT = "concept"
    """A governed concept from a vocabulary (SKOS), keyed by scheme + code."""

    EVENT = "event"
    """An occurrence/association carrying its own attributes."""

    ASSOCIATION = "association"
    """A qualified n-ary relationship with identity, attributes, or validity."""


class OnMissingTarget(StrEnum):
    """What to do when a foreign key has no resolvable target."""

    SKIP = "skip"
    """Drop the edge; never fabricate a placeholder entity."""

    QUARANTINE = "quarantine"
    """Emit a quarantined placeholder the policy explicitly permits."""

    ERROR = "error"
    """Fail construction."""


class ReviewState(StrEnum):
    PROPOSED = "proposed"
    RESOLVED = "resolved"
    APPROVED = "approved"
    PROMOTED = "promoted"


_CLASS_ROLE = {
    ClassKind.ENTITY: "record",
    ClassKind.CONCEPT: "reference",
    ClassKind.EVENT: "event",
    ClassKind.ASSOCIATION: "event",
}


class ResolvedClass(BaseModel):
    """A resolved class: a business entity, a governed concept, or an event."""

    model_config = ConfigDict(extra="forbid")

    name: str
    kind: ClassKind
    identity_keys: list[str] = Field(default_factory=list)
    alternate_keys: list[list[str]] = Field(default_factory=list)
    identity_scope: IdentityScope = IdentityScope.SOURCE
    concept_scheme: str | None = None
    source_table: str | None = None
    description: str | None = None

    @property
    def composite(self) -> bool:
        """True when the primary identity is a composite (multi-column) key."""

        return len(self.identity_keys) > 1

    @property
    def ontology_role(self) -> str:
        return _CLASS_ROLE[self.kind]

    @model_validator(mode="after")
    def _concept_needs_scheme(self) -> ResolvedClass:
        if self.kind == ClassKind.CONCEPT and not self.concept_scheme:
            raise ValueError(f"concept class {self.name!r} must declare a concept_scheme")
        return self


class ResolvedProperty(BaseModel):
    """A resolved literal property (datatype predicate) on a class."""

    model_config = ConfigDict(extra="forbid")

    class_name: str
    name: str
    datatype: JsonType
    nullable: bool = True
    cardinality: Cardinality = Field(default_factory=Cardinality.zero_or_one)
    source_column: str | None = None
    controlled_vocabulary: str | None = None
    provenance_required: bool = True


class ResolvedRelationship(BaseModel):
    """A resolved relationship (object predicate) with direction, domain, and range."""

    model_config = ConfigDict(extra="forbid")

    name: str
    domain: str
    range: str
    cardinality: Cardinality = Field(default_factory=Cardinality.zero_or_one)
    via: str = "foreign_key"  # foreign_key | categorical | association | declared
    source_columns: list[str] = Field(default_factory=list)
    target_columns: list[str] = Field(default_factory=list)
    controlled_vocabulary: str | None = None
    on_missing_target: OnMissingTarget = OnMissingTarget.SKIP
    provenance_required: bool = True
    description: str | None = None


class ForeignKeyResolution(BaseModel):
    """A resolved foreign key linking a source class to a target class key."""

    model_config = ConfigDict(extra="forbid")

    columns: list[str]
    source_class: str
    target_class: str
    target_key: list[str] = Field(default_factory=list)
    target_key_datatypes: list[JsonType] = Field(default_factory=list)
    """Datatypes of the referenced columns, in ``target_key`` order."""
    target_source_name: str | None = None
    """Source/table namespace that owns the referenced business entity."""
    target_identity_scope: IdentityScope = IdentityScope.SOURCE
    """Identity scope declared by the referenced class, not the child table."""
    relationship_name: str
    cardinality: Cardinality = Field(default_factory=Cardinality.zero_or_one)
    on_missing: OnMissingTarget = OnMissingTarget.SKIP
    self_referential: bool = False
    target_available: bool = False
    """True when the target table is present in the same catalog, so the edge can
    link to a real node; otherwise a missing target is skipped, never fabricated."""


class ResolvedAttribute(BaseModel):
    """The full resolution of one source column: role, materialization, and facets."""

    model_config = ConfigDict(extra="forbid")

    column: str
    semantic_role: ColumnRole
    materialization: Materialization
    target: str | None = None  # class name (node/edge) or property name (property)
    datatype: JsonType | None = None
    nullable: bool = True
    cardinality: Cardinality = Field(default_factory=Cardinality.zero_or_one)
    canonicalization: CanonicalizationRule | None = None
    controlled_vocabulary: str | None = None
    identity_role: IdentityRole = IdentityRole.NONE
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    review_state: ReviewState = ReviewState.RESOLVED
    provenance_required: bool = True


class EvaluationExpectations(BaseModel):
    """What the constructed graph must satisfy, derived from the same contract."""

    model_config = ConfigDict(extra="forbid")

    classes: list[str] = Field(default_factory=list)
    identity: dict[str, list[str]] = Field(default_factory=dict)
    required_properties: dict[str, list[str]] = Field(default_factory=dict)
    property_datatypes: dict[str, str] = Field(default_factory=dict)
    relationships: list[dict[str, str]] = Field(default_factory=list)
    categorical_duplication_rate: float = 0.0


class SchemaDefinitions(BaseModel):
    """A projection-independent view of the six agreement facets."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    classes: frozenset[str] = frozenset()
    properties: frozenset[tuple[str, str, str]] = frozenset()  # (class, property, datatype)
    relationships: frozenset[tuple[str, str, str]] = frozenset()  # (name, domain, range)
    identity: frozenset[tuple[str, tuple[str, ...]]] = frozenset()  # (class, keys)
    cardinality: frozenset[tuple[str, int, int]] = frozenset()  # (predicate, min, max_or_-1)

    def restricted_to(self, other: SchemaDefinitions) -> SchemaDefinitions:
        """Return this definition with only the facets ``other`` also populates.

        A projection that does not encode a facet (e.g. JSON-LD has no
        cardinality) is compared only on the facets it does carry.
        """

        return SchemaDefinitions(
            classes=self.classes if other.classes else frozenset(),
            properties=self.properties if other.properties else frozenset(),
            relationships=self.relationships if other.relationships else frozenset(),
            identity=self.identity if other.identity else frozenset(),
            cardinality=self.cardinality if other.cardinality else frozenset(),
        )


def _card_tuple(name: str, cardinality: Cardinality) -> tuple[str, int, int]:
    return (name, cardinality.min_count, -1 if cardinality.max_count is None else cardinality.max_count)


class ResolvedModelingPlan(BaseModel):
    """The resolved, construction-ready modelling contract for one table/source."""

    model_config = ConfigDict(extra="forbid")

    source_name: str
    table_name: str
    record_class: str
    namespace: str = "urn:kg:schema:"
    instance_namespace: str = "urn:kg:"
    version: str = "v1"
    rdf_profile: str = "rdf-1.1"
    shacl_version: str = "1.0"
    owl_profile: str = "owl2-rl"
    classes: list[ResolvedClass] = Field(default_factory=list)
    properties: list[ResolvedProperty] = Field(default_factory=list)
    relationships: list[ResolvedRelationship] = Field(default_factory=list)
    foreign_keys: list[ForeignKeyResolution] = Field(default_factory=list)
    attributes: list[ResolvedAttribute] = Field(default_factory=list)
    controlled_vocabularies: list[ControlledVocabulary] = Field(default_factory=list)
    provenance_required: bool = True
    review_state: ReviewState = ReviewState.RESOLVED

    @property
    def plan_id(self) -> str:
        """Deterministic identity for this resolved plan."""

        return stable_id("plan", self.source_name, self.table_name, self.version)

    @model_validator(mode="after")
    def _check_references(self) -> ResolvedModelingPlan:
        classes_by_name = {item.name: item for item in self.classes}
        names = set(classes_by_name)
        if self.record_class not in names:
            raise ValueError(f"record_class {self.record_class!r} is not among declared classes")
        for prop in self.properties:
            if prop.class_name not in names:
                raise ValueError(f"property {prop.name!r} references unknown class {prop.class_name!r}")
        for rel in self.relationships:
            missing = {rel.domain, rel.range} - names
            if missing:
                raise ValueError(f"relationship {rel.name!r} references unknown class(es) {sorted(missing)}")
        for foreign_key in self.foreign_keys:
            if foreign_key.target_key and len(foreign_key.columns) != len(
                foreign_key.target_key
            ):
                raise ValueError(
                    f"foreign key {foreign_key.columns!r} has target key "
                    f"{foreign_key.target_key!r} with different arity"
                )
            if foreign_key.target_key_datatypes and len(
                foreign_key.target_key_datatypes
            ) != len(foreign_key.target_key):
                raise ValueError(
                    f"foreign key target key {foreign_key.target_key!r} has "
                    "a different number of datatype declarations"
                )
            target = classes_by_name.get(foreign_key.target_class)
            if target is None or not foreign_key.target_key:
                continue
            declared_keys = {
                tuple(target.identity_keys),
                *(tuple(key) for key in target.alternate_keys),
            }
            declared_keys.discard(())
            if declared_keys and tuple(foreign_key.target_key) not in declared_keys:
                raise ValueError(
                    f"foreign key target key {foreign_key.target_key!r} is not a "
                    f"declared primary or alternate key of {target.name!r}"
                )
        return self

    def clazz(self, name: str) -> ResolvedClass:
        return next(item for item in self.classes if item.name == name)

    # -- Canonical intermediate -------------------------------------------------

    def to_ontology_schema(self) -> OntologySchema:
        """Lower into the canonical ontology every other projection derives from."""

        classes = [
            OntologyClass(
                name=item.name,
                role=item.ontology_role,
                key_properties=list(item.identity_keys),
                source_table=item.source_table,
                description=item.description,
            )
            for item in self.classes
        ]
        predicates: list[OntologyPredicate] = []
        for prop in self.properties:
            predicates.append(
                OntologyPredicate(
                    name=prop.name,
                    domain=prop.class_name,
                    range=prop.datatype,
                    kind="datatype",
                    cardinality=prop.cardinality,
                    source_columns=[prop.source_column] if prop.source_column else [],
                    controlled_vocabulary=prop.controlled_vocabulary,
                )
            )
        for rel in self.relationships:
            predicates.append(
                OntologyPredicate(
                    name=rel.name,
                    domain=rel.domain,
                    range=rel.range,
                    kind="object",
                    cardinality=rel.cardinality,
                    source_columns=list(rel.source_columns),
                    target_columns=list(rel.target_columns),
                    controlled_vocabulary=rel.controlled_vocabulary,
                    description=rel.description,
                )
            )
        return OntologySchema(
            schema_id=self.table_name,
            version=self.version,
            namespace=self.namespace,
            classes=sorted(classes, key=lambda item: item.name),
            predicates=predicates,
            controlled_vocabularies=list(self.controlled_vocabularies),
        )

    # -- Projections ------------------------------------------------------------

    def to_property_graph_schema(self) -> GraphSchema:
        """Project the property-graph schema (node/relationship type definitions)."""

        return self.to_ontology_schema().to_graph_schema()

    def to_jsonld_context(self) -> dict[str, object]:
        """Project an RDF 1.1 / JSON-LD ``@context`` for the same terms."""

        context: dict[str, object] = {"@version": 1.1, "@vocab": self.namespace, "kg": self.namespace}
        for item in self.classes:
            context[item.name] = {"@id": f"kg:{item.name}"}
        for prop in self.properties:
            context[prop.name] = {"@id": f"kg:{prop.name}", "@type": _xsd_iri(prop.datatype)}
        for rel in self.relationships:
            context[rel.name] = {"@id": f"kg:{rel.name}", "@type": "@id"}
        return {"@context": context}

    def to_owl2rl_turtle(self) -> str:
        """Project an OWL 2 RL ontology (classes, object/datatype properties)."""

        lines = [
            "@prefix owl: <http://www.w3.org/2002/07/owl#> .",
            "@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .",
            "@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .",
            "@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .",
            f"@prefix kg: <{self.namespace}> .",
            "",
            "kg:Ontology a owl:Ontology ;",
            f'  owl:versionInfo "{self.version}" ;',
            '  rdfs:comment "OWL 2 RL profile" .',
            "",
        ]
        for item in sorted(self.classes, key=lambda c: c.name):
            lines.append(f"kg:{item.name} a owl:Class .")
        lines.append("")
        for prop in self.properties:
            types = "owl:DatatypeProperty" + (
                ", owl:FunctionalProperty" if prop.cardinality.functional else ""
            )
            xsd_local = _XSD_LOCAL.get(prop.datatype, "string")
            lines.append(f"kg:{prop.name} a {types} ;")
            lines.append(f"  rdfs:domain kg:{prop.class_name} ;")
            lines.append(f"  rdfs:range xsd:{xsd_local} .")
        lines.append("")
        for rel in self.relationships:
            types = "owl:ObjectProperty" + (
                ", owl:FunctionalProperty" if rel.cardinality.functional else ""
            )
            lines.append(f"kg:{rel.name} a {types} ;")
            lines.append(f"  rdfs:domain kg:{rel.domain} ;")
            lines.append(f"  rdfs:range kg:{rel.range} .")
        return "\n".join(lines).rstrip() + "\n"

    def to_shacl_shapes(self) -> str:
        """Project SHACL 1.0 shapes for the same ontology."""

        from kg_construction.infrastructure.ontology.structured_ontology import (
            generate_shacl_shapes,
        )

        return generate_shacl_shapes(self.to_ontology_schema(), namespace=self.namespace)

    def to_neo4j_constraints(self) -> list[str]:
        """Project Neo4j uniqueness and existence constraints (Cypher)."""

        statements: list[str] = []
        for item in sorted(self.classes, key=lambda c: c.name):
            if not item.identity_keys:
                continue
            keys = ", ".join(f"n.{key}" for key in item.identity_keys)
            requirement = f"({keys}) IS NODE KEY" if item.composite else f"{keys} IS UNIQUE"
            statements.append(
                f"CREATE CONSTRAINT {item.name.lower()}_key IF NOT EXISTS "
                f"FOR (n:{item.name}) REQUIRE {requirement};"
            )
        for prop in self.properties:
            if prop.cardinality.required:
                statements.append(
                    f"CREATE CONSTRAINT {prop.class_name.lower()}_{prop.name.lower()}_exists "
                    f"IF NOT EXISTS FOR (n:{prop.class_name}) REQUIRE n.{prop.name} IS NOT NULL;"
                )
        return statements

    def to_evaluation_expectations(self) -> EvaluationExpectations:
        """Project the evaluation expectations the constructed graph must satisfy."""

        identity = {item.name: list(item.identity_keys) for item in self.classes if item.identity_keys}
        required_properties: dict[str, list[str]] = {}
        property_datatypes: dict[str, str] = {}
        for prop in self.properties:
            property_datatypes[f"{prop.class_name}.{prop.name}"] = prop.datatype
            if prop.cardinality.required:
                required_properties.setdefault(prop.class_name, []).append(prop.name)
        relationships = [
            {"name": rel.name, "domain": rel.domain, "range": rel.range, "cardinality": rel.cardinality.label}
            for rel in self.relationships
        ]
        return EvaluationExpectations(
            classes=sorted(item.name for item in self.classes),
            identity=identity,
            required_properties=required_properties,
            property_datatypes=property_datatypes,
            relationships=relationships,
        )

    # -- Agreement --------------------------------------------------------------

    def definitions(self) -> SchemaDefinitions:
        """Return the canonical six-facet definition set for this plan."""

        classes = frozenset(item.name for item in self.classes)
        properties = frozenset(
            (prop.class_name, prop.name, prop.datatype) for prop in self.properties
        )
        relationships = frozenset(
            (rel.name, rel.domain, rel.range) for rel in self.relationships
        )
        identity = frozenset(
            (item.name, tuple(item.identity_keys)) for item in self.classes if item.identity_keys
        )
        cardinality = frozenset(
            [_card_tuple(prop.name, prop.cardinality) for prop in self.properties]
            + [_card_tuple(rel.name, rel.cardinality) for rel in self.relationships]
        )
        return SchemaDefinitions(
            classes=classes,
            properties=properties,
            relationships=relationships,
            identity=identity,
            cardinality=cardinality,
        )


__all__ = [
    "ClassKind",
    "EvaluationExpectations",
    "ForeignKeyResolution",
    "IdentityRole",
    "IdentityScope",
    "Materialization",
    "OnMissingTarget",
    "ResolvedAttribute",
    "ResolvedClass",
    "ResolvedModelingPlan",
    "ResolvedProperty",
    "ResolvedRelationship",
    "ReviewState",
    "SchemaDefinitions",
]
