"""Governed ontology and pre-promotion SHACL gates for unstructured claims.

The unstructured product path does not learn its publication vocabulary from a
document.  It loads one reviewed, versioned schema, aligns extractor surface
predicates to that schema, and validates every claim with pySHACL before the
claim can enter the promoted graph.
"""

from __future__ import annotations

import hashlib
from importlib.resources import files
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from rdflib import RDF, RDFS, BNode, Graph, Literal, Namespace, URIRef
from rdflib.namespace import PROV, SH, XSD

from kg_construction.domain.claim import Claim, ClaimKind
from kg_construction.domain.graph_objects import RDFGraph
from kg_construction.domain.models import (
    ConstructionBatch,
    canonical_json,
    normalize_text,
)
from kg_construction.infrastructure.graph_builders.rdf_graph import RDFGraphBuilder

_FINAL_RESOURCE_SHACL_VERSION = "governed-final-resource-shacl-v1"
_PUBLICATION_SCHEMA = Namespace("urn:kg:schema:")


def _term_key(value: str) -> str:
    """Return the comparison key used for governed predicate alignment."""

    return normalize_text(value).replace("-", " ").replace("_", " ")


class GovernedPredicate(BaseModel):
    """One approved predicate and the reviewed extractor terms aligned to it."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1, pattern=r"^[a-z][a-z0-9_]*$")
    aliases: tuple[str, ...] = ()
    claim_kinds: tuple[ClaimKind, ...] = (
        ClaimKind.RELATION,
        ClaimKind.EVENT,
        ClaimKind.FACT,
        ClaimKind.INTENT,
    )

    @field_validator("aliases")
    @classmethod
    def _aliases_are_non_empty(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if any(not value.strip() for value in values):
            raise ValueError("predicate aliases must be non-empty")
        return values


class GovernedUnstructuredSchema(BaseModel):
    """Reviewed publication vocabulary for the unstructured construction path."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_id: str = Field(min_length=1)
    version: str = Field(min_length=1)
    namespace: str = Field(min_length=1)
    entity_types: tuple[str, ...] = Field(min_length=1)
    entity_type_aliases: dict[str, tuple[str, ...]] = Field(default_factory=dict)
    predicates: tuple[GovernedPredicate, ...] = Field(min_length=1)
    source_path: Path | None = None
    source_sha256: str = Field(min_length=64, max_length=64)

    @model_validator(mode="after")
    def _terms_are_unambiguous(self) -> GovernedUnstructuredSchema:
        names = [item.name for item in self.predicates]
        if len(names) != len(set(names)):
            raise ValueError("governed predicate names must be unique")
        if len(self.entity_types) != len(set(self.entity_types)):
            raise ValueError("governed entity types must be unique")
        entity_alignments: dict[str, str] = {}
        for entity_type in self.entity_types:
            for term in (entity_type, *self.entity_type_aliases.get(entity_type, ())):
                key = _term_key(term)
                previous = entity_alignments.get(key)
                if previous is not None and previous != entity_type:
                    raise ValueError(
                        f"governed entity type alignment {term!r} maps to both "
                        f"{previous!r} and {entity_type!r}"
                    )
                entity_alignments[key] = entity_type
        unknown_alias_targets = set(self.entity_type_aliases) - set(self.entity_types)
        if unknown_alias_targets:
            raise ValueError(
                "entity_type_aliases target undeclared governed types: "
                f"{sorted(unknown_alias_targets)}"
            )
        if any(
            not alias.strip()
            for aliases in self.entity_type_aliases.values()
            for alias in aliases
        ):
            raise ValueError("entity type aliases must be non-empty")
        alignments: dict[str, str] = {}
        for predicate in self.predicates:
            for term in (predicate.name, *predicate.aliases):
                key = _term_key(term)
                previous = alignments.get(key)
                if previous is not None and previous != predicate.name:
                    raise ValueError(
                        f"governed alignment term {term!r} maps to both "
                        f"{previous!r} and {predicate.name!r}"
                    )
                alignments[key] = predicate.name
        return self

    @classmethod
    def from_yaml(cls, path: Path) -> GovernedUnstructuredSchema:
        """Load and content-bind a reviewed schema document."""

        if not path.is_file():
            raise FileNotFoundError(f"governed unstructured schema does not exist: {path}")
        raw_bytes = path.read_bytes()
        return cls._from_bytes(raw_bytes, source_path=path.resolve())

    @classmethod
    def from_package_resource(cls) -> GovernedUnstructuredSchema:
        """Load the immutable wheel-safe default schema."""

        resource = files("kg_construction").joinpath(
            "resources/governed_unstructured_ontology.yaml"
        )
        return cls._from_bytes(resource.read_bytes(), source_path=None)

    @classmethod
    def _from_bytes(
        cls,
        raw_bytes: bytes,
        *,
        source_path: Path | None,
    ) -> GovernedUnstructuredSchema:
        payload = yaml.safe_load(raw_bytes)
        if not isinstance(payload, dict):
            raise ValueError("governed unstructured schema must be a mapping")
        return cls.model_validate(
            {
                **payload,
                "source_path": source_path,
                "source_sha256": hashlib.sha256(raw_bytes).hexdigest(),
            }
        )

    @classmethod
    def from_predicates(
        cls,
        predicates: set[str],
        *,
        schema_id: str = "inline-governed-unstructured-schema",
        version: str = "v1",
    ) -> GovernedUnstructuredSchema:
        """Create a content-bound schema for explicit programmatic contracts."""

        normalized = tuple(
            GovernedPredicate(name=item) for item in sorted(predicates)
        )
        payload = {
            "schema_id": schema_id,
            "version": version,
            "namespace": "urn:kg:unstructured:",
            "entity_types": (
                "Entity",
                "Person",
                "Organization",
                "Location",
                "Place",
                "Event",
                "Product",
                "Artifact",
                "Date",
                "Concept",
                "System",
                "Service",
                "Incident",
                "Identifier",
                "Version",
            ),
            "entity_type_aliases": {
                "Identifier": ("EmailAddress",),
            },
            "predicates": [item.model_dump(mode="json") for item in normalized],
        }
        return cls.model_validate(
            {
                **payload,
                "source_sha256": hashlib.sha256(
                    canonical_json(payload).encode("utf-8")
                ).hexdigest(),
            }
        )

    @property
    def predicate_names(self) -> frozenset[str]:
        return frozenset(item.name for item in self.predicates)

    def aligned_entity_type(self, entity_type: str) -> str | None:
        """Return a reviewed canonical type for one provider-emitted label."""

        key = _term_key(entity_type)
        for canonical in self.entity_types:
            if key in {
                _term_key(canonical),
                *map(_term_key, self.entity_type_aliases.get(canonical, ())),
            }:
                return canonical
        return None

    def aligned_predicate(
        self,
        predicate: str,
        *,
        kind: ClaimKind,
    ) -> str | None:
        """Return the approved predicate for one extractor term and claim kind."""

        key = _term_key(predicate)
        for definition in self.predicates:
            if kind not in definition.claim_kinds:
                continue
            if key in {_term_key(definition.name), *map(_term_key, definition.aliases)}:
                return definition.name
        return None


class ClaimShaclResult(BaseModel):
    """Auditable result of validating one claim before promotion."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    claim_id: str = Field(min_length=1)
    schema_id: str = Field(min_length=1)
    schema_version: str = Field(min_length=1)
    schema_sha256: str = Field(min_length=64, max_length=64)
    executed: bool
    conforms: bool
    results_text: str


class GovernedClaimShaclGate:
    """Validate exact evidence/provenance and the approved term with pySHACL."""

    name = "governed_unstructured_claim_shacl_gate"
    technique = "pre_promotion_pyshacl_claim_validation"

    def __init__(self, schema: GovernedUnstructuredSchema) -> None:
        self.schema = schema
        self._namespace = Namespace(schema.namespace)
        self._shapes = self._build_shapes()

    def validate(self, claim: Claim, document_text: str) -> ClaimShaclResult:
        """Execute the SHACL gate for one immutable claim."""

        from pyshacl import validate

        data = self._claim_graph(claim, document_text)
        conforms, _, results_text = validate(
            data_graph=data,
            shacl_graph=self._shapes,
            inference="none",
            abort_on_first=False,
            allow_infos=False,
            allow_warnings=False,
            advanced=False,
            meta_shacl=False,
        )
        return ClaimShaclResult(
            claim_id=claim.claim_id,
            schema_id=self.schema.schema_id,
            schema_version=self.schema.version,
            schema_sha256=self.schema.source_sha256,
            executed=True,
            conforms=bool(conforms),
            results_text=str(results_text),
        )

    def _claim_graph(self, claim: Claim, document_text: str) -> Graph:
        kg = self._namespace
        graph = Graph()
        claim_iri = URIRef(f"{self.schema.namespace}claim/{claim.claim_id}")
        graph.add((claim_iri, RDF.type, kg.Claim))
        graph.add((claim_iri, kg.kind, Literal(claim.kind.value)))
        graph.add((claim_iri, kg.subject, Literal(claim.subject)))
        graph.add((claim_iri, kg.predicate, Literal(claim.predicate)))
        graph.add((claim_iri, kg.provider, Literal(claim.provenance.provider)))
        graph.add((claim_iri, kg.modelVersion, Literal(claim.provenance.model_version)))
        graph.add(
            (
                claim_iri,
                kg.configurationId,
                Literal(claim.provenance.configuration_id),
            )
        )
        graph.add(
            (
                claim_iri,
                kg.confidence,
                Literal(claim.provenance.confidence, datatype=XSD.decimal),
            )
        )
        if claim.object is not None:
            graph.add((claim_iri, kg.object, Literal(claim.object)))
        for evidence in claim.evidence:
            evidence_iri = URIRef(
                f"{self.schema.namespace}evidence/{evidence.evidence_id}"
            )
            graph.add((claim_iri, kg.evidence, evidence_iri))
            graph.add((evidence_iri, RDF.type, kg.ExactEvidence))
            graph.add((evidence_iri, kg.quote, Literal(evidence.span.text)))
            graph.add(
                (
                    evidence_iri,
                    kg.start,
                    Literal(evidence.span.start, datatype=XSD.integer),
                )
            )
            graph.add(
                (
                    evidence_iri,
                    kg.end,
                    Literal(evidence.span.end, datatype=XSD.integer),
                )
            )
            graph.add(
                (
                    evidence_iri,
                    kg.exact,
                    Literal(
                        evidence.is_exact_for(claim.document, document_text),
                        datatype=XSD.boolean,
                    ),
                )
            )
        return graph

    def _build_shapes(self) -> Graph:
        kg = self._namespace
        graph = Graph()
        claim_shape = URIRef(f"{self.schema.namespace}ClaimShape")
        graph.add((claim_shape, RDF.type, SH.NodeShape))
        graph.add((claim_shape, SH.targetClass, kg.Claim))
        self._property_shape(graph, claim_shape, kg.kind, min_count=1, max_count=1)
        predicate_shape = self._property_shape(
            graph,
            claim_shape,
            kg.predicate,
            min_count=1,
            max_count=1,
        )
        allowed_head = self._rdf_list(
            graph,
            [Literal(name) for name in sorted(self.schema.predicate_names)],
        )
        graph.add((predicate_shape, SH["in"], allowed_head))
        self._property_shape(graph, claim_shape, kg.subject, min_count=1, max_count=1)
        self._property_shape(graph, claim_shape, kg.evidence, min_count=1)
        self._property_shape(graph, claim_shape, kg.provider, min_count=1, max_count=1)
        self._property_shape(
            graph,
            claim_shape,
            kg.modelVersion,
            min_count=1,
            max_count=1,
        )
        self._property_shape(
            graph,
            claim_shape,
            kg.configurationId,
            min_count=1,
            max_count=1,
        )

        evidence_shape = URIRef(f"{self.schema.namespace}ExactEvidenceShape")
        graph.add((evidence_shape, RDF.type, SH.NodeShape))
        graph.add((evidence_shape, SH.targetClass, kg.ExactEvidence))
        self._property_shape(graph, evidence_shape, kg.quote, min_count=1, max_count=1)
        self._property_shape(
            graph,
            evidence_shape,
            kg.start,
            min_count=1,
            max_count=1,
            datatype=XSD.integer,
        )
        self._property_shape(
            graph,
            evidence_shape,
            kg.end,
            min_count=1,
            max_count=1,
            datatype=XSD.integer,
        )
        exact_shape = self._property_shape(
            graph,
            evidence_shape,
            kg.exact,
            min_count=1,
            max_count=1,
            datatype=XSD.boolean,
        )
        graph.add((exact_shape, SH.hasValue, Literal(True, datatype=XSD.boolean)))
        return graph

    @staticmethod
    def _property_shape(
        graph: Graph,
        node_shape: URIRef,
        path: URIRef,
        *,
        min_count: int | None = None,
        max_count: int | None = None,
        datatype: URIRef | None = None,
    ) -> BNode:
        shape = BNode()
        graph.add((node_shape, SH.property, shape))
        graph.add((shape, SH.path, path))
        if min_count is not None:
            graph.add((shape, SH.minCount, Literal(min_count)))
        if max_count is not None:
            graph.add((shape, SH.maxCount, Literal(max_count)))
        if datatype is not None:
            graph.add((shape, SH.datatype, datatype))
        return shape

    @staticmethod
    def _rdf_list(graph: Graph, values: list[Any]) -> URIRef | BNode:
        if not values:
            return RDF.nil
        head = BNode()
        current: URIRef | BNode = head
        for index, value in enumerate(values):
            graph.add((current, RDF.first, value))
            next_node: URIRef | BNode = (
                RDF.nil if index == len(values) - 1 else BNode()
            )
            graph.add((current, RDF.rest, next_node))
            current = next_node
        return head


class FinalResourceShaclResult(BaseModel):
    """Content-bound SHACL verdict for the post-promotion RDF dataset."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_id: str = Field(min_length=1)
    schema_version: str = Field(min_length=1)
    schema_sha256: str = Field(min_length=64, max_length=64)
    shacl_contract_version: str = Field(min_length=1)
    shacl_contract_sha256: str = Field(min_length=64, max_length=64)
    rdf_graph_sha256: str = Field(min_length=64, max_length=64)
    rdf_triple_count: int = Field(ge=0)
    rdf_quad_count: int = Field(ge=0)
    executed: bool
    conforms: bool
    results_text: str
    results_graph_turtle: str


def final_rdf_graph_sha256(rdf_graph: RDFGraph) -> str:
    """Hash the complete RDF dataset, including provenance named graphs."""

    payload = {
        "triples": sorted(
            (
                item.subject,
                item.predicate,
                item.object,
                item.object_type,
                item.datatype,
            )
            for item in rdf_graph.triples
        ),
        "quads": sorted(
            (
                item.subject,
                item.predicate,
                item.object,
                item.object_type,
                item.datatype,
                item.graph,
            )
            for item in rdf_graph.quads
        ),
    }
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


class GovernedResourceShaclGate:
    """Validate the complete RDF resource layer after identity promotion.

    This gate deliberately runs after external identity reconciliation because
    that stage can add canonical resources and rewrite relation endpoints.  It
    validates the union of the default graph and provenance named graphs, then
    binds the exact RDF dataset and reviewed shape contract by SHA-256.
    """

    name = "governed_unstructured_final_resource_shacl_gate"
    technique = "post_promotion_resource_layer_pyshacl"

    def __init__(self, schema: GovernedUnstructuredSchema) -> None:
        self.schema = schema
        self._shapes = self._build_shapes()
        self.shacl_contract_sha256 = hashlib.sha256(
            canonical_json(
                {
                    "contract_version": _FINAL_RESOURCE_SHACL_VERSION,
                    "schema_id": schema.schema_id,
                    "schema_version": schema.version,
                    "schema_sha256": schema.source_sha256,
                    "entity_types": sorted(schema.entity_types),
                    "predicates": sorted(schema.predicate_names),
                }
            ).encode("utf-8")
        ).hexdigest()

    def validate(
        self,
        batch: ConstructionBatch,
    ) -> tuple[RDFGraph, FinalResourceShaclResult]:
        """Build and validate the exact final RDF dataset, failing no checks open."""

        from pyshacl import validate

        rdf_graph = RDFGraphBuilder().build(batch)
        data_graph = Graph()
        dataset = rdf_graph.to_rdflib_dataset()
        for subject, predicate, object_, _context in dataset.quads(
            (None, None, None, None)
        ):
            data_graph.add((subject, predicate, object_))
        conforms, results_graph, results_text = validate(
            data_graph=data_graph,
            shacl_graph=self._shapes,
            inference="none",
            abort_on_first=False,
            allow_infos=False,
            allow_warnings=False,
            advanced=False,
            meta_shacl=False,
        )
        return rdf_graph, FinalResourceShaclResult(
            schema_id=self.schema.schema_id,
            schema_version=self.schema.version,
            schema_sha256=self.schema.source_sha256,
            shacl_contract_version=_FINAL_RESOURCE_SHACL_VERSION,
            shacl_contract_sha256=self.shacl_contract_sha256,
            rdf_graph_sha256=final_rdf_graph_sha256(rdf_graph),
            rdf_triple_count=len(rdf_graph.triples),
            rdf_quad_count=len(rdf_graph.quads),
            executed=True,
            conforms=bool(conforms),
            results_text=str(results_text),
            results_graph_turtle=str(results_graph.serialize(format="turtle")),
        )

    def _build_shapes(self) -> Graph:
        kg = _PUBLICATION_SCHEMA
        graph = Graph()
        graph.bind("kg", kg)
        graph.bind("sh", SH)
        graph.bind("prov", PROV)

        identity_kind = URIRef("urn:kg:shape:IdentityResourceKind")
        graph.add((identity_kind, RDF.type, SH.NodeShape))
        graph.add(
            (
                identity_kind,
                SH["or"],
                self._rdf_list(
                    graph,
                    [
                        self._class_constraint(graph, kg.ObservedEntity),
                        self._class_constraint(graph, kg.CanonicalEntity),
                    ],
                ),
            )
        )
        claim_kind = URIRef("urn:kg:shape:ClaimOrEventKind")
        graph.add((claim_kind, RDF.type, SH.NodeShape))
        graph.add(
            (
                claim_kind,
                SH["or"],
                self._rdf_list(
                    graph,
                    [
                        self._class_constraint(graph, kg.Claim),
                        self._class_constraint(graph, kg.ClaimEvent),
                    ],
                ),
            )
        )

        document = self._node_shape(graph, "Document", kg.Document)
        self._resource_provenance(graph, document)
        self._property(graph, document, kg.document_version, min_count=1, max_count=1)
        self._property(graph, document, kg.source_type, min_count=1, max_count=1)
        self._property(graph, document, kg.content_digest, min_count=1, max_count=1)
        self._property(
            graph,
            document,
            kg.text_length,
            min_count=1,
            max_count=1,
            datatype=XSD.integer,
        )
        self._property(
            graph,
            document,
            kg.HAS_MENTION,
            value_class=kg.Mention,
        )
        asserts = self._property(graph, document, kg.ASSERTS)
        graph.add((asserts, SH.node, claim_kind))

        mention = self._node_shape(graph, "Mention", kg.Mention)
        self._resource_provenance(graph, mention)
        self._property(graph, mention, kg.domain_mention_id, min_count=1, max_count=1)
        self._property(graph, mention, kg.document_version, min_count=1, max_count=1)
        mention_type = self._property(
            graph,
            mention,
            kg.detected_entity_type,
            min_count=1,
            max_count=1,
        )
        graph.add(
            (
                mention_type,
                SH["in"],
                self._rdf_list(
                    graph,
                    [
                        Literal(item, datatype=XSD.string)
                        for item in sorted(self.schema.entity_types)
                    ],
                ),
            )
        )
        self._property(
            graph,
            mention,
            kg.start,
            min_count=1,
            max_count=1,
            datatype=XSD.integer,
        )
        self._property(
            graph,
            mention,
            kg.end,
            min_count=1,
            max_count=1,
            datatype=XSD.integer,
        )
        self._property(
            graph,
            mention,
            kg.DENOTES_OBSERVED_ENTITY,
            max_count=1,
            value_class=kg.ObservedEntity,
        )
        self._property(
            graph,
            mention,
            kg.REFERS_TO_CANONICAL_ENTITY,
            max_count=1,
            value_class=kg.CanonicalEntity,
        )
        linked_mention = self._node_shape(
            graph,
            "LinkedMention",
            kg.LinkedMention,
        )
        graph.add((linked_mention, SH["class"], kg.Mention))
        self._property(
            graph,
            linked_mention,
            kg.REFERS_TO_CANONICAL_ENTITY,
            min_count=1,
            max_count=1,
            value_class=kg.CanonicalEntity,
        )

        observed = self._node_shape(graph, "ObservedEntity", kg.ObservedEntity)
        self._resource_provenance(graph, observed)
        observed_layer = self._property(
            graph,
            observed,
            kg.identity_layer,
            min_count=1,
            max_count=1,
        )
        graph.add(
            (
                observed_layer,
                SH.hasValue,
                Literal("observed", datatype=XSD.string),
            )
        )
        self._semantic_type_property(graph, observed)
        self._property(graph, observed, kg.mention_ids, min_count=1, max_count=1)
        self._property(
            graph,
            observed,
            kg.RESOLVED_TO_CANONICAL_ENTITY,
            max_count=1,
            value_class=kg.CanonicalEntity,
        )
        external_observed = self._node_shape(
            graph,
            "ExternallyLinkedObservedEntity",
            kg.ExternallyLinkedObservedEntity,
        )
        graph.add((external_observed, SH["class"], kg.ObservedEntity))
        self._property(
            graph,
            external_observed,
            kg.external_system,
            min_count=1,
            max_count=1,
        )
        self._property(
            graph,
            external_observed,
            kg.external_id,
            min_count=1,
            max_count=1,
        )
        self._property(
            graph,
            external_observed,
            kg.RESOLVED_TO_CANONICAL_ENTITY,
            min_count=1,
            max_count=1,
            value_class=kg.CanonicalEntity,
        )

        canonical = self._node_shape(graph, "CanonicalEntity", kg.CanonicalEntity)
        self._resource_provenance(graph, canonical)
        canonical_layer = self._property(
            graph,
            canonical,
            kg.identity_layer,
            min_count=1,
            max_count=1,
        )
        graph.add(
            (
                canonical_layer,
                SH.hasValue,
                Literal("canonical", datatype=XSD.string),
            )
        )
        self._semantic_type_property(graph, canonical)
        self._property(
            graph,
            canonical,
            kg.canonical_entity_id,
            min_count=1,
            max_count=1,
        )
        self._property(graph, canonical, kg.mention_ids, min_count=1, max_count=1)
        external_canonical = self._node_shape(
            graph,
            "ExternalCanonicalEntity",
            kg.ExternalCanonicalEntity,
        )
        graph.add((external_canonical, SH["class"], kg.CanonicalEntity))
        self._property(
            graph,
            external_canonical,
            kg.external_system,
            min_count=1,
            max_count=1,
        )
        self._property(
            graph,
            external_canonical,
            kg.external_id,
            min_count=1,
            max_count=1,
        )

        claim = self._node_shape(graph, "Claim", kg.Claim)
        self._resource_provenance(graph, claim)
        self._claim_properties(graph, claim)
        subject = self._property(
            graph,
            claim,
            kg.HAS_SUBJECT,
            min_count=1,
            max_count=1,
        )
        graph.add((subject, SH.node, identity_kind))
        object_ = self._property(
            graph,
            claim,
            kg.HAS_OBJECT,
            min_count=1,
            max_count=1,
        )
        graph.add((object_, SH.node, identity_kind))

        event = self._node_shape(graph, "Event", kg.ClaimEvent)
        self._resource_provenance(graph, event)
        self._claim_properties(graph, event)
        event_roles = self._property(
            graph,
            event,
            kg.event_roles,
            min_count=1,
            max_count=1,
        )
        graph.add((event_roles, SH.minLength, Literal(3)))

        evidence = self._node_shape(graph, "Evidence", kg.Evidence)
        self._property(
            graph,
            evidence,
            PROV.wasDerivedFrom,
            min_count=1,
            max_count=1,
            node_kind=SH.IRI,
        )
        self._property(graph, evidence, kg.sourceId, min_count=1, max_count=1)
        self._property(graph, evidence, kg.sourceName, min_count=1, max_count=1)
        self._property(graph, evidence, kg.sourceType, min_count=1, max_count=1)
        self._property(graph, evidence, kg.location, min_count=1, max_count=1)

        exact_evidence = self._node_shape(
            graph,
            "ExactEvidence",
            kg.ExactEvidence,
        )
        graph.add((exact_evidence, SH["class"], kg.Evidence))
        self._property(
            graph,
            exact_evidence,
            kg.evidenceText,
            min_count=1,
            max_count=1,
        )
        self._property(
            graph,
            exact_evidence,
            kg.start,
            min_count=1,
            max_count=1,
            datatype=XSD.integer,
        )
        self._property(
            graph,
            exact_evidence,
            kg.end,
            min_count=1,
            max_count=1,
            datatype=XSD.integer,
        )
        exact = self._property(
            graph,
            exact_evidence,
            kg.exact,
            min_count=1,
            max_count=1,
            datatype=XSD.boolean,
        )
        graph.add((exact, SH.hasValue, Literal(True, datatype=XSD.boolean)))
        self._property(
            graph,
            exact_evidence,
            kg.documentVersion,
            min_count=1,
            max_count=1,
        )
        self._property(
            graph,
            exact_evidence,
            kg.domainEvidenceId,
            min_count=1,
            max_count=1,
        )

        fact = self._node_shape(graph, "Fact", kg.Fact)
        self._resource_provenance(graph, fact)
        self._property(
            graph,
            fact,
            kg.subject,
            min_count=1,
            max_count=1,
            node_kind=SH.IRI,
        )
        self._property(
            graph,
            fact,
            kg.predicate,
            min_count=1,
            max_count=1,
            node_kind=SH.IRI,
        )
        self._property(
            graph,
            fact,
            kg.object,
            min_count=1,
            max_count=1,
            node_kind=SH.IRI,
        )

        for predicate in sorted(self.schema.predicate_names):
            relation_iris = {
                "relation": kg[predicate.upper()],
                "fact": kg[predicate],
            }
            for projection, relation in relation_iris.items():
                domain_shape = URIRef(
                    f"urn:kg:shape:{predicate}:{projection}:Domain"
                )
                graph.add((domain_shape, RDF.type, SH.NodeShape))
                graph.add((domain_shape, SH.targetSubjectsOf, relation))
                graph.add((domain_shape, SH.node, identity_kind))
                range_shape = URIRef(
                    f"urn:kg:shape:{predicate}:{projection}:Range"
                )
                graph.add((range_shape, RDF.type, SH.NodeShape))
                graph.add((range_shape, SH.targetObjectsOf, relation))
                graph.add((range_shape, SH.node, identity_kind))
        return graph

    def _resource_provenance(self, graph: Graph, shape: URIRef) -> None:
        self._property(
            graph,
            shape,
            RDFS.label,
            min_count=1,
            max_count=1,
        )
        self._property(
            graph,
            shape,
            PROV.wasDerivedFrom,
            min_count=1,
            node_kind=SH.IRI,
        )
        self._property(
            graph,
            shape,
            _PUBLICATION_SCHEMA.extractorName,
            min_count=1,
            max_count=1,
        )
        self._property(
            graph,
            shape,
            _PUBLICATION_SCHEMA.technique,
            min_count=1,
            max_count=1,
        )
        self._property(
            graph,
            shape,
            _PUBLICATION_SCHEMA.schemaVersion,
            min_count=1,
            max_count=1,
        )
        self._property(
            graph,
            shape,
            _PUBLICATION_SCHEMA.provider,
            min_count=1,
            max_count=1,
        )
        self._property(
            graph,
            shape,
            _PUBLICATION_SCHEMA.modelVersion,
            min_count=1,
            max_count=1,
        )
        self._property(
            graph,
            shape,
            _PUBLICATION_SCHEMA.configurationId,
            min_count=1,
            max_count=1,
        )

    def _semantic_type_property(self, graph: Graph, shape: URIRef) -> None:
        semantic_type = self._property(
            graph,
            shape,
            _PUBLICATION_SCHEMA.semantic_entity_type,
            min_count=1,
            max_count=1,
        )
        graph.add(
            (
                semantic_type,
                SH["in"],
                self._rdf_list(
                    graph,
                    [
                        Literal(item, datatype=XSD.string)
                        for item in sorted(self.schema.entity_types)
                    ],
                ),
            )
        )

    def _claim_properties(self, graph: Graph, shape: URIRef) -> None:
        kg = _PUBLICATION_SCHEMA
        self._property(graph, shape, kg.domain_claim_id, min_count=1, max_count=1)
        self._property(graph, shape, kg.claim_kind, min_count=1, max_count=1)
        predicate = self._property(
            graph,
            shape,
            kg.predicate,
            min_count=1,
            max_count=1,
        )
        graph.add(
            (
                predicate,
                SH["in"],
                self._rdf_list(
                    graph,
                    [
                        Literal(item, datatype=XSD.string)
                        for item in sorted(self.schema.predicate_names)
                    ],
                ),
            )
        )
        promoted = self._property(
            graph,
            shape,
            kg.review_state,
            min_count=1,
            max_count=1,
        )
        graph.add(
            (
                promoted,
                SH.hasValue,
                Literal("promoted", datatype=XSD.string),
            )
        )

    @staticmethod
    def _node_shape(graph: Graph, name: str, target_class: URIRef) -> URIRef:
        shape = URIRef(f"urn:kg:shape:{name}")
        graph.add((shape, RDF.type, SH.NodeShape))
        graph.add((shape, SH.targetClass, target_class))
        return shape

    @staticmethod
    def _class_constraint(graph: Graph, value_class: URIRef) -> BNode:
        constraint = BNode()
        graph.add((constraint, SH["class"], value_class))
        return constraint

    @staticmethod
    def _property(
        graph: Graph,
        node_shape: URIRef,
        path: URIRef,
        *,
        min_count: int | None = None,
        max_count: int | None = None,
        datatype: URIRef | None = None,
        node_kind: URIRef | None = None,
        value_class: URIRef | None = None,
    ) -> BNode:
        shape = BNode()
        graph.add((node_shape, SH.property, shape))
        graph.add((shape, SH.path, path))
        if min_count is not None:
            graph.add((shape, SH.minCount, Literal(min_count)))
        if max_count is not None:
            graph.add((shape, SH.maxCount, Literal(max_count)))
        if datatype is not None:
            graph.add((shape, SH.datatype, datatype))
        if node_kind is not None:
            graph.add((shape, SH.nodeKind, node_kind))
        if value_class is not None:
            graph.add((shape, SH["class"], value_class))
        return shape

    @staticmethod
    def _rdf_list(graph: Graph, values: list[Any]) -> URIRef | BNode:
        return GovernedClaimShaclGate._rdf_list(graph, values)


__all__ = [
    "ClaimShaclResult",
    "FinalResourceShaclResult",
    "GovernedClaimShaclGate",
    "GovernedPredicate",
    "GovernedResourceShaclGate",
    "GovernedUnstructuredSchema",
    "final_rdf_graph_sha256",
]
