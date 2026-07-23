"""Form an ontology from a modelling policy and gate a release with SHACL.

Given a ``ModelingPolicy`` (which lowers to an ``OntologySchema`` with classes,
predicates, domain/range, cardinality, and datatypes), this module:

1. proposes the ontology through a proposed -> quarantine -> promoted lifecycle
   built on the existing ``GraphSchema`` lifecycle;
2. generates SHACL shapes encoding class targets, datatype constraints, object
   ranges (``sh:class`` + ``sh:nodeKind sh:IRI``), and cardinality
   (``sh:minCount`` / ``sh:maxCount``); and
3. runs a release gate: a graph passes only with zero SHACL ``Violation`` results.

It also provides a small, RDF-only materialiser that turns a policy and records
into an ontology-conformant graph, so the gate can be exercised directly.
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from kg_construction.application.modeling_policy import ModelingPolicy
from kg_construction.domain.graph_objects import RDFGraph, RDFTriple
from kg_construction.domain.modeling import ColumnRole, OntologyPredicate, OntologySchema
from kg_construction.domain.models import content_hash
from kg_construction.domain.schema import GraphSchema, SchemaLifecycleStatus
from kg_construction.infrastructure.ontology.shacl import ShaclValidator

JsonDict = dict[str, object]

logger = logging.getLogger(__name__)

_RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
_SLUG_RE = re.compile(r"[^0-9A-Za-z]+")

# JsonType -> SHACL prefixed datatype (for shapes) and full XSD IRI (for literals).
_XSD_PREFIXED: dict[str, str] = {
    "string": "xsd:string",
    "integer": "xsd:integer",
    "number": "xsd:decimal",
    "boolean": "xsd:boolean",
    "date": "xsd:date",
    "datetime": "xsd:dateTime",
    "object": "xsd:string",
    "array": "xsd:string",
}
_XSD_IRI: dict[str, str] = {
    key: "http://www.w3.org/2001/XMLSchema#" + value.split(":", 1)[1]
    for key, value in _XSD_PREFIXED.items()
}


def _term(name: str) -> str:
    """Return an IRI-safe local name."""

    safe = "".join(character if character.isalnum() else "_" for character in name)
    return safe.strip("_") or "unnamed"


def _slug(value: str) -> str:
    """Return an IRI-safe path segment for an instance value."""

    slug = _SLUG_RE.sub("-", value).strip("-").lower()
    return slug or content_hash(value, length=12)


def generate_shacl_shapes(ontology: OntologySchema, *, namespace: str = "urn:kg:schema:") -> str:
    """Generate SHACL node/property shapes from an ontology (Astrea-style)."""

    lines = [
        "@prefix sh: <http://www.w3.org/ns/shacl#> .",
        "@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .",
        f"@prefix kg: <{namespace}> .",
        "",
    ]
    predicates_by_domain: dict[str, list[OntologyPredicate]] = defaultdict(list)
    for predicate in ontology.predicates:
        predicates_by_domain[predicate.domain].append(predicate)

    for ontology_class in sorted(ontology.classes, key=lambda item: item.name):
        blocks = [
            _property_shape(predicate)
            for predicate in predicates_by_domain.get(ontology_class.name, [])
        ]
        lines.append(f"kg:{_term(ontology_class.name)}Shape a sh:NodeShape ;")
        target_line = f"  sh:targetClass kg:{_term(ontology_class.name)}"
        lines.append(target_line + (" ;" if blocks else " ."))
        for index, block in enumerate(blocks):
            suffix = " ." if index == len(blocks) - 1 else " ;"
            lines.append(f"  sh:property {block}{suffix}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _property_shape(predicate: OntologyPredicate) -> str:
    parts = [f"sh:path kg:{_term(predicate.name)}"]
    if predicate.kind == "datatype":
        parts.append(f"sh:datatype {_XSD_PREFIXED.get(predicate.range, 'xsd:string')}")
    else:
        parts.append(f"sh:class kg:{_term(predicate.range)}")
        parts.append("sh:nodeKind sh:IRI")
    if predicate.cardinality.min_count >= 1:
        parts.append(f"sh:minCount {predicate.cardinality.min_count}")
    if predicate.cardinality.max_count is not None:
        parts.append(f"sh:maxCount {predicate.cardinality.max_count}")
    return "[ " + " ; ".join(parts) + " ]"


class StructuredOntologyProposal(BaseModel):
    """A proposed ontology + generated shapes, with a governance lifecycle."""

    model_config = ConfigDict(extra="forbid")

    ontology: OntologySchema
    graph_schema: GraphSchema
    shacl_turtle: str
    owl_turtle: str | None = None
    quarantined_terms: list[str] = Field(default_factory=list)
    status: SchemaLifecycleStatus = "proposed"

    @property
    def promotable(self) -> bool:
        """A proposal can be promoted only once quarantined terms are resolved."""

        return not self.quarantined_terms

    def _with_status(self, status: SchemaLifecycleStatus) -> StructuredOntologyProposal:
        return self.model_copy(
            update={"status": status, "graph_schema": self.graph_schema.with_status(status)}
        )

    def approve(self) -> StructuredOntologyProposal:
        return self._with_status("approved")

    def reject(self) -> StructuredOntologyProposal:
        return self._with_status("rejected")

    def promote(self) -> StructuredOntologyProposal:
        if not self.promotable:
            raise ValueError(
                f"cannot promote: {len(self.quarantined_terms)} term(s) are quarantined "
                f"({self.quarantined_terms})"
            )
        return self._with_status("promoted")


def propose_structured_ontology(
    policy: ModelingPolicy,
    *,
    namespace: str = "urn:kg:schema:",
    version: str = "v1",
    emit_owl: bool = True,
) -> StructuredOntologyProposal:
    """Propose an ontology + SHACL shapes from a modelling policy."""

    ontology = policy.to_ontology_schema().model_copy(
        update={"namespace": namespace, "version": version}
    )
    graph_schema = ontology.to_graph_schema().with_status("proposed")
    quarantined = ontology.unresolved_predicate_references()
    logger.info(
        "Proposed ontology with %d classes and %d predicates (%d quarantined terms)",
        len(ontology.classes),
        len(ontology.predicates),
        len(quarantined),
    )
    return StructuredOntologyProposal(
        ontology=ontology,
        graph_schema=graph_schema,
        shacl_turtle=generate_shacl_shapes(ontology, namespace=namespace),
        owl_turtle=graph_schema.to_owl_turtle() if emit_owl else None,
        quarantined_terms=ontology.unresolved_predicate_references(),
        status="proposed",
    )


def _class_iri(namespace: str, name: str) -> str:
    return f"{namespace}{_term(name)}"


def _predicate_iri(namespace: str, name: str) -> str:
    return f"{namespace}{_term(name)}"


def _instance_iri(instance_namespace: str, class_name: str, value: str) -> str:
    return f"{instance_namespace}{_term(class_name)}/{_slug(value)}"


def materialize_structured_rdf(
    policy: ModelingPolicy,
    records: list[JsonDict],
    *,
    namespace: str = "urn:kg:schema:",
    instance_namespace: str = "urn:kg:",
) -> RDFGraph:
    """Materialise an ontology-conformant RDF graph from a policy and records.

    Used to exercise the SHACL gate. Categorical and reference values collapse into
    shared, typed reference-node IRIs (one per canonical value); measures,
    temporals, and free text become typed literals.
    """

    ontology = policy.to_ontology_schema()
    predicates = [p for p in ontology.predicates if p.domain == policy.record_type and p.source_columns]
    id_column = policy.id_columns[0] if policy.id_columns else None
    triples: dict[tuple[str, str, str, str, str], RDFTriple] = {}

    def add(subject: str, predicate: str, obj: str, *, literal: str | None = None) -> None:
        object_type: Literal["iri", "literal"] = "literal" if literal is not None else "iri"
        datatype = _XSD_IRI.get(literal) if literal is not None else None
        key = (subject, predicate, obj, object_type, datatype or "")
        triples.setdefault(
            key,
            RDFTriple(subject=subject, predicate=predicate, object=obj, object_type=object_type, datatype=datatype),
        )

    def type_node(iri: str, class_name: str) -> None:
        add(iri, _RDF_TYPE, _class_iri(namespace, class_name))

    for index, row in enumerate(records):
        identity_value = str(row.get(id_column)) if id_column and row.get(id_column) not in (None, "") else str(index)
        record_iri = _instance_iri(instance_namespace, policy.record_type, identity_value)
        type_node(record_iri, policy.record_type)

        for predicate in predicates:
            column = predicate.source_columns[0]
            value = row.get(column)
            if value in (None, ""):
                continue
            predicate_iri = _predicate_iri(namespace, predicate.name)
            if predicate.kind == "datatype":
                add(record_iri, predicate_iri, str(value), literal=predicate.range)
                continue
            column_policy = policy.column_policy(column)
            is_self_reference = (
                column_policy is not None
                and column_policy.role == ColumnRole.ENTITY_REFERENCE
                and predicate.range == policy.record_type
            )
            if is_self_reference:
                target_iri = _instance_iri(instance_namespace, policy.record_type, str(value))
            else:
                canonical = policy.canonicalization_for(column).apply(str(value))
                target_iri = _instance_iri(instance_namespace, predicate.range, canonical)
            type_node(target_iri, predicate.range)
            add(record_iri, predicate_iri, target_iri)

    return RDFGraph(triples=[triples[key] for key in sorted(triples)])


class ShaclReleaseReport(BaseModel):
    """Result of the structured SHACL release gate."""

    model_config = ConfigDict(extra="forbid")

    conforms: bool
    violation_count: int
    warning_count: int
    info_count: int
    results_text: str
    shacl_turtle: str

    @property
    def release_ok(self) -> bool:
        """A release graph passes only with zero SHACL Violations."""

        return self.violation_count == 0


class StructuredReleaseGate:
    """Validate a structured RDF graph against proposed SHACL shapes."""

    name = "structured_shacl_release_gate"

    def __init__(self, validator: ShaclValidator | None = None) -> None:
        self.validator = validator or ShaclValidator()

    def validate(
        self,
        rdf_graph: RDFGraph,
        *,
        shacl_turtle: str | None = None,
        proposal: StructuredOntologyProposal | None = None,
        inference: str = "none",
    ) -> ShaclReleaseReport:
        """Validate ``rdf_graph`` and count results by severity."""

        shapes = shacl_turtle or (proposal.shacl_turtle if proposal else None)
        if shapes is None:
            raise ValueError("Provide shacl_turtle or a proposal for the release gate.")
        result = self.validator.validate_rdf_graph(
            rdf_graph, shacl_turtle=shapes, inference=inference
        )
        violation, warning, info = _count_severities(result.results_graph_turtle)
        logger.info(
            "SHACL release gate: conforms=%s violations=%d warnings=%d",
            result.conforms,
            violation,
            warning,
        )
        return ShaclReleaseReport(
            conforms=result.conforms,
            violation_count=violation,
            warning_count=warning,
            info_count=info,
            results_text=result.results_text,
            shacl_turtle=shapes,
        )


def _count_severities(results_graph_turtle: str) -> tuple[int, int, int]:
    from rdflib import Graph, Namespace

    sh = Namespace("http://www.w3.org/ns/shacl#")
    graph = Graph()
    graph.parse(data=results_graph_turtle, format="turtle")
    violations = len(list(graph.subjects(sh.resultSeverity, sh.Violation)))
    warnings = len(list(graph.subjects(sh.resultSeverity, sh.Warning)))
    infos = len(list(graph.subjects(sh.resultSeverity, sh.Info)))
    return violations, warnings, infos


__all__ = [
    "ShaclReleaseReport",
    "StructuredOntologyProposal",
    "StructuredReleaseGate",
    "generate_shacl_shapes",
    "materialize_structured_rdf",
    "propose_structured_ontology",
]
