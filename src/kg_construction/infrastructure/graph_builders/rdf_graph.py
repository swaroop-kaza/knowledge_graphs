"""Standards-based RDF triple and quad builder."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from kg_construction.domain.graph_objects import RDFGraph, RDFQuad, RDFTriple
from kg_construction.domain.models import ConstructionBatch, EvidenceRef
from kg_construction.infrastructure.ontology.validation import (
    EvidenceRequiredValidator,
    ValidationReport,
)

RDF = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
RDFS = "http://www.w3.org/2000/01/rdf-schema#"
PROV = "http://www.w3.org/ns/prov#"
XSD = "http://www.w3.org/2001/XMLSchema#"


class RDFGraphBuilder:
    """Build RDFLib-serializable RDF graphs and provenance named graphs."""

    name = "rdf_graph_builder"

    def __init__(
        self,
        *,
        resource_iri: str = "urn:kg:",
        schema_iri: str = "urn:kg:schema:",
        data_graph: str = "urn:kg:graph:data",
        evidence_graph: str = "urn:kg:graph:evidence",
    ) -> None:
        self.resource_iri = resource_iri.rstrip(":") + ":"
        self.schema_iri = schema_iri.rstrip(":") + ":"
        self.data_graph = data_graph
        self.evidence_graph = evidence_graph

    def build(
        self,
        batch: ConstructionBatch,
        *,
        validation_report: ValidationReport | None = None,
    ) -> RDFGraph:
        report = validation_report or EvidenceRequiredValidator().validate(batch)
        if validation_report is None and not report.valid:
            raise ValueError("Cannot build RDF graph: unvalidated candidates without evidence.")
        invalid_ids = report.invalid_ids

        triples: list[RDFTriple] = []
        quads: list[RDFQuad] = []

        for entity in batch.entities:
            if entity.candidate_id in invalid_ids:
                continue
            subject_id = str(entity.properties.get("canonical_entity_id", entity.candidate_id))
            subject = self._resource("entity", subject_id)
            entity_triples = [
                self._iri_triple(subject, self._rdf("type"), self._schema(entity.entity_type)),
                self._literal_triple(subject, self._rdfs("label"), entity.name),
            ]
            for key, value in entity.properties.items():
                if value is None:
                    continue
                entity_triples.append(
                    self._literal_triple(
                        subject,
                        self._schema(key),
                        serialize_literal_value(value),
                        datatype=literal_datatype(value),
                    )
                )
            for evidence in entity.evidence:
                entity_triples.append(
                    self._iri_triple(
                        subject,
                        self._prov("wasDerivedFrom"),
                        self._resource("evidence", evidence.evidence_id),
                    )
                )
                quads.extend(self._evidence_quads(evidence))
            triples.extend(entity_triples)
            quads.extend(self._data_quads(entity_triples))

        for relation in batch.relations:
            if relation.candidate_id in invalid_ids:
                continue
            relation_triple = self._iri_triple(
                self._resource("entity", relation.source),
                self._schema(relation.relation_type),
                self._resource("entity", relation.target),
            )
            triples.append(relation_triple)
            quads.append(self._quad_from_triple(relation_triple, self.data_graph))
            relation_subject = self._resource("fact", relation.candidate_id)
            fact_triples = [
                self._iri_triple(relation_subject, self._rdf("type"), self._schema("Fact")),
                self._iri_triple(relation_subject, self._schema("subject"), relation_triple.subject),
                self._iri_triple(relation_subject, self._schema("predicate"), relation_triple.predicate),
                self._iri_triple(relation_subject, self._schema("object"), relation_triple.object),
            ]
            for key, value in relation.properties.items():
                if value is None:
                    continue
                fact_triples.append(
                    self._literal_triple(
                        relation_subject,
                        self._schema(key),
                        serialize_literal_value(value),
                        datatype=literal_datatype(value),
                    )
                )
            for evidence in relation.evidence:
                fact_triples.append(
                    self._iri_triple(
                        relation_subject,
                        self._prov("wasDerivedFrom"),
                        self._resource("evidence", evidence.evidence_id),
                    )
                )
                quads.extend(self._evidence_quads(evidence))
            triples.extend(fact_triples)
            quads.extend(self._data_quads(fact_triples))

        for fact in batch.facts:
            if fact.fact_id in invalid_ids:
                continue
            if fact.properties.get("rdf_subject_iri") and fact.properties.get("rdf_predicate_iri"):
                object_type = str(fact.properties.get("rdf_object_type", "iri"))
                fact_triple = RDFTriple(
                    subject=str(fact.properties.get("rdf_subject_iri", fact.subject)),
                    predicate=str(fact.properties.get("rdf_predicate_iri", fact.predicate)),
                    object=str(fact.properties.get("rdf_object", fact.object)),
                    object_type="literal" if object_type == "literal" else "iri",
                    datatype=(
                        str(fact.properties["rdf_datatype"])
                        if fact.properties.get("rdf_datatype")
                        else None
                    ),
                )
            else:
                fact_triple = self._iri_triple(
                    self._resource("resource", fact.subject),
                    self._schema(fact.predicate),
                    self._resource("resource", fact.object),
                )
            triples.append(fact_triple)
            quads.append(
                self._quad_from_triple(
                    fact_triple,
                    str(fact.properties.get("rdf_graph") or self.data_graph),
                )
            )
            for evidence in fact.evidence:
                evidence_triples = [
                    self._iri_triple(
                        self._resource("fact", fact.fact_id),
                        self._prov("wasDerivedFrom"),
                        self._resource("evidence", evidence.evidence_id),
                    )
                ]
                triples.extend(evidence_triples)
                quads.extend(self._data_quads(evidence_triples))
                quads.extend(self._evidence_quads(evidence))

        return RDFGraph(
            triples=dedupe_triples(triples),
            quads=dedupe_quads(quads),
            prefixes={
                "kg": self.schema_iri,
                "res": self.resource_iri,
                "prov": PROV,
                "rdf": RDF,
                "rdfs": RDFS,
                "xsd": XSD,
            },
            metadata={
                "builder": self.name,
                "validated": report.valid,
                "triple_count": len(triples),
                "quad_count": len(quads),
                "graph_model": "rdf",
                "data_graph": self.data_graph,
                "evidence_graph": self.evidence_graph,
                "rdflib_serializable": True,
            },
        )

    def _evidence_quads(self, evidence: EvidenceRef) -> list[RDFQuad]:
        evidence_subject = self._resource("evidence", evidence.evidence_id)
        source_subject = self._resource("source", evidence.source_id)
        triples = [
            self._iri_triple(evidence_subject, self._rdf("type"), self._schema("Evidence")),
            self._iri_triple(evidence_subject, self._prov("wasDerivedFrom"), source_subject),
            self._literal_triple(evidence_subject, self._schema("sourceId"), evidence.source_id),
            self._literal_triple(evidence_subject, self._schema("sourceName"), evidence.source_name),
            self._literal_triple(evidence_subject, self._schema("sourceType"), evidence.source_type),
            self._literal_triple(
                evidence_subject,
                self._schema("location"),
                json.dumps(evidence.location, sort_keys=True),
            ),
        ]
        if evidence.text:
            triples.append(self._literal_triple(evidence_subject, self._schema("evidenceText"), evidence.text))
        return [self._quad_from_triple(triple, self.evidence_graph) for triple in triples]

    def _data_quads(self, triples: list[RDFTriple]) -> list[RDFQuad]:
        return [self._quad_from_triple(triple, self.data_graph) for triple in triples]

    @staticmethod
    def _quad_from_triple(triple: RDFTriple, graph: str) -> RDFQuad:
        return RDFQuad(
            subject=triple.subject,
            predicate=triple.predicate,
            object=triple.object,
            object_type=triple.object_type,
            datatype=triple.datatype,
            graph=graph,
        )

    @staticmethod
    def _iri_triple(subject: str, predicate: str, object_iri: str) -> RDFTriple:
        return RDFTriple(subject=subject, predicate=predicate, object=object_iri, object_type="iri")

    @staticmethod
    def _literal_triple(
        subject: str,
        predicate: str,
        value: str,
        *,
        datatype: str | None = f"{XSD}string",
    ) -> RDFTriple:
        return RDFTriple(
            subject=subject,
            predicate=predicate,
            object=value,
            object_type="literal",
            datatype=datatype,
        )

    def _resource(self, namespace: str, value: str) -> str:
        return f"{self.resource_iri}{namespace}:{safe_iri_fragment(value)}"

    def _schema(self, value: str) -> str:
        return f"{self.schema_iri}{safe_iri_fragment(value)}"

    @staticmethod
    def _rdf(value: str) -> str:
        return f"{RDF}{value}"

    @staticmethod
    def _rdfs(value: str) -> str:
        return f"{RDFS}{value}"

    @staticmethod
    def _prov(value: str) -> str:
        return f"{PROV}{value}"


def safe_iri_fragment(value: str) -> str:
    """Return a conservative fragment safe for generated URN-like IRIs."""

    cleaned = str(value).strip().replace(" ", "_").replace("/", "_").replace("#", "_")
    cleaned = "".join(character if character.isalnum() or character in "_:-." else "_" for character in cleaned)
    return cleaned or "unknown"


def serialize_literal_value(value: Any) -> str:
    """Serialize property values before RDF literal conversion."""

    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        return str(value)
    if isinstance(value, list | dict):
        return json.dumps(value, sort_keys=True, default=str)
    return str(value)


def literal_datatype(value: Any) -> str:
    """Infer an XSD datatype for simple scalar values."""

    if isinstance(value, bool):
        return f"{XSD}boolean"
    if isinstance(value, int):
        return f"{XSD}integer"
    if isinstance(value, float):
        return f"{XSD}decimal"
    if isinstance(value, datetime):
        return f"{XSD}dateTime"
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.isdigit() or (stripped.startswith("-") and stripped[1:].isdigit()):
            return f"{XSD}integer"
        try:
            float(stripped)
        except ValueError:
            return f"{XSD}string"
        return f"{XSD}decimal"
    return f"{XSD}string"


def dedupe_triples(triples: list[RDFTriple]) -> list[RDFTriple]:
    """Return triples in stable order without duplicates."""

    return sorted({triple.triple_id: triple for triple in triples}.values(), key=lambda item: item.triple_id)


def dedupe_quads(quads: list[RDFQuad]) -> list[RDFQuad]:
    """Return quads in stable order without duplicates."""

    return sorted({quad.quad_id: quad for quad in quads}.values(), key=lambda item: item.quad_id)


__all__ = [
    "RDFGraphBuilder",
    "literal_datatype",
    "safe_iri_fragment",
    "serialize_literal_value",
]
