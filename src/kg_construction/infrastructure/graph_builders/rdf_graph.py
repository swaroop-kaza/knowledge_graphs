"""Standards-based RDF triple and quad builder."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from kg_construction.domain.graph_objects import RDFGraph, RDFQuad, RDFTriple
from kg_construction.domain.models import ConstructionBatch, EvidenceRef
from kg_construction.domain.validation import ValidationReport
from kg_construction.infrastructure.ontology.validation import EvidenceRequiredValidator

RDF = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
RDFS = "http://www.w3.org/2000/01/rdf-schema#"
PROV = "http://www.w3.org/ns/prov#"
XSD = "http://www.w3.org/2001/XMLSchema#"
_XSD_BY_JSON_TYPE = {
    "string": f"{XSD}string",
    "integer": f"{XSD}integer",
    "number": f"{XSD}decimal",
    "boolean": f"{XSD}boolean",
    "date": f"{XSD}date",
    "datetime": f"{XSD}dateTime",
    "object": f"{XSD}string",
    "array": f"{XSD}string",
}


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
        declared_datatypes = _resolved_property_datatypes(batch)
        source_text_by_id = {
            source.source_id: source.content
            for source in batch.source_records
            if source.content is not None
        }
        entity_resource_ids = {
            str(identifier)
            for entity in batch.entities
            for identifier in (
                entity.candidate_id,
                entity.properties.get("canonical_entity_id"),
            )
            if identifier is not None
        }

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
                self._literal_triple(
                    subject,
                    self._schema("extractorName"),
                    entity.extractor_name,
                ),
                self._literal_triple(
                    subject,
                    self._schema("technique"),
                    entity.technique,
                ),
                self._literal_triple(
                    subject,
                    self._schema("schemaVersion"),
                    entity.schema_version,
                ),
                self._literal_triple(
                    subject,
                    self._schema("confidence"),
                    serialize_literal_value(entity.confidence),
                    datatype=literal_datatype(entity.confidence),
                ),
            ]
            identity_layer = entity.properties.get("identity_layer")
            semantic_entity_type = entity.properties.get("semantic_entity_type")
            if identity_layer == "observed":
                entity_triples.append(
                    self._iri_triple(
                        subject,
                        self._rdf("type"),
                        self._schema("ObservedEntity"),
                    )
                )
            if identity_layer == "canonical":
                entity_triples.append(
                    self._iri_triple(
                        subject,
                        self._rdf("type"),
                        self._schema("CanonicalEntity"),
                    )
                )
            if (
                entity.entity_type == "Mention"
                and entity.properties.get("resolution_status")
                in {"linked", "externally_linked"}
            ):
                entity_triples.append(
                    self._iri_triple(
                        subject,
                        self._rdf("type"),
                        self._schema("LinkedMention"),
                    )
                )
            if (
                identity_layer == "observed"
                and entity.properties.get("identity_status") == "externally_linked"
            ):
                entity_triples.append(
                    self._iri_triple(
                        subject,
                        self._rdf("type"),
                        self._schema("ExternallyLinkedObservedEntity"),
                    )
                )
            if (
                identity_layer == "canonical"
                and (
                    entity.properties.get("identity_status") == "externally_linked"
                    or entity.properties.get("external_system") is not None
                    or entity.properties.get("external_id") is not None
                )
            ):
                entity_triples.append(
                    self._iri_triple(
                        subject,
                        self._rdf("type"),
                        self._schema("ExternalCanonicalEntity"),
                    )
                )
            if (
                entity.entity_type == "Event"
                and entity.properties.get("domain_claim_id") is not None
            ):
                entity_triples.append(
                    self._iri_triple(
                        subject,
                        self._rdf("type"),
                        self._schema("ClaimEvent"),
                    )
                )
            if (
                identity_layer == "canonical"
                and isinstance(semantic_entity_type, str)
                and semantic_entity_type.strip()
            ):
                entity_triples.append(
                    self._iri_triple(
                        subject,
                        self._rdf("type"),
                        self._schema(semantic_entity_type),
                    )
                )
            for key, value in _candidate_provenance_properties(
                entity.provenance_metadata
            ).items():
                entity_triples.append(
                    self._literal_triple(
                        subject,
                        self._schema(key),
                        serialize_literal_value(value),
                        datatype=literal_datatype(value),
                    )
                )
            for key, value in entity.properties.items():
                if value is None:
                    continue
                datatype = declared_datatypes.get(
                    (entity.entity_type, key),
                    literal_datatype(value),
                )
                entity_triples.append(
                    self._literal_triple(
                        subject,
                        self._schema(key),
                        serialize_literal_value(value),
                        datatype=datatype,
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
                quads.extend(
                    self._evidence_quads(
                        evidence,
                        source_text=source_text_by_id.get(evidence.source_id),
                    )
                )
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
                self._literal_triple(
                    relation_subject,
                    self._rdfs("label"),
                    relation.relation_type,
                ),
                self._iri_triple(relation_subject, self._schema("subject"), relation_triple.subject),
                self._iri_triple(relation_subject, self._schema("predicate"), relation_triple.predicate),
                self._iri_triple(relation_subject, self._schema("object"), relation_triple.object),
                self._literal_triple(
                    relation_subject,
                    self._schema("extractorName"),
                    relation.extractor_name,
                ),
                self._literal_triple(
                    relation_subject,
                    self._schema("technique"),
                    relation.technique,
                ),
                self._literal_triple(
                    relation_subject,
                    self._schema("schemaVersion"),
                    relation.schema_version,
                ),
                self._literal_triple(
                    relation_subject,
                    self._schema("confidence"),
                    serialize_literal_value(relation.confidence),
                    datatype=literal_datatype(relation.confidence),
                ),
            ]
            for key, value in _candidate_provenance_properties(
                relation.provenance_metadata
            ).items():
                fact_triples.append(
                    self._literal_triple(
                        relation_subject,
                        self._schema(key),
                        serialize_literal_value(value),
                        datatype=literal_datatype(value),
                    )
                )
            for key, value in relation.properties.items():
                if value is None:
                    continue
                property_name = (
                    f"statementMetadata_{key}"
                    if key in {"subject", "predicate", "object"}
                    else key
                )
                fact_triples.append(
                    self._literal_triple(
                        relation_subject,
                        self._schema(property_name),
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
                quads.extend(
                    self._evidence_quads(
                        evidence,
                        source_text=source_text_by_id.get(evidence.source_id),
                    )
                )
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
                    self._resource(
                        (
                            "entity"
                            if fact.subject in entity_resource_ids
                            else "resource"
                        ),
                        fact.subject,
                    ),
                    self._schema(fact.predicate),
                    self._resource(
                        (
                            "entity"
                            if fact.object in entity_resource_ids
                            else "resource"
                        ),
                        fact.object,
                    ),
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
                quads.extend(
                    self._evidence_quads(
                        evidence,
                        source_text=source_text_by_id.get(evidence.source_id),
                    )
                )

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
                "identity_policy_version": "kg-identity-v2",
                "data_graph": self.data_graph,
                "evidence_graph": self.evidence_graph,
                "rdflib_serializable": True,
            },
        )

    def _evidence_quads(
        self,
        evidence: EvidenceRef,
        *,
        source_text: str | None,
    ) -> list[RDFQuad]:
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
        if evidence.location.get("kind") == "span":
            start = evidence.location.get("start")
            end = evidence.location.get("end")
            document_version = evidence.metadata.get("document_version")
            domain_evidence_id = evidence.metadata.get("domain_evidence_id")
            exact = (
                isinstance(start, int)
                and isinstance(end, int)
                and 0 <= start < end
                and source_text is not None
                and evidence.text is not None
                and source_text[start:end] == evidence.text
            )
            triples.extend(
                [
                    self._iri_triple(
                        evidence_subject,
                        self._rdf("type"),
                        self._schema("ExactEvidence"),
                    ),
                    self._literal_triple(
                        evidence_subject,
                        self._schema("start"),
                        serialize_literal_value(start),
                        datatype=literal_datatype(start),
                    ),
                    self._literal_triple(
                        evidence_subject,
                        self._schema("end"),
                        serialize_literal_value(end),
                        datatype=literal_datatype(end),
                    ),
                    self._literal_triple(
                        evidence_subject,
                        self._schema("exact"),
                        serialize_literal_value(exact),
                        datatype=literal_datatype(exact),
                    ),
                ]
            )
            if document_version is not None:
                triples.append(
                    self._literal_triple(
                        evidence_subject,
                        self._schema("documentVersion"),
                        serialize_literal_value(document_version),
                    )
                )
            if domain_evidence_id is not None:
                triples.append(
                    self._literal_triple(
                        evidence_subject,
                        self._schema("domainEvidenceId"),
                        serialize_literal_value(domain_evidence_id),
                    )
                )
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
        unsigned = stripped[1:] if stripped.startswith("-") else stripped
        if unsigned.isdigit():
            # Never coerce a lexical value with significant leading zeros.
            if len(unsigned) > 1 and unsigned.startswith("0"):
                return f"{XSD}string"
            return f"{XSD}integer"
        try:
            float(stripped)
        except ValueError:
            return f"{XSD}string"
        return f"{XSD}decimal"
    return f"{XSD}string"


def _resolved_property_datatypes(
    batch: ConstructionBatch,
) -> dict[tuple[str, str], str]:
    """Read explicit property datatypes from serialized resolved plans."""

    payload = batch.metadata.get("resolved_modeling_plans", {})
    if not isinstance(payload, dict):
        return {}
    result: dict[tuple[str, str], str] = {}
    for raw_plan in payload.values():
        if not isinstance(raw_plan, dict):
            continue
        properties = raw_plan.get("properties", [])
        if not isinstance(properties, list):
            continue
        for raw_property in properties:
            if not isinstance(raw_property, dict):
                continue
            class_name = raw_property.get("class_name")
            name = raw_property.get("name")
            datatype = raw_property.get("datatype")
            if (
                not isinstance(class_name, str)
                or not isinstance(name, str)
                or not isinstance(datatype, str)
            ):
                continue
            result[(class_name, name)] = _XSD_BY_JSON_TYPE.get(
                datatype,
                f"{XSD}string",
            )
    return result


def _candidate_provenance_properties(
    provenance: dict[str, Any],
) -> dict[str, Any]:
    """Project provider provenance into stable RDF properties.

    Candidate provenance is first-class graph data, not exporter-only metadata.
    Only the governed keys are emitted so arbitrary provider payloads cannot
    silently expand the publication vocabulary.
    """

    mapping = {
        "provider": "provider",
        "model_version": "modelVersion",
        "configuration_id": "configurationId",
        "extractor": "sourceExtractor",
        "resolution_provider": "resolutionProvider",
    }
    return {
        rdf_name: provenance[source_name]
        for source_name, rdf_name in mapping.items()
        if provenance.get(source_name) is not None
    }


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
