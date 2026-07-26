"""Property graph builder for validated construction batches."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from datetime import datetime
from typing import Any, TypedDict

from kg_construction.domain.graph_objects import PropertyEdge, PropertyGraph, PropertyNode
from kg_construction.domain.models import (
    ConstructionBatch,
    EntityCandidate,
    EvidenceRef,
    GraphFact,
    RelationCandidate,
    canonical_json,
    stable_id,
)
from kg_construction.domain.validation import ValidationReport
from kg_construction.infrastructure.ontology.validation import EvidenceRequiredValidator

JsonDict = dict[str, Any]


class _PropertyValueSupport(TypedDict):
    value: object
    candidate_ids: list[str]


_SURVIVORSHIP_POLICY = (
    "highest_confidence_then_evidence_count_then_candidate_id_then_canonical_payload"
)
_UNION_COLLECTION_PROPERTIES = frozenset(
    {"raw_values", "source_occurrences"}
)

# These fields describe where/how an assertion was extracted.  They must be
# retained as assertion support, but they do not change a timeless semantic
# relationship's identity.
_ASSERTION_ONLY_RELATION_PROPERTIES = {
    "_array_index",
    "_json_path",
    "_json_property_paths",
    "array_index",
    "line_number",
    "original_source",
    "original_target",
    "parser",
    "source_column",
    "source_key",
    "source_name",
    "source_path",
    "source_value",
    "target_column",
    "target_name",
}

_ASSERTION_ONLY_FACT_PROPERTIES = {
    "csvw_column",
    "csvw_datatype",
    "csvw_propertyUrl",
    "csvw_role",
    "csvw_valueUrl",
    "join_conditions",
    "mapping_id",
    "object_map",
    "original_object",
    "original_subject",
    "parent_evidence_id",
    "parent_triples_map",
    "parser",
    "predicate_map",
    "rdf_object",
    "rdf_object_type",
    "rdf_datatype",
    "rdf_graph",
    "rdf_language",
    "rdf_predicate_iri",
    "rdf_subject_iri",
    "raw_value",
    "referencing_object_map",
    "source_name",
    "source_path",
    "source_reference",
    "standard",
    "subject_map",
    "virtual_mapping_table",
}

# A relationship carrying one of these fields describes a particular
# occurrence or validity interval.  Values remain in the semantic key, so two
# purchases/calls/events between the same endpoints are never collapsed merely
# because their (subject, predicate, object) tuple matches.
_OCCURRENCE_PROPERTIES = {
    "assertion_instance_id",
    "end_time",
    "event.id",
    "eventId",
    "event_id",
    "message_id",
    "observed_at",
    "occurred_at",
    "occurrenceId",
    "occurrence_id",
    "relation_instance_id",
    "request_id",
    "span_id",
    "start_time",
    "timestamp",
    "trace_id",
    "transaction_id",
    "valid_from",
    "valid_to",
}

_EVENT_ENTITY_TYPES = {
    "event",
    "interaction",
    "logevent",
    "message",
    "observation",
    "occurrence",
    "span",
    "transaction",
}


def _json_value(value: object) -> object:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        items = [_json_value(item) for item in value]
        return sorted(items, key=canonical_json)
    if isinstance(value, dict):
        return {str(key): _json_value(value[key]) for key in sorted(value, key=str)}
    return value


def _json_dict(value: JsonDict) -> JsonDict:
    serialized = _json_value(value)
    return serialized if isinstance(serialized, dict) else {}


def _dedupe_json_values(values: Iterable[object]) -> list[object]:
    """Return JSON values once each in a deterministic order."""

    unique: dict[str, object] = {}
    for value in values:
        serialized = _json_value(value)
        unique.setdefault(canonical_json(serialized), serialized)
    return [unique[key] for key in sorted(unique)]


def _ordered_evidence(evidence: Iterable[EvidenceRef]) -> list[EvidenceRef]:
    """Union evidence by its content-addressed ID deterministically."""

    evidence_by_id: dict[str, EvidenceRef] = {}
    for item in evidence:
        current = evidence_by_id.get(item.evidence_id)
        if current is None:
            evidence_by_id[item.evidence_id] = item
            continue
        # Metadata is not part of EvidenceRef's ID.  If enrichment produced two
        # forms, choose deterministically instead of depending on input order.
        current_key = canonical_json(current.model_dump(mode="json"))
        item_key = canonical_json(item.model_dump(mode="json"))
        if item_key < current_key:
            evidence_by_id[item.evidence_id] = item
    return [evidence_by_id[key] for key in sorted(evidence_by_id)]


def _evidence_properties(evidence: Iterable[EvidenceRef]) -> JsonDict:
    ordered = _ordered_evidence(evidence)
    locations = _dedupe_json_values(item.location for item in ordered if item.location)
    source_paths = sorted(
        {
            str(item.location["path"])
            for item in ordered
            if item.location.get("kind") == "jsonpath" and item.location.get("path")
        }
    )
    return {
        "evidence_ids": [item.evidence_id for item in ordered],
        "source_ids": sorted({item.source_id for item in ordered}),
        "source_names": sorted({item.source_name for item in ordered}),
        "evidence_texts": sorted({item.text for item in ordered if item.text}),
        "evidence_locations": locations,
        "source_paths": source_paths,
        "evidence_support": [
            {
                "evidence_id": item.evidence_id,
                "source_id": item.source_id,
                "source_type": item.source_type,
                "source_name": item.source_name,
                "location": _json_value(item.location),
                "text": item.text,
                "metadata": _json_value(item.metadata),
                "created_at": item.created_at.isoformat(),
            }
            for item in ordered
        ],
    }


def _assertion_id(candidate: EntityCandidate | RelationCandidate | GraphFact) -> str:
    return candidate.fact_id if isinstance(candidate, GraphFact) else candidate.candidate_id


def _candidate_rank(
    candidate: EntityCandidate | RelationCandidate | GraphFact,
) -> tuple[float, int, str, str]:
    """Sort strongest candidates first without using ingestion order."""

    return (
        -candidate.confidence,
        -len({item.evidence_id for item in candidate.evidence}),
        _assertion_id(candidate),
        canonical_json(candidate.model_dump(mode="json")),
    )


def _merge_properties(
    candidates: list[EntityCandidate] | list[RelationCandidate] | list[GraphFact],
) -> tuple[JsonDict, JsonDict, dict[str, str]]:
    """Apply deterministic field survivorship and retain conflicting values."""

    ranked = sorted(candidates, key=_candidate_rank)
    merged: JsonDict = {}
    conflicts: JsonDict = {}
    property_sources: dict[str, str] = {}
    keys = sorted({key for candidate in ranked for key in candidate.properties})
    for key in keys:
        if key in _UNION_COLLECTION_PROPERTIES:
            items: list[object] = []
            supporting_candidate_ids: list[str] = []
            for candidate in ranked:
                value = candidate.properties.get(key)
                if not isinstance(value, list):
                    continue
                items.extend(value)
                supporting_candidate_ids.append(_assertion_id(candidate))
            if items:
                merged[key] = _dedupe_json_values(items)
                property_sources[key] = ",".join(
                    sorted(set(supporting_candidate_ids))
                )
            continue
        supporting_values: dict[str, _PropertyValueSupport] = {}
        winner_found = False
        for candidate in ranked:
            if key not in candidate.properties or candidate.properties[key] is None:
                continue
            value = _json_value(candidate.properties[key])
            value_key = canonical_json(value)
            support = supporting_values.setdefault(
                value_key,
                {"value": value, "candidate_ids": []},
            )
            candidate_id = _assertion_id(candidate)
            support["candidate_ids"].append(candidate_id)
            if not winner_found:
                merged[key] = value
                property_sources[key] = candidate_id
                winner_found = True
        if len(supporting_values) > 1:
            conflicts[key] = [
                {
                    "value": supporting_values[value_key]["value"],
                    "candidate_ids": sorted(
                        set(supporting_values[value_key]["candidate_ids"])
                    ),
                }
                for value_key in sorted(supporting_values)
            ]
    return merged, conflicts, property_sources


def _entity_provenance(candidate: EntityCandidate) -> JsonDict:
    return {
        "candidate_id": candidate.candidate_id,
        "entity_type": candidate.entity_type,
        "name": candidate.name,
        "confidence": candidate.confidence,
        "extractor_name": candidate.extractor_name,
        "technique": candidate.technique,
        "schema_version": candidate.schema_version,
        "created_at": candidate.created_at.isoformat(),
        "evidence_ids": sorted({item.evidence_id for item in candidate.evidence}),
        "provenance_metadata": _json_value(candidate.provenance_metadata),
    }


def _relation_provenance(candidate: RelationCandidate) -> JsonDict:
    return {
        "candidate_id": candidate.candidate_id,
        "source": candidate.source,
        "relation_type": candidate.relation_type,
        "target": candidate.target,
        "confidence": candidate.confidence,
        "extractor_name": candidate.extractor_name,
        "technique": candidate.technique,
        "schema_version": candidate.schema_version,
        "created_at": candidate.created_at.isoformat(),
        "properties": _json_value(candidate.properties),
        "evidence_ids": sorted({item.evidence_id for item in candidate.evidence}),
        "provenance_metadata": _json_value(candidate.provenance_metadata),
    }


def _fact_provenance(fact: GraphFact) -> JsonDict:
    return {
        "fact_id": fact.fact_id,
        "subject": fact.subject,
        "predicate": fact.predicate,
        "object": fact.object,
        "confidence": fact.confidence,
        "extractor_name": fact.extractor_name,
        "technique": fact.technique,
        "schema_version": fact.schema_version,
        "validation_status": fact.validation_status,
        "created_at": fact.created_at.isoformat(),
        "properties": _json_value(fact.properties),
        "evidence_ids": sorted({item.evidence_id for item in fact.evidence}),
        "provenance_metadata": _json_value(fact.provenance_metadata),
    }


def _is_event_entity_type(entity_type: str) -> bool:
    compact = "".join(character for character in entity_type.casefold() if character.isalnum())
    return compact in _EVENT_ENTITY_TYPES or compact.endswith("event")


def _semantic_relation_key(
    relation: RelationCandidate,
    *,
    source_id: str,
    target_id: str,
    endpoint_types: set[str],
) -> str:
    """Return a conservative identity key for one semantic relationship.

    Arbitrary properties are treated as semantic qualifiers unless explicitly
    known to be extraction provenance.  Event-like assertions without a stable
    occurrence identifier remain separate rather than risking an over-merge.
    """

    semantic_properties = {
        key: _json_value(relation.properties[key])
        for key in sorted(relation.properties)
        if key not in _ASSERTION_ONLY_RELATION_PROPERTIES
    }
    occurrence_properties = {
        key: _json_value(value)
        for key, value in sorted(
            {**relation.provenance_metadata, **relation.properties}.items()
        )
        if key in _OCCURRENCE_PROPERTIES and value not in (None, "")
    }
    if occurrence_properties:
        semantic_properties["_occurrence"] = occurrence_properties

    explicitly_event_like = bool(
        relation.properties.get("event_like")
        or relation.provenance_metadata.get("event_like")
        or any(_is_event_entity_type(entity_type) for entity_type in endpoint_types)
    )
    if explicitly_event_like and not occurrence_properties:
        semantic_properties["_unresolved_occurrence_candidate_id"] = relation.candidate_id

    return canonical_json(
        {
            "source": source_id,
            "type": relation.relation_type,
            "target": target_id,
            "qualifiers": semantic_properties,
        }
    )


def _semantic_fact_key(
    fact: GraphFact,
    *,
    subject_id: str,
    object_id: str,
) -> str:
    """Return the semantic identity of an RDF/generic graph assertion."""

    semantic_properties = {
        key: _json_value(fact.properties[key])
        for key in sorted(fact.properties)
        if key not in _ASSERTION_ONLY_FACT_PROPERTIES
    }
    occurrence_properties = {
        key: _json_value(value)
        for key, value in sorted({**fact.provenance_metadata, **fact.properties}.items())
        if key in _OCCURRENCE_PROPERTIES and value not in (None, "")
    }
    if occurrence_properties:
        semantic_properties["_occurrence"] = occurrence_properties
    return canonical_json(
        {
            "source": subject_id,
            "type": fact.predicate,
            "target": object_id,
            "object_type": str(fact.properties.get("rdf_object_type", "iri")),
            "datatype": fact.properties.get("rdf_datatype"),
            "language": fact.properties.get("rdf_language"),
            "graph": fact.properties.get("rdf_graph"),
            "qualifiers": semantic_properties,
        }
    )


def _ensure_resource_node(
    nodes: dict[str, PropertyNode],
    *,
    node_id: str,
    role: str,
) -> None:
    if node_id in nodes:
        return
    nodes[node_id] = PropertyNode(
        node_id=node_id,
        labels=["Resource"],
        properties={"iri": node_id, "resource_role": role},
    )


class PropertyGraphBuilder:
    """Build GraphRAG-ready property graphs after validation."""

    name = "property_graph_builder"

    def build(
        self,
        batch: ConstructionBatch,
        *,
        validation_report: ValidationReport | None = None,
        include_fact_nodes: bool = True,
    ) -> PropertyGraph:
        report = validation_report or EvidenceRequiredValidator().validate(batch)
        if validation_report is None and not report.valid:
            raise ValueError("Cannot build graph: unvalidated candidates without evidence.")
        invalid_ids = report.invalid_ids

        nodes: dict[str, PropertyNode] = {}
        edges: dict[str, PropertyEdge] = {}
        entity_groups: dict[str, list[EntityCandidate]] = defaultdict(list)
        endpoint_ids: dict[str, str] = {}
        endpoint_types: dict[str, set[str]] = defaultdict(set)

        for entity in batch.entities:
            if entity.candidate_id in invalid_ids:
                continue
            canonical_id = str(entity.properties.get("canonical_entity_id", entity.candidate_id))
            entity_groups[canonical_id].append(entity)
            endpoint_ids[entity.candidate_id] = canonical_id
            endpoint_types[entity.candidate_id].add(entity.entity_type)
            endpoint_types[canonical_id].add(entity.entity_type)

        for node_id, candidates in sorted(entity_groups.items()):
            ranked = sorted(candidates, key=_candidate_rank)
            entity_survivor = ranked[0]
            merged_properties, conflicts, property_sources = _merge_properties(ranked)
            evidence = _ordered_evidence(
                item for candidate in ranked for item in candidate.evidence
            )
            labels = [
                "Entity",
                *sorted(
                    {
                        candidate.entity_type
                        for candidate in ranked
                        if candidate.entity_type != "Entity"
                    }
                ),
            ]
            properties: JsonDict = {
                **merged_properties,
                "candidate_id": entity_survivor.candidate_id,
                "candidate_ids": sorted({candidate.candidate_id for candidate in ranked}),
                "candidate_names": sorted({candidate.name for candidate in ranked}),
                "name": entity_survivor.name,
                "confidence": entity_survivor.confidence,
                "extractor_name": entity_survivor.extractor_name,
                "technique": entity_survivor.technique,
                "schema_version": entity_survivor.schema_version,
                "created_at": entity_survivor.created_at,
                "survivor_candidate_id": entity_survivor.candidate_id,
                "survivorship_policy": _SURVIVORSHIP_POLICY,
                "property_sources": property_sources,
                "candidate_provenance": [
                    _entity_provenance(candidate)
                    for candidate in sorted(ranked, key=_candidate_rank)
                ],
                **_evidence_properties(evidence),
            }
            if conflicts:
                properties["property_conflicts"] = conflicts
            nodes[node_id] = PropertyNode(
                node_id=node_id,
                labels=labels,
                properties=_json_dict(properties),
            )

        relation_groups: dict[str, list[tuple[RelationCandidate, str, str]]] = defaultdict(list)
        for relation in batch.relations:
            if relation.candidate_id in invalid_ids:
                continue
            source_id = endpoint_ids.get(relation.source, relation.source)
            target_id = endpoint_ids.get(relation.target, relation.target)
            relation_endpoint_types = {
                *endpoint_types.get(relation.source, set()),
                *endpoint_types.get(relation.target, set()),
                *endpoint_types.get(source_id, set()),
                *endpoint_types.get(target_id, set()),
            }
            semantic_key = _semantic_relation_key(
                relation,
                source_id=source_id,
                target_id=target_id,
                endpoint_types=relation_endpoint_types,
            )
            relation_groups[semantic_key].append((relation, source_id, target_id))

        for semantic_key, relation_assertions in sorted(relation_groups.items()):
            ranked_relations = sorted(
                (item[0] for item in relation_assertions),
                key=_candidate_rank,
            )
            relation_survivor = ranked_relations[0]
            source_id, target_id = next(
                (source, target)
                for relation, source, target in relation_assertions
                if relation.candidate_id == relation_survivor.candidate_id
            )
            edge_id = stable_id("edge", "semantic_relation", semantic_key)
            merged_properties, conflicts, property_sources = _merge_properties(ranked_relations)
            evidence = _ordered_evidence(
                item for relation in ranked_relations for item in relation.evidence
            )
            edge_properties: JsonDict = {
                **merged_properties,
                "semantic_relation_key": semantic_key,
                "candidate_id": relation_survivor.candidate_id,
                "candidate_ids": sorted({item.candidate_id for item in ranked_relations}),
                "assertion_count": len({item.candidate_id for item in ranked_relations}),
                "confidence": relation_survivor.confidence,
                "extractor_name": relation_survivor.extractor_name,
                "technique": relation_survivor.technique,
                "schema_version": relation_survivor.schema_version,
                "created_at": relation_survivor.created_at,
                "survivor_candidate_id": relation_survivor.candidate_id,
                "survivorship_policy": _SURVIVORSHIP_POLICY,
                "property_sources": property_sources,
                "assertion_support": [
                    _relation_provenance(relation)
                    for relation in sorted(ranked_relations, key=_candidate_rank)
                ],
                **_evidence_properties(evidence),
            }
            if conflicts:
                edge_properties["property_conflicts"] = conflicts
            edges[edge_id] = PropertyEdge(
                edge_id=edge_id,
                source_id=source_id,
                target_id=target_id,
                type=relation_survivor.relation_type,
                properties=_json_dict(edge_properties),
            )

            if include_fact_nodes:
                for relation in sorted(ranked_relations, key=_candidate_rank):
                    fact_properties = {
                        "candidate_id": relation.candidate_id,
                        "confidence": relation.confidence,
                        "extractor_name": relation.extractor_name,
                        "technique": relation.technique,
                        "schema_version": relation.schema_version,
                        "created_at": relation.created_at,
                        "canonical_edge_id": edge_id,
                        **relation.properties,
                        **_evidence_properties(relation.evidence),
                        "provenance_metadata": _json_value(relation.provenance_metadata),
                    }
                    nodes[relation.candidate_id] = PropertyNode(
                        node_id=relation.candidate_id,
                        labels=["Fact", "RelationshipFact", relation.relation_type],
                        properties=_json_dict(fact_properties),
                    )

        fact_groups: dict[str, list[tuple[GraphFact, str, str]]] = defaultdict(list)
        for fact in batch.facts:
            if fact.fact_id in invalid_ids:
                continue
            subject_id = endpoint_ids.get(fact.subject, fact.subject)
            object_type = str(fact.properties.get("rdf_object_type", "iri"))
            if object_type == "literal":
                object_id = stable_id(
                    "literal",
                    fact.object,
                    fact.properties.get("rdf_datatype", ""),
                    fact.properties.get("rdf_language", ""),
                )
            else:
                object_id = endpoint_ids.get(fact.object, fact.object)
            semantic_key = _semantic_fact_key(
                fact,
                subject_id=subject_id,
                object_id=object_id,
            )
            fact_groups[semantic_key].append((fact, subject_id, object_id))

        for semantic_key, fact_assertions in sorted(fact_groups.items()):
            ranked_facts = sorted(
                (item[0] for item in fact_assertions),
                key=_candidate_rank,
            )
            fact_survivor = ranked_facts[0]
            subject_id, object_id = next(
                (subject, object_value)
                for asserted_fact, subject, object_value in fact_assertions
                if asserted_fact.fact_id == fact_survivor.fact_id
            )
            object_type = str(fact_survivor.properties.get("rdf_object_type", "iri"))
            _ensure_resource_node(nodes, node_id=subject_id, role="fact_subject")
            if object_type == "literal":
                if object_id not in nodes:
                    literal_properties = {
                        "value": fact_survivor.object,
                        "datatype": fact_survivor.properties.get("rdf_datatype"),
                        "language": fact_survivor.properties.get("rdf_language"),
                    }
                    nodes[object_id] = PropertyNode(
                        node_id=object_id,
                        labels=["Literal"],
                        properties=_json_dict(literal_properties),
                    )
            else:
                _ensure_resource_node(nodes, node_id=object_id, role="fact_object")

            merged_properties, conflicts, property_sources = _merge_properties(
                ranked_facts
            )
            evidence = _ordered_evidence(
                item for fact in ranked_facts for item in fact.evidence
            )
            edge_id = stable_id("edge", "semantic_fact", semantic_key)
            fact_edge_properties: JsonDict = {
                **merged_properties,
                "semantic_relation_key": semantic_key,
                "fact_id": fact_survivor.fact_id,
                "fact_ids": sorted({fact.fact_id for fact in ranked_facts}),
                "assertion_count": len({fact.fact_id for fact in ranked_facts}),
                "subject": fact_survivor.subject,
                "predicate": fact_survivor.predicate,
                "object": fact_survivor.object,
                "confidence": fact_survivor.confidence,
                "extractor_name": fact_survivor.extractor_name,
                "technique": fact_survivor.technique,
                "schema_version": fact_survivor.schema_version,
                "validation_status": fact_survivor.validation_status,
                "created_at": fact_survivor.created_at,
                "survivor_fact_id": fact_survivor.fact_id,
                "survivorship_policy": _SURVIVORSHIP_POLICY,
                "property_sources": property_sources,
                "assertion_support": [
                    _fact_provenance(fact)
                    for fact in sorted(ranked_facts, key=_candidate_rank)
                ],
                **_evidence_properties(evidence),
            }
            if conflicts:
                fact_edge_properties["property_conflicts"] = conflicts
            edges[edge_id] = PropertyEdge(
                edge_id=edge_id,
                source_id=subject_id,
                target_id=object_id,
                type=fact_survivor.predicate,
                properties=_json_dict(fact_edge_properties),
            )

            if include_fact_nodes:
                for asserted_fact in sorted(ranked_facts, key=_candidate_rank):
                    fact_properties = {
                        "subject": asserted_fact.subject,
                        "predicate": asserted_fact.predicate,
                        "object": asserted_fact.object,
                        "confidence": asserted_fact.confidence,
                        "extractor_name": asserted_fact.extractor_name,
                        "technique": asserted_fact.technique,
                        "schema_version": asserted_fact.schema_version,
                        "validation_status": asserted_fact.validation_status,
                        "created_at": asserted_fact.created_at,
                        "canonical_edge_id": edge_id,
                        **asserted_fact.properties,
                        **_evidence_properties(asserted_fact.evidence),
                        "provenance_metadata": _json_value(
                            asserted_fact.provenance_metadata
                        ),
                    }
                    nodes[asserted_fact.fact_id] = PropertyNode(
                        node_id=asserted_fact.fact_id,
                        labels=["Fact", asserted_fact.predicate],
                        properties=_json_dict(fact_properties),
                    )

        ordered_nodes = [nodes[node_id] for node_id in sorted(nodes)]
        ordered_edges = [edges[edge_id] for edge_id in sorted(edges)]
        return PropertyGraph(
            nodes=ordered_nodes,
            edges=ordered_edges,
            metadata={
                "builder": self.name,
                "validated": report.valid,
                "finding_count": len(report.findings),
                "node_count": len(nodes),
                "edge_count": len(edges),
                "graph_model": "property_graph",
                "graphrag_ready": True,
                "identity_policy_version": "kg-identity-v2",
                "canonical_node_survivorship": _SURVIVORSHIP_POLICY,
                "semantic_relation_consolidation": True,
            },
        )


__all__ = ["PropertyGraphBuilder"]
