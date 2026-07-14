"""Property graph builder for validated construction batches."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from kg_construction.domain.graph_objects import PropertyEdge, PropertyGraph, PropertyNode
from kg_construction.domain.models import ConstructionBatch, EvidenceRef, stable_id
from kg_construction.infrastructure.ontology.validation import (
    EvidenceRequiredValidator,
    ValidationReport,
)

JsonDict = dict[str, Any]


def _json_value(value: object) -> object:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    return value


def _json_dict(value: JsonDict) -> JsonDict:
    serialized = _json_value(value)
    return serialized if isinstance(serialized, dict) else {}


def _evidence_properties(evidence: list[EvidenceRef]) -> JsonDict:
    locations = [item.location for item in evidence if item.location]
    source_paths = sorted(
        {
            str(item.location["path"])
            for item in evidence
            if item.location.get("kind") == "jsonpath" and item.location.get("path")
        }
    )
    return {
        "evidence_ids": [item.evidence_id for item in evidence],
        "source_ids": sorted({item.source_id for item in evidence}),
        "source_names": sorted({item.source_name for item in evidence}),
        "evidence_texts": [item.text for item in evidence if item.text],
        "evidence_locations": locations,
        "source_paths": source_paths,
    }


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

        for entity in batch.entities:
            if entity.candidate_id in invalid_ids:
                continue
            canonical_id = str(entity.properties.get("canonical_entity_id", entity.candidate_id))
            node_id = canonical_id
            labels = ["Entity", entity.entity_type]
            properties = {
                "candidate_id": entity.candidate_id,
                "name": entity.name,
                "confidence": entity.confidence,
                "extractor_name": entity.extractor_name,
                "technique": entity.technique,
                "schema_version": entity.schema_version,
                "created_at": entity.created_at,
                **entity.properties,
                **_evidence_properties(entity.evidence),
            }
            if node_id in nodes:
                merged = {**nodes[node_id].properties, **_json_dict(properties)}
                merged["candidate_ids"] = sorted(
                    {
                        *merged.get("candidate_ids", [nodes[node_id].properties.get("candidate_id")]),
                        entity.candidate_id,
                    }
                )
                nodes[node_id] = nodes[node_id].model_copy(update={"properties": merged})
            else:
                nodes[node_id] = PropertyNode(
                    node_id=node_id,
                    labels=labels,
                    properties=_json_dict(properties),
                )

        for relation in batch.relations:
            if relation.candidate_id in invalid_ids:
                continue
            edge_id = stable_id("edge", relation.source, relation.relation_type, relation.target, relation.candidate_id)
            edge_properties = {
                "candidate_id": relation.candidate_id,
                "confidence": relation.confidence,
                "extractor_name": relation.extractor_name,
                "technique": relation.technique,
                "schema_version": relation.schema_version,
                "created_at": relation.created_at,
                **relation.properties,
                **_evidence_properties(relation.evidence),
            }
            edges[edge_id] = PropertyEdge(
                edge_id=edge_id,
                source_id=relation.source,
                target_id=relation.target,
                type=relation.relation_type,
                properties=_json_dict(edge_properties),
            )
            if include_fact_nodes:
                fact_node = PropertyNode(
                    node_id=relation.candidate_id,
                    labels=["Fact", "RelationshipFact", relation.relation_type],
                    properties=_json_dict(edge_properties),
                )
                nodes[fact_node.node_id] = fact_node

        for fact in batch.facts:
            if fact.fact_id in invalid_ids:
                continue
            fact_properties = {
                "subject": fact.subject,
                "predicate": fact.predicate,
                "object": fact.object,
                "confidence": fact.confidence,
                "extractor_name": fact.extractor_name,
                "technique": fact.technique,
                "schema_version": fact.schema_version,
                "validation_status": fact.validation_status,
                "created_at": fact.created_at,
                **fact.properties,
                **_evidence_properties(fact.evidence),
            }
            nodes[fact.fact_id] = PropertyNode(
                node_id=fact.fact_id,
                labels=["Fact", fact.predicate],
                properties=_json_dict(fact_properties),
            )
            edge_id = stable_id("edge", fact.subject, fact.predicate, fact.object, fact.fact_id)
            edges[edge_id] = PropertyEdge(
                edge_id=edge_id,
                source_id=fact.subject,
                target_id=fact.object,
                type=fact.predicate,
                properties=_json_dict(fact_properties),
            )

        return PropertyGraph(
            nodes=list(nodes.values()),
            edges=list(edges.values()),
            metadata={
                "builder": self.name,
                "validated": report.valid,
                "finding_count": len(report.findings),
                "node_count": len(nodes),
                "edge_count": len(edges),
                "graph_model": "property_graph",
                "graphrag_ready": True,
            },
        )


__all__ = ["PropertyGraphBuilder"]
