"""Evidence graph builder."""

from __future__ import annotations

from kg_construction.domain.graph_objects import PropertyEdge, PropertyGraph, PropertyNode
from kg_construction.domain.models import ConstructionBatch, EvidenceRef, stable_id


class EvidenceGraphBuilder:
    """Build the audit graph connecting sources, evidence, candidates, facts, and runs."""

    name = "evidence_graph_builder"

    def build(self, batch: ConstructionBatch) -> PropertyGraph:
        nodes: dict[str, PropertyNode] = {}
        edges: dict[str, PropertyEdge] = {}

        for source in batch.source_records:
            nodes[source.source_id] = PropertyNode(
                node_id=source.source_id,
                labels=["Source", source.source_type],
                properties={
                    "source_type": source.source_type,
                    "source_name": source.source_name,
                    "created_at": source.created_at.isoformat(),
                    **source.metadata,
                },
            )

        for candidate_id, candidate_type, evidence_items in self._candidate_evidence(batch):
            nodes[candidate_id] = PropertyNode(
                node_id=candidate_id,
                labels=["Candidate", candidate_type],
                properties={"candidate_type": candidate_type},
            )
            for evidence in evidence_items:
                self._add_evidence(nodes, edges, evidence, candidate_id)

        return PropertyGraph(
            nodes=list(nodes.values()),
            edges=list(edges.values()),
            metadata={
                "builder": self.name,
                "graph_model": "evidence_graph",
                "node_count": len(nodes),
                "edge_count": len(edges),
            },
        )

    @staticmethod
    def _candidate_evidence(batch: ConstructionBatch) -> list[tuple[str, str, list[EvidenceRef]]]:
        items: list[tuple[str, str, list[EvidenceRef]]] = []
        items.extend((entity.candidate_id, "EntityCandidate", entity.evidence) for entity in batch.entities)
        items.extend((relation.candidate_id, "RelationCandidate", relation.evidence) for relation in batch.relations)
        items.extend((fact.fact_id, "GraphFact", fact.evidence) for fact in batch.facts)
        return items

    @staticmethod
    def _add_evidence(
        nodes: dict[str, PropertyNode],
        edges: dict[str, PropertyEdge],
        evidence: EvidenceRef,
        candidate_id: str,
    ) -> None:
        nodes[evidence.evidence_id] = PropertyNode(
            node_id=evidence.evidence_id,
            labels=["Evidence", evidence.source_type],
            properties={
                "source_id": evidence.source_id,
                "source_name": evidence.source_name,
                "location": evidence.location,
                "text": evidence.text,
                "created_at": evidence.created_at.isoformat(),
                **evidence.metadata,
            },
        )
        source_edge_id = stable_id("edge", evidence.source_id, "HAS_EVIDENCE", evidence.evidence_id)
        edges[source_edge_id] = PropertyEdge(
            edge_id=source_edge_id,
            source_id=evidence.source_id,
            target_id=evidence.evidence_id,
            type="HAS_EVIDENCE",
            properties={"source_type": evidence.source_type},
        )
        evidence_edge_id = stable_id("edge", candidate_id, "EVIDENCED_BY", evidence.evidence_id)
        edges[evidence_edge_id] = PropertyEdge(
            edge_id=evidence_edge_id,
            source_id=candidate_id,
            target_id=evidence.evidence_id,
            type="EVIDENCED_BY",
            properties={"source_id": evidence.source_id},
        )


__all__ = ["EvidenceGraphBuilder"]
