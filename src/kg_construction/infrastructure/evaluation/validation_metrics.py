"""Graph validation and hygiene metrics."""

from __future__ import annotations

from typing import Any

from kg_construction.domain.graph_objects import PropertyGraph, RDFGraph
from kg_construction.domain.models import ConstructionBatch
from kg_construction.infrastructure.ontology.validation import ValidationReport


def validation_failure_metrics(
    batch: ConstructionBatch,
    validation_report: ValidationReport,
) -> dict[str, float]:
    """Return distinct-object failure rate plus unbounded finding density."""

    total_objects = max(1, len(batch.entities) + len(batch.relations) + len(batch.facts))
    failed_object_ids = {
        (finding.object_type, finding.object_id)
        for finding in validation_report.findings
        if finding.object_id
    }
    return {
        "validation_failure_rate": min(1.0, len(failed_object_ids) / total_objects),
        "validation_finding_density": len(validation_report.findings) / total_objects,
    }


class ValidationMetrics:
    """Compute validity scores from validation reports and graph objects."""

    name = "validation_metrics"

    def evaluate(
        self,
        batch: ConstructionBatch,
        validation_report: ValidationReport,
        *,
        property_graph: PropertyGraph | None = None,
        rdf_graph: RDFGraph | None = None,
    ) -> dict[str, float]:
        total_objects = max(1, len(batch.entities) + len(batch.relations) + len(batch.facts))
        schema_violations = sum(
            1
            for finding in validation_report.findings
            if finding.rule in {"allowed_label", "allowed_relationship", "required_properties"}
        )
        evidence_violations = sum(
            1
            for finding in validation_report.findings
            if finding.rule == "evidence_required"
        )
        duplicate_edge_rate = self._duplicate_edge_rate(property_graph)
        dangling_edge_rate = self._dangling_edge_rate(property_graph)
        orphan_node_rate = self._orphan_node_rate(property_graph)
        rdf_output_validity = 1.0 if rdf_graph is None else self._rdf_validity(rdf_graph)
        unsupported_extraction_rate = self._unsupported_extraction_rate(batch)
        result = {
            "schema_validity_score": max(0.0, 1.0 - schema_violations / total_objects),
            "rdf_output_validity_score": rdf_output_validity,
            "shacl_validity_score": rdf_output_validity,
            "evidence_coverage": max(0.0, 1.0 - evidence_violations / total_objects),
            "provenance_coverage": self._provenance_coverage(batch),
            "duplicate_edge_rate": duplicate_edge_rate,
            "dangling_edge_rate": dangling_edge_rate,
            "referential_integrity_score": 1.0 - dangling_edge_rate,
            "orphan_node_rate": orphan_node_rate,
            "schema_violation_rate": schema_violations / total_objects,
            "unsupported_extraction_rate": unsupported_extraction_rate,
            "hallucination_risk_proxy": unsupported_extraction_rate,
            "hallucination_rate": unsupported_extraction_rate,
        }
        result.update(validation_failure_metrics(batch, validation_report))
        return result

    @staticmethod
    def _provenance_coverage(batch: ConstructionBatch) -> float:
        objects: list[Any] = [*batch.entities, *batch.relations, *batch.facts]
        if not objects:
            return 1.0
        return sum(1 for item in objects if item.evidence and item.extractor_name and item.technique) / len(objects)

    @staticmethod
    def _unsupported_extraction_rate(batch: ConstructionBatch) -> float:
        objects: list[Any] = [*batch.entities, *batch.relations, *batch.facts]
        if not objects:
            return 0.0
        unsupported = sum(
            1
            for item in objects
            if not any(evidence.source_id and evidence.location for evidence in item.evidence)
        )
        return unsupported / len(objects)

    @staticmethod
    def _duplicate_edge_rate(property_graph: PropertyGraph | None) -> float:
        if property_graph is None or not property_graph.edges:
            return 0.0
        # Canonical builders expose the qualifier-aware key used to consolidate
        # equivalent assertions. Other edges remain distinct by ID: a shared
        # source/type/target tuple alone cannot prove that qualified relations
        # or repeated events are duplicates.
        keys = [
            ("semantic_relation", str(edge.properties["semantic_relation_key"]))
            if edge.properties.get("semantic_relation_key")
            else ("edge", edge.edge_id)
            for edge in property_graph.edges
        ]
        return 1.0 - len(set(keys)) / len(keys)

    @staticmethod
    def _orphan_node_rate(property_graph: PropertyGraph | None) -> float:
        if property_graph is None or not property_graph.nodes:
            return 0.0
        connected = {edge.source_id for edge in property_graph.edges} | {edge.target_id for edge in property_graph.edges}
        domain_nodes = {
            node.node_id
            for node in property_graph.nodes
            if "Entity" in node.labels
        }
        if not domain_nodes:
            return 0.0
        return len(domain_nodes - connected) / len(domain_nodes)

    @staticmethod
    def _dangling_edge_rate(property_graph: PropertyGraph | None) -> float:
        if property_graph is None or not property_graph.edges:
            return 0.0
        node_ids = {node.node_id for node in property_graph.nodes}
        dangling = sum(
            1
            for edge in property_graph.edges
            if edge.source_id not in node_ids or edge.target_id not in node_ids
        )
        return dangling / len(property_graph.edges)

    @staticmethod
    def _rdf_validity(rdf_graph: RDFGraph) -> float:
        triples = len(rdf_graph.triples) + len(rdf_graph.quads)
        if triples == 0:
            return 1.0
        try:
            rdf_graph.to_rdflib_graph()
            rdf_graph.to_rdflib_dataset().serialize(format="nquads")
        except Exception:
            return 0.0
        invalid = 0
        for triple in rdf_graph.triples:
            if not triple.subject or not triple.predicate or triple.object is None:
                invalid += 1
        for quad in rdf_graph.quads:
            if not quad.subject or not quad.predicate or quad.object is None or not quad.graph:
                invalid += 1
        return max(0.0, 1.0 - invalid / triples)


__all__ = ["ValidationMetrics", "validation_failure_metrics"]
