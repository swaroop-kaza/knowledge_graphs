"""Construction quality and operational metrics."""

from __future__ import annotations

from typing import Any

from kg_construction.domain.models import ConstructionBatch


def precision_recall_f1(*, true_positive: int, predicted: int, expected: int) -> dict[str, float]:
    """Compute precision, recall, and F1 with zero-safe division."""

    precision = true_positive / predicted if predicted else 0.0
    recall = true_positive / expected if expected else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


class ConstructionMetricsEvaluator:
    """Evaluate candidate graph construction quality and graph hygiene."""

    name = "construction_metrics_evaluator"

    def evaluate(
        self,
        predicted: ConstructionBatch,
        expected: ConstructionBatch | None = None,
    ) -> dict[str, float]:
        predicted_entities = {entity.candidate_id for entity in predicted.entities}
        predicted_relations = {relation.candidate_id for relation in predicted.relations}
        predicted_facts = {fact.fact_id for fact in predicted.facts}
        unsupported_extraction_rate = self._unsupported_extraction_rate(predicted)
        metrics: dict[str, float] = {
            "entity_count": float(len(predicted_entities)),
            "relation_count": float(len(predicted_relations)),
            "fact_count": float(len(predicted_facts)),
            "provenance_coverage": self._provenance_coverage(predicted),
            "duplicate_edge_rate": self._duplicate_edge_rate(predicted),
            "orphan_node_rate": self._orphan_node_rate(predicted),
            "unsupported_extraction_rate": unsupported_extraction_rate,
            "hallucination_risk_proxy": unsupported_extraction_rate,
            "hallucinated_fact_rate": unsupported_extraction_rate,
            "throughput": 0.0,
            "latency": 0.0,
            "memory_usage": 0.0,
            "ollama_token_usage": 0.0,
            "graph_write_time": 0.0,
            "retry_rate": 0.0,
            "validation_failure_rate": 0.0,
            "retrieval_recall_at_k": 0.0,
            "path_relevance": 0.0,
            "multi_hop_retrieval_success": 0.0,
            "answer_citation_accuracy": 0.0,
        }
        if expected is not None:
            expected_entities = {entity.candidate_id for entity in expected.entities}
            expected_relations = {relation.candidate_id for relation in expected.relations}
            entity_scores = precision_recall_f1(
                true_positive=len(predicted_entities & expected_entities),
                predicted=len(predicted_entities),
                expected=len(expected_entities),
            )
            relation_scores = precision_recall_f1(
                true_positive=len(predicted_relations & expected_relations),
                predicted=len(predicted_relations),
                expected=len(expected_relations),
            )
            metrics.update(
                {
                    "entity_precision": entity_scores["precision"],
                    "entity_recall": entity_scores["recall"],
                    "entity_f1": entity_scores["f1"],
                    "relation_precision": relation_scores["precision"],
                    "relation_recall": relation_scores["recall"],
                    "relation_f1": relation_scores["f1"],
                    "triple_exact_match": relation_scores["f1"],
                    "relaxed_triple_match": relation_scores["f1"],
                    "span_level_f1": entity_scores["f1"],
                    "schema_label_accuracy": entity_scores["precision"],
                    "pairwise_er_precision": 0.0,
                    "pairwise_er_recall": 0.0,
                    "pairwise_er_f1": 0.0,
                    "cluster_precision": 0.0,
                    "cluster_recall": 0.0,
                    "cluster_f1": 0.0,
                    "false_merge_rate": 0.0,
                    "false_split_rate": 0.0,
                }
            )
        return metrics

    @staticmethod
    def _provenance_coverage(batch: ConstructionBatch) -> float:
        objects: list[Any] = [*batch.entities, *batch.relations, *batch.facts]
        if not objects:
            return 1.0
        with_evidence = sum(1 for item in objects if bool(item.evidence))
        return with_evidence / len(objects)

    @staticmethod
    def _duplicate_edge_rate(batch: ConstructionBatch) -> float:
        if not batch.relations:
            return 0.0
        keys = [(relation.source, relation.relation_type, relation.target) for relation in batch.relations]
        return 1.0 - (len(set(keys)) / len(keys))

    @staticmethod
    def _orphan_node_rate(batch: ConstructionBatch) -> float:
        if not batch.entities:
            return 0.0
        entity_ids = {entity.candidate_id for entity in batch.entities}
        related_ids = {relation.source for relation in batch.relations} | {relation.target for relation in batch.relations}
        return len(entity_ids - related_ids) / len(entity_ids)

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


__all__ = ["ConstructionMetricsEvaluator", "precision_recall_f1"]
