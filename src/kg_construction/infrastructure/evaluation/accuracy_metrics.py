"""Extraction accuracy metrics."""

from __future__ import annotations

from kg_construction.domain.models import ConstructionBatch
from kg_construction.infrastructure.evaluation.metrics import precision_recall_f1


def entity_key(entity_type: str, name: str) -> tuple[str, str]:
    """Normalize an entity for comparison."""

    return entity_type.casefold(), name.casefold().strip()


def relation_key(source: str, relation_type: str, target: str) -> tuple[str, str, str]:
    """Normalize a relation for comparison."""

    return source.casefold(), relation_type.casefold(), target.casefold()


class AccuracyMetrics:
    """Compute extraction metrics when a gold construction batch is available."""

    name = "accuracy_metrics"

    def evaluate(
        self,
        predicted: ConstructionBatch,
        expected: ConstructionBatch | None = None,
    ) -> dict[str, float]:
        if expected is None:
            return {
                "entity_precision": 0.0,
                "entity_recall": 0.0,
                "entity_f1": 0.0,
                "relation_precision": 0.0,
                "relation_recall": 0.0,
                "relation_f1": 0.0,
                "triple_exact_match": 0.0,
                "relaxed_triple_match": 0.0,
                "span_f1": 0.0,
            }

        predicted_entities = {
            entity_key(entity.entity_type, entity.name)
            for entity in predicted.entities
        }
        expected_entities = {
            entity_key(entity.entity_type, entity.name)
            for entity in expected.entities
        }
        predicted_relations = {
            relation_key(relation.source, relation.relation_type, relation.target)
            for relation in predicted.relations
        }
        expected_relations = {
            relation_key(relation.source, relation.relation_type, relation.target)
            for relation in expected.relations
        }
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
        return {
            "entity_precision": entity_scores["precision"],
            "entity_recall": entity_scores["recall"],
            "entity_f1": entity_scores["f1"],
            "relation_precision": relation_scores["precision"],
            "relation_recall": relation_scores["recall"],
            "relation_f1": relation_scores["f1"],
            "triple_exact_match": relation_scores["f1"],
            "relaxed_triple_match": relation_scores["f1"],
            "span_f1": entity_scores["f1"],
        }


__all__ = ["AccuracyMetrics", "entity_key", "relation_key"]
