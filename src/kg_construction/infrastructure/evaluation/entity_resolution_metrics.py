"""Entity resolution metrics."""

from __future__ import annotations

from itertools import combinations

from kg_construction.domain.models import ConstructionBatch
from kg_construction.infrastructure.evaluation.metrics import precision_recall_f1


def predicted_merge_pairs(batch: ConstructionBatch) -> set[tuple[str, str]]:
    """Return candidate pairs grouped by canonical entity id."""

    groups: dict[str, list[str]] = {}
    for entity in batch.entities:
        canonical_id = str(entity.properties.get("canonical_entity_id", entity.candidate_id))
        groups.setdefault(canonical_id, []).append(entity.candidate_id)
    pairs: set[tuple[str, str]] = set()
    for members in groups.values():
        for left, right in combinations(sorted(members), 2):
            pairs.add((left, right))
    return pairs


class EntityResolutionMetrics:
    """Evaluate entity resolution against optional gold merge pairs."""

    name = "entity_resolution_metrics"

    def evaluate(
        self,
        predicted: ConstructionBatch,
        *,
        gold_pairs: set[tuple[str, str]] | None = None,
    ) -> dict[str, float]:
        predicted_pairs = predicted_merge_pairs(predicted)
        if not gold_pairs:
            return {
                "pairwise_er_precision": 1.0 if predicted_pairs else 0.0,
                "pairwise_er_recall": 0.0,
                "pairwise_er_f1": 0.0,
                "cluster_precision": 1.0 if predicted_pairs else 0.0,
                "cluster_recall": 0.0,
                "cluster_f1": 0.0,
                "false_merge_rate": 0.0,
                "false_split_rate": 0.0,
            }
        scores = precision_recall_f1(
            true_positive=len(predicted_pairs & gold_pairs),
            predicted=len(predicted_pairs),
            expected=len(gold_pairs),
        )
        false_merge_rate = (len(predicted_pairs - gold_pairs) / len(predicted_pairs)) if predicted_pairs else 0.0
        false_split_rate = (len(gold_pairs - predicted_pairs) / len(gold_pairs)) if gold_pairs else 0.0
        return {
            "pairwise_er_precision": scores["precision"],
            "pairwise_er_recall": scores["recall"],
            "pairwise_er_f1": scores["f1"],
            "cluster_precision": scores["precision"],
            "cluster_recall": scores["recall"],
            "cluster_f1": scores["f1"],
            "false_merge_rate": false_merge_rate,
            "false_split_rate": false_split_rate,
        }


__all__ = ["EntityResolutionMetrics", "predicted_merge_pairs"]
