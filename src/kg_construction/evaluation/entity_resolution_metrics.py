"""Entity resolution metrics."""

from __future__ import annotations

from itertools import combinations

from kg_construction.domain.models import ConstructionBatch, EntityCandidate
from kg_construction.evaluation.metrics import precision_recall_f1


def evaluation_record_id(
    entity: EntityCandidate,
    *,
    record_id_property: str | None = None,
) -> str:
    """Return the portable record ID used to compare a candidate with ER gold.

    By default the candidate ID is the evaluation ID, preserving the historical
    contract.  A fixture may instead declare a candidate property (for example,
    ``record_id``) whose value is stable across runs and source loading methods.
    An explicitly configured property is required on every candidate so a typo
    cannot silently produce an incomparable benchmark.
    """

    if record_id_property is None:
        return entity.candidate_id
    value = entity.properties.get(record_id_property)
    if value is None or not str(value).strip():
        raise ValueError(
            "Entity-resolution evaluation requires candidate property "
            f"{record_id_property!r}; it is missing or empty on candidate "
            f"{entity.candidate_id!r}."
        )
    return str(value).strip()


def evaluation_record_id_map(
    batch: ConstructionBatch,
    *,
    record_id_property: str | None = None,
) -> dict[str, str]:
    """Map candidate IDs to unique IDs in the gold evaluation namespace."""

    mapped: dict[str, str] = {}
    candidate_by_record_id: dict[str, str] = {}
    for entity in batch.entities:
        record_id = evaluation_record_id(
            entity,
            record_id_property=record_id_property,
        )
        previous = candidate_by_record_id.get(record_id)
        if previous is not None and previous != entity.candidate_id:
            raise ValueError(
                "Entity-resolution evaluation record IDs must be unique per "
                f"candidate; {record_id!r} maps to both {previous!r} and "
                f"{entity.candidate_id!r}."
            )
        candidate_by_record_id[record_id] = entity.candidate_id
        mapped[entity.candidate_id] = record_id
    return mapped


def predicted_merge_pairs(
    batch: ConstructionBatch,
    *,
    record_id_property: str | None = None,
) -> set[tuple[str, str]]:
    """Return evaluation-record pairs grouped by canonical entity ID."""

    evaluation_ids = evaluation_record_id_map(
        batch,
        record_id_property=record_id_property,
    )
    groups: dict[str, set[str]] = {}
    for entity in batch.entities:
        canonical_id = str(entity.properties.get("canonical_entity_id", entity.candidate_id))
        groups.setdefault(canonical_id, set()).add(evaluation_ids[entity.candidate_id])
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
        gold_record_ids: set[str] | None = None,
        record_id_property: str | None = None,
    ) -> dict[str, float]:
        evaluation_ids = evaluation_record_id_map(
            predicted,
            record_id_property=record_id_property,
        )
        predicted_record_ids = set(evaluation_ids.values())
        predicted_pairs = predicted_merge_pairs(
            predicted,
            record_id_property=record_id_property,
        )
        if gold_pairs is None:
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
        comparable_gold_ids = gold_record_ids or {
            record_id
            for pair in gold_pairs
            for record_id in pair
        }
        overlap = predicted_record_ids & comparable_gold_ids
        union = predicted_record_ids | comparable_gold_ids
        if comparable_gold_ids and not overlap:
            property_contract = record_id_property or "candidate_id"
            raise ValueError(
                "Entity-resolution prediction and gold record-ID universes are "
                "disjoint. Configure GoldEntityResolutionFixture."
                "evaluation_record_id_property to the candidate property containing "
                "the human record IDs. "
                f"Contract={property_contract!r}, predicted sample="
                f"{sorted(predicted_record_ids)[:3]!r}, gold sample="
                f"{sorted(comparable_gold_ids)[:3]!r}."
            )
        scores = (
            {"precision": 1.0, "recall": 1.0, "f1": 1.0}
            if not predicted_pairs and not gold_pairs
            else precision_recall_f1(
                true_positive=len(predicted_pairs & gold_pairs),
                predicted=len(predicted_pairs),
                expected=len(gold_pairs),
            )
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
            "er_predicted_record_support": float(len(predicted_record_ids)),
            "er_gold_record_support": float(len(comparable_gold_ids)),
            "er_record_id_overlap_support": float(len(overlap)),
            "er_record_id_overlap_rate": (
                len(overlap) / len(comparable_gold_ids)
                if comparable_gold_ids
                else 0.0
            ),
            "er_record_id_precision": (
                len(overlap) / len(predicted_record_ids)
                if predicted_record_ids
                else 0.0
            ),
            "er_record_id_symmetric_coverage": (
                len(overlap) / len(union)
                if union
                else 0.0
            ),
            "er_record_id_exact_match": float(
                predicted_record_ids == comparable_gold_ids
            ),
        }


__all__ = [
    "EntityResolutionMetrics",
    "evaluation_record_id",
    "evaluation_record_id_map",
    "predicted_merge_pairs",
]
