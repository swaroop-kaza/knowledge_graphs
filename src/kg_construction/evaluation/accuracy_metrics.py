"""Extraction accuracy metrics."""

from __future__ import annotations

from collections import Counter
from collections.abc import Hashable, Mapping
from typing import TypeAlias

from kg_construction.domain.models import (
    ConstructionBatch,
    EntityCandidate,
    EvidenceRef,
    RelationCandidate,
    canonical_json,
)
from kg_construction.evaluation.metrics import precision_recall_f1

EntityKey: TypeAlias = tuple[str, str, str, str]
SpanKey: TypeAlias = tuple[str, str, int, int, str]
RelationKey: TypeAlias = tuple[Hashable, str, Hashable, str]

_SEMANTIC_RELATION_QUALIFIERS = frozenset(
    {
        "attribution",
        "claim_kind",
        "confidence_status",
        "end_time",
        "event_predicate",
        "event_roles",
        "modality",
        "negated",
        "occurred_at",
        "quantity",
        "role",
        "start_time",
        "status",
        "temporal",
        "valid_from",
        "valid_to",
    }
)


def entity_key(entity_type: str, name: str) -> tuple[str, str]:
    """Normalize an entity label and surface for compatibility callers."""

    return entity_type.casefold(), name.casefold().strip()


def relation_key(
    source: str,
    relation_type: str,
    target: str,
) -> tuple[str, str, str]:
    """Normalize an ordered relation for compatibility callers."""

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
            predicted_entities = _entity_occurrences(predicted)
            predicted_spans = _span_occurrences(predicted)
            predicted_relations = _relation_occurrences(predicted)
            return {
                "entity_precision": 0.0,
                "entity_recall": 0.0,
                "entity_f1": 0.0,
                "entity_gold_support": 0.0,
                "entity_prediction_support": float(sum(predicted_entities.values())),
                "entity_gold_coverage": 0.0,
                "span_precision": 0.0,
                "span_recall": 0.0,
                "relation_precision": 0.0,
                "relation_recall": 0.0,
                "relation_f1": 0.0,
                "relation_gold_support": 0.0,
                "relation_prediction_support": float(
                    sum(predicted_relations.values())
                ),
                "relation_gold_coverage": 0.0,
                "triple_exact_match": 0.0,
                "relaxed_triple_match": 0.0,
                "span_f1": 0.0,
                "span_gold_support": 0.0,
                "span_prediction_support": float(sum(predicted_spans.values())),
                "span_gold_coverage": 0.0,
            }

        predicted_entities = _entity_occurrences(predicted)
        expected_entities = _entity_occurrences(expected)
        predicted_spans = _span_occurrences(predicted)
        expected_spans = _span_occurrences(expected)
        predicted_relations = _relation_occurrences(predicted)
        expected_relations = _relation_occurrences(expected)

        entity_true_positive = sum(
            (predicted_entities & expected_entities).values()
        )
        span_true_positive = sum((predicted_spans & expected_spans).values())
        relation_true_positive = sum(
            (predicted_relations & expected_relations).values()
        )
        predicted_entity_count = sum(predicted_entities.values())
        expected_entity_count = sum(expected_entities.values())
        predicted_span_count = sum(predicted_spans.values())
        expected_span_count = sum(expected_spans.values())
        predicted_relation_count = sum(predicted_relations.values())
        expected_relation_count = sum(expected_relations.values())

        entity_scores = precision_recall_f1(
            true_positive=entity_true_positive,
            predicted=predicted_entity_count,
            expected=expected_entity_count,
        )
        span_scores = precision_recall_f1(
            true_positive=span_true_positive,
            predicted=predicted_span_count,
            expected=expected_span_count,
        )
        relation_scores = precision_recall_f1(
            true_positive=relation_true_positive,
            predicted=predicted_relation_count,
            expected=expected_relation_count,
        )
        return {
            "entity_precision": entity_scores["precision"],
            "entity_recall": entity_scores["recall"],
            "entity_f1": entity_scores["f1"],
            "entity_gold_support": float(expected_entity_count),
            "entity_prediction_support": float(predicted_entity_count),
            "entity_gold_coverage": entity_scores["recall"],
            "span_precision": span_scores["precision"],
            "span_recall": span_scores["recall"],
            "span_f1": span_scores["f1"],
            "span_gold_support": float(expected_span_count),
            "span_prediction_support": float(predicted_span_count),
            "span_gold_coverage": span_scores["recall"],
            "relation_precision": relation_scores["precision"],
            "relation_recall": relation_scores["recall"],
            "relation_f1": relation_scores["f1"],
            "relation_gold_support": float(expected_relation_count),
            "relation_prediction_support": float(predicted_relation_count),
            "relation_gold_coverage": relation_scores["recall"],
            "triple_exact_match": relation_scores["f1"],
            "relaxed_triple_match": relation_scores["f1"],
        }


def _entity_occurrences(batch: ConstructionBatch) -> Counter[EntityKey]:
    """Count entity observations without collapsing repeated occurrences.

    Entity identity uses snapshot/version/type/surface while ``Counter`` retains
    repeated occurrences. Exact offsets are deliberately scored by the separate
    span metric, so a boundary error cannot be hidden behind entity F1 and the
    two scores cannot alias one another.
    """

    occurrences: Counter[EntityKey] = Counter()
    for entity in _scored_entities(batch):
        entity_type, surface = _entity_observation(entity)
        span_evidence = _span_evidence(entity.evidence)
        if span_evidence:
            for evidence, _, _ in span_evidence:
                occurrences[
                    (
                        evidence.source_id,
                        _document_version(entity, evidence),
                        entity_type.casefold(),
                        surface.casefold().strip(),
                    )
                ] += 1
            continue
        token = _non_span_occurrence(entity)
        occurrences[
            (
                token,
                "",
                entity_type.casefold(),
                surface.casefold().strip(),
            )
        ] += 1
    return occurrences


def _span_occurrences(batch: ConstructionBatch) -> Counter[SpanKey]:
    """Count exact typed spans by document/version/start/end/type."""

    occurrences: Counter[SpanKey] = Counter()
    for entity in _scored_entities(batch):
        entity_type, _ = _entity_observation(entity)
        for evidence, start, end in _span_evidence(entity.evidence):
            occurrences[
                (
                    evidence.source_id,
                    _document_version(entity, evidence),
                    start,
                    end,
                    entity_type.casefold(),
                )
            ] += 1
    return occurrences


def _relation_occurrences(batch: ConstructionBatch) -> Counter[RelationKey]:
    """Count qualifier-aware directed semantic relation occurrences."""

    materialized_unstructured = _has_materialized_mentions(batch)
    endpoints = {
        entity.candidate_id: _semantic_endpoint(entity)
        for entity in batch.entities
    }
    for entity in batch.entities:
        canonical_id = entity.properties.get("canonical_entity_id")
        if canonical_id not in (None, ""):
            endpoints[str(canonical_id)] = _semantic_endpoint(entity)

    occurrences: Counter[RelationKey] = Counter()
    for relation in batch.relations:
        if materialized_unstructured and not _is_direct_semantic_relation(relation):
            continue
        occurrences[
            (
                endpoints.get(relation.source, ("external", relation.source.casefold())),
                relation.relation_type.casefold(),
                endpoints.get(relation.target, ("external", relation.target.casefold())),
                _relation_qualifiers(relation),
            )
        ] += 1
    return occurrences


def _semantic_endpoint(entity: EntityCandidate) -> Hashable:
    entity_type, surface = _entity_observation(entity)
    return (
        "entity",
        entity_type.casefold(),
        surface.casefold().strip(),
    )


def _scored_entities(batch: ConstructionBatch) -> list[EntityCandidate]:
    """Return semantic mention observations for materialized text pipelines.

    The unstructured pipeline deliberately emits Document, Mention,
    observed/canonical identity, Claim, and Event resources.  Only Mention is an
    extraction observation.  Scoring every materialization layer would count
    the same detected entity repeatedly and treat technical resources as NER
    false positives.  Legacy extractors do not emit Mention resources, so their
    original entity candidates remain the scoring surface.
    """

    mentions = [
        entity for entity in batch.entities if entity.entity_type.casefold() == "mention"
    ]
    return mentions or batch.entities


def _entity_observation(entity: EntityCandidate) -> tuple[str, str]:
    if entity.entity_type.casefold() == "mention":
        entity_type = entity.properties.get("detected_entity_type")
        surface = entity.properties.get("surface")
        return (
            str(entity_type or "Entity"),
            str(surface or entity.name),
        )
    if entity.entity_type.casefold() == "canonicalentity":
        semantic_type = entity.properties.get("semantic_entity_type")
        return str(semantic_type or "Entity"), entity.name
    return entity.entity_type, entity.name


def _has_materialized_mentions(batch: ConstructionBatch) -> bool:
    return any(
        entity.entity_type.casefold() == "mention" for entity in batch.entities
    )


def _is_direct_semantic_relation(relation: RelationCandidate) -> bool:
    """Distinguish asserted domain edges from provenance/materialization edges."""

    if relation.properties.get("semantic_relation") is True:
        return True
    claim_kind = str(relation.properties.get("claim_kind", "")).casefold()
    predicate = relation.properties.get("predicate")
    if claim_kind not in {"relation", "intent"} or not isinstance(predicate, str):
        return False
    normalized_predicate = "".join(
        character.casefold() for character in predicate if character.isalnum()
    )
    normalized_type = "".join(
        character.casefold() for character in relation.relation_type if character.isalnum()
    )
    return bool(normalized_predicate) and normalized_type == normalized_predicate


def _relation_qualifiers(relation: RelationCandidate) -> str:
    """Return all semantic qualifiers in canonical form.

    Direction is represented by the ordered source/target tuple. Confidence,
    extractor provenance, timestamps, and evidence IDs are intentionally not
    qualifiers; relation properties such as negation, modality, time, quantity,
    attribution, and event role are.
    """

    qualifiers = {
        key: value
        for key, value in relation.properties.items()
        if key in _SEMANTIC_RELATION_QUALIFIERS
        or key.endswith(("_at", "_time", "_date"))
        or key.startswith(("valid_", "temporal_", "quantity_"))
    }
    event_roles = qualifiers.get("event_roles")
    if isinstance(event_roles, list):
        qualifiers["event_roles"] = [
            {
                key: value
                for key, value in role.items()
                if key in {"role", "value"}
            }
            if isinstance(role, Mapping)
            else role
            for role in event_roles
        ]
    return canonical_json(qualifiers)


def _span_evidence(
    evidence_items: list[EvidenceRef],
) -> list[tuple[EvidenceRef, int, int]]:
    spans: list[tuple[EvidenceRef, int, int]] = []
    for evidence in evidence_items:
        location = evidence.location
        start = location.get("start")
        end = location.get("end")
        if (
            location.get("kind") == "span"
            and isinstance(start, int)
            and isinstance(end, int)
            and 0 <= start < end
        ):
            spans.append((evidence, start, end))
    return spans


def _document_version(
    entity: EntityCandidate,
    evidence: EvidenceRef,
) -> str:
    value = (
        entity.properties.get("document_version")
        or evidence.metadata.get("document_version")
        or evidence.source_id
    )
    return str(value)


def _non_span_occurrence(entity: EntityCandidate) -> str:
    if entity.evidence:
        return "|".join(sorted(item.evidence_id for item in entity.evidence))
    return entity.candidate_id


__all__ = ["AccuracyMetrics", "entity_key", "relation_key"]
