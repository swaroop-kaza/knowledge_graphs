from __future__ import annotations

from kg_construction.domain.models import (
    ConstructionBatch,
    EntityCandidate,
    EvidenceRef,
    RelationCandidate,
    SourceRecord,
)
from kg_construction.evaluation import AccuracyMetrics
from kg_construction.unstructured.pipeline import UnstructuredPipeline


def _entity(
    source: SourceRecord,
    *,
    name: str,
    entity_type: str,
    start: int,
    end: int,
) -> EntityCandidate:
    evidence = EvidenceRef.build(
        source,
        location={"kind": "span", "start": start, "end": end},
        text=(source.content or "")[start:end],
        metadata={"document_version": source.source_id},
    )
    return EntityCandidate.build(
        entity_type=entity_type,
        name=name,
        evidence=[evidence],
        extractor_name="test",
        technique="test",
    )


def test_entity_occurrences_and_exact_spans_are_distinct_metrics() -> None:
    source = SourceRecord.from_text(
        "Ada met Ada.",
        source_name="memory://occurrences",
    )
    first = _entity(source, name="Ada", entity_type="Person", start=0, end=3)
    second = _entity(source, name="Ada", entity_type="Person", start=8, end=11)
    shifted = _entity(source, name="Ada", entity_type="Person", start=1, end=3)

    occurrence_scores = AccuracyMetrics().evaluate(
        ConstructionBatch(entities=[first]),
        ConstructionBatch(entities=[first, second]),
    )
    boundary_scores = AccuracyMetrics().evaluate(
        ConstructionBatch(entities=[shifted]),
        ConstructionBatch(entities=[first]),
    )

    assert occurrence_scores["entity_recall"] == 0.5
    assert occurrence_scores["span_recall"] == 0.5
    assert occurrence_scores["entity_gold_support"] == 2
    assert occurrence_scores["span_gold_support"] == 2
    assert boundary_scores["entity_f1"] == 1.0
    assert boundary_scores["span_f1"] == 0.0


def test_relations_align_semantics_direction_and_declared_qualifiers() -> None:
    source = SourceRecord.from_text(
        "Ada never founded Acme.",
        source_name="memory://relations",
    )
    ada = _entity(source, name="Ada", entity_type="Person", start=0, end=3)
    acme = _entity(source, name="Acme", entity_type="Organization", start=18, end=22)
    evidence = ada.evidence

    def relation(
        *,
        source_id: str = ada.candidate_id,
        target_id: str = acme.candidate_id,
        negated: bool = True,
        internal_id: str,
    ) -> RelationCandidate:
        return RelationCandidate.build(
            source=source_id,
            relation_type="FOUNDED",
            target=target_id,
            properties={
                "negated": negated,
                "domain_claim_id": internal_id,
            },
            evidence=evidence,
            extractor_name="test",
            technique="test",
        )

    predicted = ConstructionBatch(
        entities=[ada, acme],
        relations=[relation(internal_id="runtime-predicted")],
    )
    equivalent_gold = ConstructionBatch(
        entities=[ada, acme],
        relations=[relation(internal_id="fixture-gold")],
    )
    reversed_gold = ConstructionBatch(
        entities=[ada, acme],
        relations=[
            relation(
                source_id=acme.candidate_id,
                target_id=ada.candidate_id,
                internal_id="fixture-gold",
            )
        ],
    )
    qualifier_gold = ConstructionBatch(
        entities=[ada, acme],
        relations=[relation(negated=False, internal_id="fixture-gold")],
    )

    assert AccuracyMetrics().evaluate(
        predicted,
        equivalent_gold,
    )["relation_f1"] == 1.0
    assert AccuracyMetrics().evaluate(
        predicted,
        reversed_gold,
    )["relation_f1"] == 0.0
    assert AccuracyMetrics().evaluate(
        predicted,
        qualifier_gold,
    )["relation_f1"] == 0.0


def test_empty_gold_and_predictions_never_pass_by_convention() -> None:
    scores = AccuracyMetrics().evaluate(ConstructionBatch(), ConstructionBatch())

    assert scores["entity_f1"] == 0.0
    assert scores["span_f1"] == 0.0
    assert scores["relation_f1"] == 0.0
    assert scores["entity_gold_support"] == 0.0
    assert scores["relation_prediction_support"] == 0.0


def test_unstructured_materialization_scores_mentions_and_semantic_edges_only() -> None:
    source = SourceRecord.from_text(
        "Ada Lovelace worked with Charles Babbage.",
        source_name="memory://semantic-scoring",
    )
    predicted = UnstructuredPipeline().extract(source)
    mentions = [
        entity for entity in predicted.entities if entity.entity_type == "Mention"
    ]
    direct_relation = next(
        relation
        for relation in predicted.relations
        if relation.properties.get("claim_kind") == "relation"
        and relation.relation_type == "WORKS_WITH"
    )

    gold_entities = [
        EntityCandidate.build(
            entity_type=str(mention.properties["detected_entity_type"]),
            name=str(mention.properties["surface"]),
            evidence=mention.evidence,
            extractor_name="curator",
            technique="gold",
        )
        for mention in mentions
    ]
    gold_relation = RelationCandidate.build(
        source=gold_entities[0].candidate_id,
        relation_type="WORKS_WITH",
        target=gold_entities[1].candidate_id,
        properties={
            "claim_kind": "relation",
            "negated": False,
            "modality": "asserted",
            "attribution": None,
            "temporal": None,
            "quantity": None,
            "event_roles": [],
        },
        evidence=direct_relation.evidence,
        extractor_name="curator",
        technique="gold",
    )
    expected = ConstructionBatch(
        source_records=[source],
        entities=gold_entities,
        relations=[gold_relation],
    )

    scores = AccuracyMetrics().evaluate(predicted, expected)

    assert len(predicted.entities) > 2
    assert len(predicted.relations) > 1
    assert scores["entity_prediction_support"] == 2
    assert scores["span_prediction_support"] == 2
    assert scores["relation_prediction_support"] == 1
    assert scores["entity_f1"] == 1.0
    assert scores["span_f1"] == 1.0
    assert scores["relation_f1"] == 1.0
