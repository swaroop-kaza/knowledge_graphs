"""Gold-standard benchmark fixtures and evaluators."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from kg_construction.domain.models import ConstructionBatch

JsonDict = dict[str, Any]


class GoldEntitySpan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str
    label: str
    start: int
    end: int


class GoldTriple(BaseModel):
    model_config = ConfigDict(extra="forbid")

    subject: str
    predicate: str
    object: str


class GoldRelation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: str
    relation_type: str
    target: str


class GoldEntityResolutionRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    record_id: str
    entity_type: str
    name: str
    cluster_id: str


class GoldEntityResolutionFixture(BaseModel):
    model_config = ConfigDict(extra="forbid")

    fixture_id: str
    records: list[GoldEntityResolutionRecord]
    metadata: JsonDict = Field(default_factory=dict)

    @classmethod
    def from_json_file(cls, path: str | Path) -> GoldEntityResolutionFixture:
        return cls.model_validate(json.loads(Path(path).read_text(encoding="utf-8")))

    def gold_pairs(self) -> set[tuple[str, str]]:
        pairs: set[tuple[str, str]] = set()
        clusters: dict[str, list[str]] = {}
        for record in self.records:
            clusters.setdefault(record.cluster_id, []).append(record.record_id)
        for members in clusters.values():
            for left_index, left in enumerate(sorted(members)):
                for right in sorted(members)[left_index + 1 :]:
                    pairs.add((left, right))
        return pairs


class GoldRetrievalEdge(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: str
    relation_type: str
    target: str


class GoldRetrievalContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    context_id: str
    text: str
    nodes: list[str] = Field(default_factory=list)
    edges: list[GoldRetrievalEdge] = Field(default_factory=list)
    citation_ids: list[str] = Field(default_factory=list)


class GoldRetrievalCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query_id: str
    query: str
    relevant_context_ids: list[str]
    expected_answer_contains: list[str] = Field(default_factory=list)
    expected_path: list[str] = Field(default_factory=list)
    expected_citation_ids: list[str] = Field(default_factory=list)


class GoldRetrievalFixture(BaseModel):
    model_config = ConfigDict(extra="forbid")

    fixture_id: str
    contexts: list[GoldRetrievalContext]
    cases: list[GoldRetrievalCase]
    metadata: JsonDict = Field(default_factory=dict)

    @classmethod
    def from_json_file(cls, path: str | Path) -> GoldRetrievalFixture:
        return cls.model_validate(json.loads(Path(path).read_text(encoding="utf-8")))


class GoldBenchmarkFixture(BaseModel):
    model_config = ConfigDict(extra="forbid")

    fixture_id: str
    text: str | None = None
    entity_spans: list[GoldEntitySpan] = Field(default_factory=list)
    triples: list[GoldTriple] = Field(default_factory=list)
    relations: list[GoldRelation] = Field(default_factory=list)
    metadata: JsonDict = Field(default_factory=dict)

    @classmethod
    def from_json_file(cls, path: str | Path) -> GoldBenchmarkFixture:
        return cls.model_validate(json.loads(Path(path).read_text(encoding="utf-8")))


class GoldBenchmarkScore(BaseModel):
    model_config = ConfigDict(extra="forbid")

    precision: float
    recall: float
    f1: float
    true_positive: int
    false_positive: int
    false_negative: int


class GoldBenchmarkEvaluator:
    """Evaluate construction output against gold spans, triples, and relations."""

    name = "gold_benchmark_evaluator"

    def evaluate_entity_spans(
        self,
        batch: ConstructionBatch,
        gold: GoldBenchmarkFixture,
    ) -> GoldBenchmarkScore:
        predicted = {
            (entity.name, entity.entity_type, start, end)
            for entity in batch.entities
            if (start := _int_property(entity.properties.get("start"))) is not None
            and (end := _int_property(entity.properties.get("end"))) is not None
        }
        expected = {(span.text, span.label, span.start, span.end) for span in gold.entity_spans}
        return _score_sets(predicted, expected)

    def evaluate_entities(self, batch: ConstructionBatch, gold: GoldBenchmarkFixture) -> GoldBenchmarkScore:
        predicted = {(entity.name, entity.entity_type) for entity in batch.entities}
        expected = {(span.text, span.label) for span in gold.entity_spans}
        return _score_sets(predicted, expected)

    def evaluate_triples(self, batch: ConstructionBatch, gold: GoldBenchmarkFixture) -> GoldBenchmarkScore:
        predicted = {(fact.subject, fact.predicate, fact.object) for fact in batch.facts}
        expected = {(triple.subject, triple.predicate, triple.object) for triple in gold.triples}
        return _score_sets(predicted, expected)

    def evaluate_relations(self, batch: ConstructionBatch, gold: GoldBenchmarkFixture) -> GoldBenchmarkScore:
        predicted = {(relation.source, relation.relation_type, relation.target) for relation in batch.relations}
        expected = {(relation.source, relation.relation_type, relation.target) for relation in gold.relations}
        return _score_sets(predicted, expected)

    def evaluate_relation_mentions(
        self,
        batch: ConstructionBatch,
        gold: GoldBenchmarkFixture,
    ) -> GoldBenchmarkScore:
        name_by_id = {
            str(key): str(value)
            for key, value in dict(gold.metadata.get("entity_names", {})).items()
        }
        predicted = {
            (
                str(relation.properties.get("source_name", relation.source)),
                relation.relation_type,
                str(relation.properties.get("target_name", relation.target)),
            )
            for relation in batch.relations
        }
        expected = {
            (
                name_by_id.get(relation.source, relation.source),
                relation.relation_type,
                name_by_id.get(relation.target, relation.target),
            )
            for relation in gold.relations
        }
        return _score_sets(predicted, expected)


def _score_sets(predicted: set[tuple[Any, ...]], expected: set[tuple[Any, ...]]) -> GoldBenchmarkScore:
    true_positive = len(predicted & expected)
    false_positive = len(predicted - expected)
    false_negative = len(expected - predicted)
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
    return GoldBenchmarkScore(
        precision=round(precision, 6),
        recall=round(recall, 6),
        f1=round(f1, 6),
        true_positive=true_positive,
        false_positive=false_positive,
        false_negative=false_negative,
    )


def _int_property(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


__all__ = [
    "GoldBenchmarkEvaluator",
    "GoldBenchmarkFixture",
    "GoldBenchmarkScore",
    "GoldEntityResolutionFixture",
    "GoldEntityResolutionRecord",
    "GoldEntitySpan",
    "GoldRelation",
    "GoldRetrievalCase",
    "GoldRetrievalContext",
    "GoldRetrievalEdge",
    "GoldRetrievalFixture",
    "GoldTriple",
]
