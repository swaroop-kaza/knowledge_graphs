"""Measure how much of a source's entities and intent the graph captured.

Turns the recall side of extraction into reported numbers, so the framework can
say how much of a source was captured rather than assuming nothing was missed:

- **entity_coverage** -- recall of gold salient mentions (fraction captured);
- **nil_rate** -- fraction of extracted entities not linked to a knowledge base;
- **claim_coverage** -- recall of gold claims/triples (the graph-level intent); and
- **intent_coverage_llm** (optional) -- an LLM-as-judge check of whether each gold
  intent is represented in the constructed graph.
"""

from __future__ import annotations

import logging
from typing import Protocol

from pydantic import BaseModel, ConfigDict

from kg_construction.domain.models import ConstructionBatch
from kg_construction.infrastructure.resolution.common import resolution_key

JsonDict = dict[str, object]

logger = logging.getLogger(__name__)


class IntentJudge(Protocol):
    """Injectable judge: is ``intent`` represented in ``context``?"""

    def judge(self, intent: str, context: str) -> bool:
        """Return True when the intent is supported by the graph context."""


class CoverageReport(BaseModel):
    """Reported extraction-coverage numbers."""

    model_config = ConfigDict(extra="forbid")

    entity_coverage: float = 0.0
    entity_precision: float = 0.0
    gold_entity_count: int = 0
    matched_entity_count: int = 0
    nil_rate: float = 1.0
    linked_entity_count: int = 0
    claim_coverage: float = 0.0
    gold_claim_count: int = 0
    matched_claim_count: int = 0
    intent_coverage_llm: float | None = None
    gold_intent_count: int = 0
    covered_intent_count: int = 0


class CoverageEvaluator:
    """Compute recall-oriented coverage metrics against gold references."""

    name = "extraction_coverage_evaluator"

    def evaluate(
        self,
        batch: ConstructionBatch,
        *,
        gold_entities: list[JsonDict],
        gold_claims: list[JsonDict] | None = None,
        gold_intents: list[str] | None = None,
        judge: IntentJudge | None = None,
        context: str | None = None,
    ) -> CoverageReport:
        entity_coverage, entity_precision, matched_entities = self._entity_coverage(batch, gold_entities)
        nil_rate, linked = self._nil_rate(batch)
        claim_coverage, matched_claims = self._claim_coverage(batch, gold_claims or [])
        intent_llm, covered_intents = self._intent_coverage_llm(batch, gold_intents, judge, context)

        logger.info(
            "Coverage: entities %.2f (%d/%d), claims %.2f, NIL rate %.2f",
            entity_coverage,
            matched_entities,
            len(gold_entities),
            claim_coverage,
            nil_rate,
        )
        return CoverageReport(
            entity_coverage=entity_coverage,
            entity_precision=entity_precision,
            gold_entity_count=len(gold_entities),
            matched_entity_count=matched_entities,
            nil_rate=nil_rate,
            linked_entity_count=linked,
            claim_coverage=claim_coverage,
            gold_claim_count=len(gold_claims or []),
            matched_claim_count=matched_claims,
            intent_coverage_llm=intent_llm,
            gold_intent_count=len(gold_intents or []),
            covered_intent_count=covered_intents,
        )

    @staticmethod
    def _entity_coverage(
        batch: ConstructionBatch, gold_entities: list[JsonDict]
    ) -> tuple[float, float, int]:
        extracted: set[str] = set()
        for entity in batch.entities:
            if entity.name:
                extracted.add(resolution_key(entity.name))
            canonical_name = entity.properties.get("canonical_name")
            if canonical_name:
                extracted.add(resolution_key(str(canonical_name)))
        extracted.discard("")
        if not gold_entities:
            return 1.0, 1.0, 0
        gold_keys = {resolution_key(str(item.get("name", ""))) for item in gold_entities}
        gold_keys.discard("")
        matched = gold_keys & extracted
        recall = len(matched) / len(gold_keys) if gold_keys else 1.0
        precision = len(extracted & gold_keys) / len(extracted) if extracted else 0.0
        return recall, precision, len(matched)

    @staticmethod
    def _nil_rate(batch: ConstructionBatch) -> tuple[float, int]:
        if not batch.entities:
            return 1.0, 0
        linked = sum(1 for entity in batch.entities if entity.properties.get("link_status") == "linked")
        return 1.0 - (linked / len(batch.entities)), linked

    def _claim_coverage(
        self, batch: ConstructionBatch, gold_claims: list[JsonDict]
    ) -> tuple[float, int]:
        if not gold_claims:
            return 1.0, 0
        name_by_id = self._name_index(batch)
        extracted_triples: set[tuple[str, str, str]] = set()
        for relation in batch.relations:
            subject = name_by_id.get(relation.source, relation.source)
            target = name_by_id.get(relation.target, relation.target)
            extracted_triples.add(
                (resolution_key(subject), resolution_key(relation.relation_type), resolution_key(target))
            )
        for fact in batch.facts:
            subject = name_by_id.get(fact.subject, fact.subject)
            target = name_by_id.get(fact.object, fact.object)
            extracted_triples.add(
                (resolution_key(subject), resolution_key(fact.predicate), resolution_key(target))
            )
        gold_triples = {
            (
                resolution_key(str(claim.get("subject", ""))),
                resolution_key(str(claim.get("predicate", ""))),
                resolution_key(str(claim.get("object", ""))),
            )
            for claim in gold_claims
        }
        matched = gold_triples & extracted_triples
        return len(matched) / len(gold_triples) if gold_triples else 1.0, len(matched)

    def _intent_coverage_llm(
        self,
        batch: ConstructionBatch,
        gold_intents: list[str] | None,
        judge: IntentJudge | None,
        context: str | None,
    ) -> tuple[float | None, int]:
        if not gold_intents or judge is None:
            return None, 0
        graph_context = context if context is not None else self._graph_context(batch)
        covered = sum(1 for intent in gold_intents if judge.judge(intent, graph_context))
        return covered / len(gold_intents), covered

    @staticmethod
    def _name_index(batch: ConstructionBatch) -> dict[str, str]:
        index: dict[str, str] = {}
        for entity in batch.entities:
            name = str(entity.properties.get("canonical_name", entity.name))
            index[entity.candidate_id] = name
            canonical_id = entity.properties.get("canonical_entity_id")
            if canonical_id:
                index[str(canonical_id)] = name
        return index

    def _graph_context(self, batch: ConstructionBatch) -> str:
        name_by_id = self._name_index(batch)
        entity_lines = sorted({f"{entity.name} ({entity.entity_type})" for entity in batch.entities})
        relation_lines = sorted(
            {
                f"{name_by_id.get(relation.source, relation.source)} "
                f"-{relation.relation_type}-> {name_by_id.get(relation.target, relation.target)}"
                for relation in batch.relations
            }
        )
        return "Entities:\n" + "\n".join(entity_lines) + "\nRelations:\n" + "\n".join(relation_lines)


__all__ = ["CoverageEvaluator", "CoverageReport", "IntentJudge"]
