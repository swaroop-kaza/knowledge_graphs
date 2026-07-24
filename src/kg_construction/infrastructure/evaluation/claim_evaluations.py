"""Hard, evidence-aware evaluations for unstructured KG construction.

The suite does not collapse extraction quality into one opaque score.  It runs
every metric on every invocation and separately measures:

* canonicalized entity observations and exact mention spans;
* relations, event roles, intents, and facts;
* negation, modality, temporal scope, attribution, and quantities;
* exact evidence and complete provenance.

``hard_gate`` turns those measurements into an executable release decision.
There is no "not evaluated" success state: an expected metric missing from the
suite is itself a gate failure.
"""

from __future__ import annotations

from collections.abc import Hashable, Iterable, Mapping
from typing import Any, TypeVar

from pydantic import BaseModel, ConfigDict, Field

from kg_construction.application.unstructured_pipeline import PipelineResult
from kg_construction.domain.claim import Claim, ClaimKind
from kg_construction.domain.models import normalize_text

JsonDict = dict[str, Any]
Signature = tuple[str, ...]
ScoreItem = TypeVar("ScoreItem", bound=Hashable)


class EvalScore(BaseModel):
    """One fully executed metric result."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    precision: float = Field(ge=0.0, le=1.0)
    recall: float = Field(ge=0.0, le=1.0)
    f1: float = Field(ge=0.0, le=1.0)
    gold_count: int = Field(ge=0)
    predicted_count: int = Field(ge=0)
    matched: int = Field(ge=0)
    aligned_count: int = Field(default=0, ge=0)
    executed: bool = True


class HardCoverageThresholds(BaseModel):
    """Minimum release thresholds; defaults deliberately require a perfect fixture."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    entity: float = Field(default=1.0, ge=0.0, le=1.0)
    entity_span: float = Field(default=1.0, ge=0.0, le=1.0)
    relation: float = Field(default=1.0, ge=0.0, le=1.0)
    event_role: float = Field(default=1.0, ge=0.0, le=1.0)
    intent: float = Field(default=1.0, ge=0.0, le=1.0)
    fact: float = Field(default=1.0, ge=0.0, le=1.0)
    negation: float = Field(default=1.0, ge=0.0, le=1.0)
    modality: float = Field(default=1.0, ge=0.0, le=1.0)
    temporal: float = Field(default=1.0, ge=0.0, le=1.0)
    attribution: float = Field(default=1.0, ge=0.0, le=1.0)
    quantity: float = Field(default=1.0, ge=0.0, le=1.0)
    evidence: float = Field(default=1.0, ge=0.0, le=1.0)
    provenance: float = Field(default=1.0, ge=0.0, le=1.0)
    claim_coverage: float = Field(default=1.0, ge=0.0, le=1.0)
    maximum_unsupported_claim_rate: float = Field(default=0.0, ge=0.0, le=1.0)

    @property
    def metric_minimums(self) -> Mapping[str, float]:
        return {
            "entity": self.entity,
            "entity_span": self.entity_span,
            "relation": self.relation,
            "event_role": self.event_role,
            "intent": self.intent,
            "fact": self.fact,
            "negation": self.negation,
            "modality": self.modality,
            "temporal": self.temporal,
            "attribution": self.attribution,
            "quantity": self.quantity,
            "evidence": self.evidence,
            "provenance": self.provenance,
        }


class HardCoverageReport(BaseModel):
    """Auditable result of applying all hard release thresholds."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    passed: bool
    scores: dict[str, EvalScore]
    claim_coverage: float = Field(ge=0.0, le=1.0)
    unsupported_claim_rate: float = Field(ge=0.0, le=1.0)
    failures: tuple[str, ...] = ()


def _prf(name: str, gold: set[ScoreItem], predicted: set[ScoreItem]) -> EvalScore:
    matched = gold & predicted
    precision = len(matched) / len(predicted) if predicted else (1.0 if not gold else 0.0)
    recall = len(matched) / len(gold) if gold else 1.0
    f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
    return EvalScore(
        name=name,
        precision=precision,
        recall=recall,
        f1=f1,
        gold_count=len(gold),
        predicted_count=len(predicted),
        matched=len(matched),
    )


def _canonical(value: str) -> str:
    """Normalize a surface form without interpreting it as a global identity."""

    return normalize_text(value)


def _claim_signature(claim: Claim) -> Signature:
    return (
        _canonical(claim.subject),
        normalize_text(claim.predicate),
        _canonical(claim.object or ""),
    )


def _gold_signature(claim: JsonDict) -> Signature:
    return (
        normalize_text(str(claim.get("subject", ""))),
        normalize_text(str(claim.get("predicate", ""))),
        normalize_text(str(claim.get("object", ""))),
    )


def _structured_gold(items: Iterable[Any]) -> list[JsonDict]:
    return [item for item in items if isinstance(item, dict)]


class ClaimEvaluationSuite:
    """Run every held-out extraction, qualifier, and traceability metric."""

    name = "claim_evaluation_suite"

    required_metrics = frozenset(
        {
            "entity",
            "entity_span",
            "relation",
            "event_role",
            "intent",
            "fact",
            "intent_fact",
            "negation",
            "modality",
            "temporal",
            "attribution",
            "quantity",
            "evidence",
            "provenance",
        }
    )

    def evaluate(self, result: PipelineResult, gold: JsonDict) -> dict[str, EvalScore]:
        claims = tuple(result.promoted)
        gold_claims = _structured_gold(gold.get("gold_claims", ()))
        gold_intents = list(gold.get("gold_intents", ()))
        gold_facts = _structured_gold(gold.get("gold_facts", ()))
        return {
            "entity": self._entity(result, gold),
            "entity_span": self._entity_span(result, gold),
            "relation": self._semantic_kind(
                "relation", claims, gold_claims, ClaimKind.RELATION
            ),
            "event_role": self._event_role(
                claims, _structured_gold(gold.get("gold_events", ()))
            ),
            "intent": self._semantic_kind(
                "intent",
                claims,
                _structured_gold(gold_intents),
                ClaimKind.INTENT,
            ),
            "fact": self._semantic_kind("fact", claims, gold_facts, ClaimKind.FACT),
            # Kept for the modelling-correctness API. Unlike the old metric, it
            # does not replace the separate, hard intent and fact measurements.
            "intent_fact": self._intent_fact(claims, gold_intents, gold_facts),
            "negation": self._attribute(claims, gold_claims, "negation"),
            "modality": self._attribute(claims, gold_claims, "modality"),
            "temporal": self._attribute(claims, gold_claims, "temporal"),
            "attribution": self._attribute(claims, gold_claims, "attribution"),
            "quantity": self._attribute(claims, gold_claims, "quantity"),
            "evidence": self._traceability(result, "evidence"),
            "provenance": self._traceability(result, "provenance"),
        }

    def hard_gate(
        self,
        result: PipelineResult,
        gold: JsonDict,
        thresholds: HardCoverageThresholds | None = None,
    ) -> HardCoverageReport:
        """Apply every threshold and reject unsupported semantic output."""

        policy = thresholds or HardCoverageThresholds()
        scores = self.evaluate(result, gold)
        failures: list[str] = []

        missing = self.required_metrics - scores.keys()
        if missing:
            failures.append(f"required metrics did not execute: {', '.join(sorted(missing))}")

        for name, minimum in policy.metric_minimums.items():
            score = scores.get(name)
            if score is None or not score.executed:
                failures.append(f"{name} did not execute")
            elif score.f1 < minimum:
                failures.append(f"{name} F1 {score.f1:.3f} is below {minimum:.3f}")

        semantic = tuple(scores[name] for name in ("relation", "event_role", "intent", "fact"))
        gold_units = sum(score.gold_count for score in semantic)
        predicted_units = sum(score.predicted_count for score in semantic)
        matched_units = sum(score.matched for score in semantic)
        claim_coverage = matched_units / gold_units if gold_units else 1.0
        unsupported_claim_rate = (
            (predicted_units - matched_units) / predicted_units if predicted_units else 0.0
        )
        if claim_coverage < policy.claim_coverage:
            failures.append(
                f"claim coverage {claim_coverage:.3f} is below {policy.claim_coverage:.3f}"
            )
        if unsupported_claim_rate > policy.maximum_unsupported_claim_rate:
            failures.append(
                "unsupported claim rate "
                f"{unsupported_claim_rate:.3f} exceeds "
                f"{policy.maximum_unsupported_claim_rate:.3f}"
            )

        return HardCoverageReport(
            passed=not failures,
            scores=scores,
            claim_coverage=claim_coverage,
            unsupported_claim_rate=unsupported_claim_rate,
            failures=tuple(failures),
        )

    def _entity(self, result: PipelineResult, gold: JsonDict) -> EvalScore:
        predicted = {
            (
                _canonical(mention.coreference_anchor or mention.surface),
                normalize_text(mention.entity_type),
            )
            for mention in result.mentions
        }
        gold_entities = {
            (
                normalize_text(str(item.get("name", ""))),
                normalize_text(str(item.get("type", ""))),
            )
            for item in _structured_gold(gold.get("gold_entities", ()))
        }
        return _prf("entity", gold_entities, predicted)

    def _entity_span(self, result: PipelineResult, gold: JsonDict) -> EvalScore:
        predicted = {
            (
                str(mention.span.start),
                str(mention.span.end),
                mention.surface,
                normalize_text(mention.entity_type),
            )
            for mention in result.mentions
        }
        gold_mentions = {
            (
                str(item.get("start", "")),
                str(item.get("end", "")),
                str(item.get("text", "")),
                normalize_text(str(item.get("type", ""))),
            )
            for item in _structured_gold(gold.get("gold_mentions", ()))
        }
        return _prf("entity_span", gold_mentions, predicted)

    @staticmethod
    def _semantic_kind(
        name: str,
        claims: tuple[Claim, ...],
        gold_items: list[JsonDict],
        kind: ClaimKind,
    ) -> EvalScore:
        predicted = {
            _claim_signature(claim) for claim in claims if claim.kind == kind
        }
        gold = {_gold_signature(item) for item in gold_items}
        return _prf(name, gold, predicted)

    @staticmethod
    def _event_role(claims: tuple[Claim, ...], gold_events: list[JsonDict]) -> EvalScore:
        predicted = {
            (
                normalize_text(claim.predicate),
                normalize_text(role.role),
                _canonical(role.value),
            )
            for claim in claims
            if claim.kind == ClaimKind.EVENT
            for role in claim.event_roles
        }
        gold = {
            (
                normalize_text(str(event.get("event", ""))),
                normalize_text(str(role)),
                normalize_text(str(filler)),
            )
            for event in gold_events
            for role, filler in dict(event.get("roles", {})).items()
        }
        return _prf("event_role", gold, predicted)

    @staticmethod
    def _intent_fact(
        claims: tuple[Claim, ...],
        gold_intents: list[Any],
        gold_facts: list[JsonDict],
    ) -> EvalScore:
        structured = _structured_gold(gold_intents) + gold_facts
        legacy_text = [item for item in gold_intents if isinstance(item, str)]
        if structured:
            predicted = {
                _claim_signature(claim)
                for claim in claims
                if claim.kind in {ClaimKind.INTENT, ClaimKind.FACT}
            }
            exact = _prf(
                "intent_fact",
                {_gold_signature(item) for item in structured},
                predicted,
            )
            if not legacy_text:
                return exact
        else:
            exact = _prf("intent_fact", set(), set())

        # Backward compatibility for modelling fixtures that supplied prose
        # rather than typed intent annotations.
        covered = 0
        for statement in legacy_text:
            normalized = normalize_text(statement)
            if any(
                _canonical(claim.subject) in normalized
                and bool(claim.object)
                and _canonical(claim.object or "") in normalized
                for claim in claims
            ):
                covered += 1
        gold_count = exact.gold_count + len(legacy_text)
        matched = exact.matched + covered
        recall = matched / gold_count if gold_count else 1.0
        precision = exact.precision if structured else recall
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        return EvalScore(
            name="intent_fact",
            precision=precision,
            recall=recall,
            f1=f1,
            gold_count=gold_count,
            predicted_count=exact.predicted_count,
            matched=matched,
        )

    @staticmethod
    def _attribute(
        claims: tuple[Claim, ...],
        gold_claims: list[JsonDict],
        attribute: str,
    ) -> EvalScore:
        by_signature = {
            _claim_signature(claim): claim
            for claim in claims
            if claim.kind == ClaimKind.RELATION
        }
        aligned = 0
        agreements = 0
        for gold in gold_claims:
            claim = by_signature.get(_gold_signature(gold))
            if claim is None:
                continue
            aligned += 1
            agreements += int(_attribute_agrees(claim, gold, attribute))

        # Recall is deliberately over every gold claim. Missing a relation must
        # not make qualifier accuracy look better by shrinking the denominator.
        recall = agreements / len(gold_claims) if gold_claims else 1.0
        precision = agreements / aligned if aligned else (1.0 if not gold_claims else 0.0)
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        return EvalScore(
            name=attribute,
            precision=precision,
            recall=recall,
            f1=f1,
            gold_count=len(gold_claims),
            predicted_count=aligned,
            matched=agreements,
            aligned_count=aligned,
        )

    @staticmethod
    def _traceability(result: PipelineResult, name: str) -> EvalScore:
        claims = result.promoted
        if name == "evidence":
            matched = sum(claim.has_exact_evidence(result.document_text) for claim in claims)
        else:
            matched = sum(
                claim.has_complete_provenance(result.document_text) for claim in claims
            )
        coverage = matched / len(claims) if claims else 1.0
        return EvalScore(
            name=name,
            precision=coverage,
            recall=coverage,
            f1=coverage,
            gold_count=len(claims),
            predicted_count=len(claims),
            matched=matched,
            aligned_count=len(claims),
        )


def _attribute_agrees(claim: Claim, gold: JsonDict, attribute: str) -> bool:
    if attribute == "negation":
        return claim.negated == bool(gold.get("negated", False))
    if attribute == "modality":
        return claim.modality.value == str(gold.get("modality", "asserted"))
    if attribute == "temporal":
        expected = gold.get("temporal")
        return (
            normalize_text(claim.temporal or "")
            == normalize_text(str(expected) if expected is not None else "")
        )
    if attribute == "attribution":
        expected = gold.get("attribution")
        return (
            normalize_text(claim.attribution or "")
            == normalize_text(str(expected) if expected is not None else "")
        )
    if attribute == "quantity":
        expected = gold.get("quantity")
        return (
            normalize_text(claim.quantity or "")
            == normalize_text(str(expected) if expected is not None else "")
        )
    return False


__all__ = [
    "ClaimEvaluationSuite",
    "EvalScore",
    "HardCoverageReport",
    "HardCoverageThresholds",
]
