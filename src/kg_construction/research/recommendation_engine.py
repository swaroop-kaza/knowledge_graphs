"""Recommendation engine for comparative KG-construction research."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from kg_construction.research.scoring import (
    ResearchScreeningThresholds,
    TechniqueScore,
)


class RecommendationDecision(BaseModel):
    """One non-authoritative comparison decision."""

    model_config = ConfigDict(extra="forbid")

    name: str
    technique_name: str | None = None
    final_score: float = 0.0
    confidence_interval: tuple[float, float] = (0.0, 0.0)
    confidence_interval_method: str = "not_estimated"
    confidence_interval_note: str = (
        "No statistical confidence interval was estimated for the weighted score."
    )
    reason: str


class RecommendationSummary(BaseModel):
    """Complete recommendation summary."""

    model_config = ConfigDict(extra="forbid")

    best_single_technique: RecommendationDecision
    best_hybrid_technique: RecommendationDecision
    threshold_passing_technique: RecommendationDecision
    high_recall_technique: RecommendationDecision
    low_cost_technique: RecommendationDecision
    graphrag_ready_technique: RecommendationDecision
    contingency_recommendation: str | None = None
    limitations: list[str] = Field(default_factory=list)
    evaluated_gold_tasks: list[str] = Field(default_factory=list)
    research_gate_passing_techniques: list[str] = Field(default_factory=list)

    @property
    def production_safe_technique(self) -> RecommendationDecision:
        """Deprecated alias; research cannot establish production safety."""

        return self.threshold_passing_technique

    @property
    def fallback_recommendation(self) -> str | None:
        """Deprecated alias for the research contingency text."""

        return self.contingency_recommendation

    @property
    def publication_eligible_techniques(self) -> list[str]:
        """Deprecated alias; release eligibility belongs to certification."""

        return self.research_gate_passing_techniques


class RecommendationEngine:
    """Pick best techniques using scores and explicit decision rules."""

    def __init__(self, thresholds: ResearchScreeningThresholds | None = None) -> None:
        self.thresholds = thresholds or ResearchScreeningThresholds()

    def recommend(
        self,
        scores: list[TechniqueScore],
        *,
        research_gate_passing_techniques: set[str] | None = None,
        evaluated_gold_tasks: set[str] | None = None,
    ) -> RecommendationSummary:
        ranked = list(scores)
        gold_tasks = sorted(evaluated_gold_tasks or set())
        gate_passing_names = sorted(research_gate_passing_techniques or set())
        if not ranked:
            empty = RecommendationDecision(
                name="unavailable",
                reason="No candidate technique produced a graph.",
            )
            return RecommendationSummary(
                best_single_technique=empty,
                best_hybrid_technique=empty,
                threshold_passing_technique=empty,
                high_recall_technique=empty,
                low_cost_technique=empty,
                graphrag_ready_technique=empty,
                contingency_recommendation=(
                    "Inspect the failed candidates and rerun the complete locked "
                    "construction profile; do not omit providers."
                ),
                limitations=["No executable candidates were available."],
                evaluated_gold_tasks=gold_tasks,
                research_gate_passing_techniques=gate_passing_names,
            )

        best_single = self._decision(
            "best_ranked_or_smoke_result",
            ranked[0],
            "Highest final weighted score in this non-authoritative comparison.",
        )
        hybrids = [score for score in ranked if score.category == "hybrid" or "hybrid" in score.pipeline_name]
        best_hybrid = (
            self._decision("best_hybrid_technique", hybrids[0], "Highest final score among hybrid pipelines.")
            if hybrids
            else RecommendationDecision(
                name="best_hybrid_technique",
                technique_name=None,
                reason="No hybrid pipeline was executable for this dataset.",
            )
        )
        threshold_passing = self._threshold_passing(
            ranked,
            research_gate_passing_techniques=research_gate_passing_techniques,
        )
        high_recall = max(ranked, key=lambda item: item.metrics.get("entity_recall", item.dimensions.get("coverage", 0.0)))
        low_cost = max(ranked, key=lambda item: item.dimensions.get("cost_efficiency", 0.0))
        graphrag = max(
            ranked,
            key=lambda item: (
                item.metrics.get("retrieval_recall_at_k", 0.0) * 0.3
                + item.metrics.get("citation_accuracy", item.metrics.get("answer_citation_accuracy", 0.0)) * 0.25
                + item.metrics.get("evidence_coverage", 0.0) * 0.25
                + item.metrics.get("path_relevance", 0.0) * 0.1
                + (1.0 - item.penalties.get("latency", 0.0)) * 0.1
            ),
        )
        contingency = None
        limitations: list[str] = []
        if threshold_passing.technique_name is None:
            contingency = (
                "Inspect failed research gates and rerun the complete locked "
                "construction profile; do not omit providers."
            )
            limitations.append(
                "No technique satisfied all configured research thresholds."
            )
        if not gold_tasks:
            limitations.append("No gold labels were supplied, so accuracy is estimated from graph quality and coverage.")
        elif all(score.dimensions.get("accuracy", 0.0) == 0.0 for score in ranked):
            limitations.append(
                "Gold tasks were evaluated, but every candidate received zero in the "
                "gold-backed accuracy dimension."
            )

        return RecommendationSummary(
            best_single_technique=best_single,
            best_hybrid_technique=best_hybrid,
            threshold_passing_technique=threshold_passing,
            high_recall_technique=self._decision("high_recall_technique", high_recall, "Highest recall or coverage proxy."),
            low_cost_technique=self._decision("low_cost_technique", low_cost, "Highest cost-efficiency score."),
            graphrag_ready_technique=self._decision("graphrag_ready_technique", graphrag, "Best GraphRAG weighted evidence/retrieval/latency balance."),
            contingency_recommendation=contingency,
            limitations=limitations,
            evaluated_gold_tasks=gold_tasks,
            research_gate_passing_techniques=gate_passing_names,
        )

    def _threshold_passing(
        self,
        ranked: list[TechniqueScore],
        *,
        research_gate_passing_techniques: set[str] | None = None,
    ) -> RecommendationDecision:
        for score in ranked:
            if (
                research_gate_passing_techniques is not None
                and score.technique_name not in research_gate_passing_techniques
            ):
                continue
            metrics = score.metrics
            unsupported_extraction_rate = metrics.get(
                "unsupported_extraction_rate",
                metrics.get("hallucination_risk_proxy", metrics.get("hallucination_rate", 0.0)),
            )
            if (
                metrics.get("evidence_coverage", 0.0) >= self.thresholds.evidence_coverage
                and metrics.get("schema_validity_score", 0.0) >= self.thresholds.schema_validity
                and unsupported_extraction_rate <= self.thresholds.unsupported_extraction_threshold
                and metrics.get("validation_failure_rate", 1.0) <= self.thresholds.validation_failure_rate
            ):
                return self._decision(
                    "threshold_passing_technique",
                    score,
                    "Satisfies the configured research thresholds and gold/task gates.",
                )
        gate_reason = (
            " No candidate passed all configured research gates."
            if research_gate_passing_techniques is not None
            else ""
        )
        return RecommendationDecision(
            name="threshold_passing_technique",
            technique_name=None,
            reason=(
                "No technique satisfied the configured research thresholds."
                f"{gate_reason}"
            ),
        )

    @staticmethod
    def _decision(name: str, score: TechniqueScore, reason: str) -> RecommendationDecision:
        return RecommendationDecision(
            name=name,
            technique_name=score.technique_name,
            final_score=score.final_score,
            confidence_interval=score.confidence_interval,
            confidence_interval_method=score.confidence_interval_method,
            confidence_interval_note=score.confidence_interval_note,
            reason=f"{reason} {score.recommendation}",
        )


__all__ = ["RecommendationDecision", "RecommendationEngine", "RecommendationSummary"]
