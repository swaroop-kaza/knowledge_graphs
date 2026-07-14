"""Recommendation engine for KG construction technique selection."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from kg_construction.application.scoring import ProductionThresholds, TechniqueScore


class RecommendationDecision(BaseModel):
    """A recommendation slot such as best single or production-safe."""

    model_config = ConfigDict(extra="forbid")

    name: str
    technique_name: str | None = None
    final_score: float = 0.0
    confidence_interval: tuple[float, float] = (0.0, 0.0)
    reason: str


class RecommendationSummary(BaseModel):
    """Complete recommendation summary."""

    model_config = ConfigDict(extra="forbid")

    best_single_technique: RecommendationDecision
    best_hybrid_technique: RecommendationDecision
    production_safe_technique: RecommendationDecision
    high_recall_technique: RecommendationDecision
    low_cost_technique: RecommendationDecision
    graphrag_ready_technique: RecommendationDecision
    fallback_recommendation: str | None = None
    limitations: list[str] = Field(default_factory=list)


class RecommendationEngine:
    """Pick best techniques using scores and explicit decision rules."""

    def __init__(self, thresholds: ProductionThresholds | None = None) -> None:
        self.thresholds = thresholds or ProductionThresholds()

    def recommend(self, scores: list[TechniqueScore]) -> RecommendationSummary:
        ranked = list(scores)
        if not ranked:
            empty = RecommendationDecision(
                name="unavailable",
                reason="No candidate technique produced a graph.",
            )
            return RecommendationSummary(
                best_single_technique=empty,
                best_hybrid_technique=empty,
                production_safe_technique=empty,
                high_recall_technique=empty,
                low_cost_technique=empty,
                graphrag_ready_technique=empty,
                fallback_recommendation="deterministic mapping + rules + Ollama for low-confidence cases + human review",
                limitations=["No executable candidates were available."],
            )

        best_single = self._decision("best_single_technique", ranked[0], "Highest final weighted score.")
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
        production_safe = self._production_safe(ranked)
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
        fallback = None
        limitations: list[str] = []
        if production_safe.technique_name is None:
            fallback = "deterministic mapping + rules + GLiNER/Ollama only for low-confidence cases + human review"
            limitations.append("No technique satisfied all production-safe thresholds.")
        if all(score.metrics.get("entity_f1", 0.0) == 0.0 for score in ranked):
            limitations.append("No gold labels were supplied, so accuracy is estimated from graph quality and coverage.")

        return RecommendationSummary(
            best_single_technique=best_single,
            best_hybrid_technique=best_hybrid,
            production_safe_technique=production_safe,
            high_recall_technique=self._decision("high_recall_technique", high_recall, "Highest recall or coverage proxy."),
            low_cost_technique=self._decision("low_cost_technique", low_cost, "Highest cost-efficiency score."),
            graphrag_ready_technique=self._decision("graphrag_ready_technique", graphrag, "Best GraphRAG weighted evidence/retrieval/latency balance."),
            fallback_recommendation=fallback,
            limitations=limitations,
        )

    def _production_safe(self, ranked: list[TechniqueScore]) -> RecommendationDecision:
        for score in ranked:
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
                    "production_safe_technique",
                    score,
                    "Satisfies production thresholds for evidence, schema validity, unsupported extraction, and validation failure rate.",
                )
        return RecommendationDecision(
            name="production_safe_technique",
            technique_name=None,
            reason="No technique satisfied the configured production-safe thresholds.",
        )

    @staticmethod
    def _decision(name: str, score: TechniqueScore, reason: str) -> RecommendationDecision:
        return RecommendationDecision(
            name=name,
            technique_name=score.technique_name,
            final_score=score.final_score,
            confidence_interval=score.confidence_interval,
            reason=f"{reason} {score.recommendation}",
        )


__all__ = ["RecommendationDecision", "RecommendationEngine", "RecommendationSummary"]
