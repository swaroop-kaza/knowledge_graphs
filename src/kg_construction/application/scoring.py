"""Weighted mathematical scoring for technique ranking."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from kg_construction.application.dataset_profiler import clamp
from kg_construction.application.registry import TechniqueDescriptor


class ScoringWeights(BaseModel):
    """Default weights from the evaluation framework."""

    model_config = ConfigDict(extra="forbid")

    accuracy: float = 0.25
    validation: float = 0.20
    coverage: float = 0.15
    entity_resolution: float = 0.10
    retrieval: float = 0.10
    scalability: float = 0.08
    cost_efficiency: float = 0.07
    maintainability: float = 0.05


class PenaltyWeights(BaseModel):
    """Penalty weights from the evaluation framework."""

    model_config = ConfigDict(extra="forbid")

    unsupported_extraction: float | None = None
    hallucination: float = 0.25
    schema_violation: float = 0.20
    latency: float = 0.10
    human_review: float = 0.05


class ProductionThresholds(BaseModel):
    """Production-safe decision thresholds."""

    model_config = ConfigDict(extra="forbid")

    evidence_coverage: float = 0.95
    schema_validity: float = 0.95
    unsupported_extraction_rate: float | None = None
    hallucination_rate: float = 0.02
    validation_failure_rate: float = 0.05

    @property
    def unsupported_extraction_threshold(self) -> float:
        """Return the preferred threshold while supporting older configs."""

        if self.unsupported_extraction_rate is not None:
            return self.unsupported_extraction_rate
        return self.hallucination_rate


class TechniqueScore(BaseModel):
    """Final score and decomposed scoring dimensions."""

    model_config = ConfigDict(extra="forbid")

    technique_name: str
    pipeline_name: str
    category: str = "unknown"
    method_type: str = "unknown"
    status: str = "implemented"
    final_score: float
    quality_score: float = 0.0
    runtime_score: float = 0.0
    dimensions: dict[str, float] = Field(default_factory=dict)
    penalties: dict[str, float] = Field(default_factory=dict)
    metrics: dict[str, float] = Field(default_factory=dict)
    applicability_score: float = 0.0
    confidence_interval: tuple[float, float] = (0.0, 0.0)
    recommendation: str = ""


def bounded_metric(metrics: dict[str, float], name: str, default: float = 0.0) -> float:
    """Read and clamp a bounded metric."""

    return clamp(float(metrics.get(name, default)))


def inverse_unbounded(value: float) -> float:
    """Normalize lower-is-better unbounded values into 0..1."""

    return 1.0 / (1.0 + max(0.0, value))


class ScoringEngine:
    """Compute FinalScore(t) from metrics, weights, and penalties."""

    def __init__(
        self,
        *,
        weights: ScoringWeights | None = None,
        penalties: PenaltyWeights | None = None,
    ) -> None:
        self.weights = weights or ScoringWeights()
        self.penalty_weights = penalties or PenaltyWeights()

    def score(
        self,
        *,
        technique_name: str,
        pipeline_name: str,
        metrics: dict[str, float],
        descriptor: TechniqueDescriptor | None = None,
        applicability_score: float = 0.0,
    ) -> TechniqueScore:
        dimensions = {
            "accuracy": self._accuracy_score(metrics),
            "validation": self._validation_score(metrics),
            "coverage": self._coverage_score(metrics),
            "entity_resolution": self._entity_resolution_score(metrics),
            "retrieval": self._retrieval_score(metrics),
            "scalability": self._scalability_score(metrics),
            "cost_efficiency": self._cost_efficiency_score(metrics),
            "maintainability": self._maintainability_score(descriptor),
        }
        unsupported_extraction_rate = bounded_metric(
            metrics,
            "unsupported_extraction_rate",
            bounded_metric(
                metrics,
                "hallucination_risk_proxy",
                bounded_metric(metrics, "hallucination_rate", bounded_metric(metrics, "hallucinated_fact_rate")),
            ),
        )
        penalties = {
            "unsupported_extraction": unsupported_extraction_rate,
            "schema_violation": bounded_metric(metrics, "schema_violation_rate", 1.0 - bounded_metric(metrics, "schema_validity_score", 1.0)),
            "latency": bounded_metric(metrics, "latency_penalty", 1.0 - inverse_unbounded(metrics.get("latency", 0.0))),
            "human_review": bounded_metric(metrics, "human_review_rate", 0.0),
        }
        quality_positive = (
            dimensions["accuracy"] * self.weights.accuracy
            + dimensions["validation"] * self.weights.validation
            + dimensions["coverage"] * self.weights.coverage
            + dimensions["entity_resolution"] * self.weights.entity_resolution
            + dimensions["retrieval"] * self.weights.retrieval
            + dimensions["maintainability"] * self.weights.maintainability
        )
        runtime_positive = (
            dimensions["scalability"] * self.weights.scalability
            + dimensions["cost_efficiency"] * self.weights.cost_efficiency
        )
        unsupported_extraction_weight = (
            self.penalty_weights.unsupported_extraction
            if self.penalty_weights.unsupported_extraction is not None
            else self.penalty_weights.hallucination
        )
        quality_negative = (
            penalties["unsupported_extraction"] * unsupported_extraction_weight
            + penalties["schema_violation"] * self.penalty_weights.schema_violation
            + penalties["human_review"] * self.penalty_weights.human_review
        )
        runtime_negative = penalties["latency"] * self.penalty_weights.latency
        quality_raw = quality_positive - quality_negative
        runtime_raw = runtime_positive - runtime_negative
        final_score = clamp(quality_raw + runtime_raw)
        return TechniqueScore(
            technique_name=technique_name,
            pipeline_name=pipeline_name,
            category=descriptor.category if descriptor else "unknown",
            method_type=descriptor.method_type if descriptor else "unknown",
            status=descriptor.status if descriptor else "implemented",
            final_score=final_score,
            quality_score=clamp(quality_raw),
            runtime_score=clamp(runtime_raw),
            dimensions=dimensions,
            penalties=penalties,
            metrics=metrics,
            applicability_score=applicability_score,
            confidence_interval=(max(0.0, final_score - 0.05), min(1.0, final_score + 0.05)),
            recommendation=self._recommendation(final_score, dimensions, penalties),
        )

    @staticmethod
    def _accuracy_score(metrics: dict[str, float]) -> float:
        values = [
            bounded_metric(metrics, "entity_f1"),
            bounded_metric(metrics, "relation_f1"),
            bounded_metric(metrics, "triple_exact_match"),
            bounded_metric(metrics, "span_f1", bounded_metric(metrics, "span_level_f1")),
            bounded_metric(metrics, "estimated_precision"),
        ]
        non_zero = [value for value in values if value > 0]
        if non_zero:
            return sum(non_zero) / len(non_zero)
        return bounded_metric(metrics, "provenance_coverage", 0.5) * 0.5

    @staticmethod
    def _validation_score(metrics: dict[str, float]) -> float:
        return sum(
            [
                bounded_metric(metrics, "schema_validity_score", 1.0),
                bounded_metric(metrics, "rdf_output_validity_score", bounded_metric(metrics, "shacl_validity_score", 1.0)),
                bounded_metric(metrics, "evidence_coverage", bounded_metric(metrics, "provenance_coverage", 1.0)),
                bounded_metric(metrics, "provenance_coverage", 1.0),
                1.0 - bounded_metric(metrics, "duplicate_edge_rate"),
                1.0 - bounded_metric(metrics, "orphan_node_rate"),
                1.0 - bounded_metric(metrics, "validation_failure_rate"),
            ]
        ) / 7

    @staticmethod
    def _coverage_score(metrics: dict[str, float]) -> float:
        return sum(
            [
                bounded_metric(metrics, "entity_coverage", 0.0),
                bounded_metric(metrics, "relation_coverage", 0.0),
                bounded_metric(metrics, "source_coverage", 0.0),
                bounded_metric(metrics, "attribute_coverage", 0.0),
            ]
        ) / 4

    @staticmethod
    def _entity_resolution_score(metrics: dict[str, float]) -> float:
        values = [
            bounded_metric(metrics, "pairwise_er_f1"),
            bounded_metric(metrics, "cluster_f1"),
            1.0 - bounded_metric(metrics, "false_merge_rate"),
            1.0 - bounded_metric(metrics, "false_split_rate"),
        ]
        return sum(values) / len(values)

    @staticmethod
    def _retrieval_score(metrics: dict[str, float]) -> float:
        return sum(
            [
                bounded_metric(metrics, "retrieval_recall_at_k"),
                bounded_metric(metrics, "path_relevance"),
                bounded_metric(metrics, "multi_hop_success_rate", bounded_metric(metrics, "multi_hop_retrieval_success")),
                bounded_metric(metrics, "citation_accuracy", bounded_metric(metrics, "answer_citation_accuracy")),
                bounded_metric(metrics, "answer_faithfulness"),
            ]
        ) / 5

    @staticmethod
    def _scalability_score(metrics: dict[str, float]) -> float:
        return bounded_metric(metrics, "scalability_score", inverse_unbounded(metrics.get("latency", 0.0)))

    @staticmethod
    def _cost_efficiency_score(metrics: dict[str, float]) -> float:
        return bounded_metric(metrics, "cost_efficiency_score", inverse_unbounded(metrics.get("ollama_cost_proxy", 0.0)))

    @staticmethod
    def _maintainability_score(descriptor: TechniqueDescriptor | None) -> float:
        if descriptor is None:
            return 0.7
        method_bonus = {
            "deterministic": 1.0,
            "rules": 0.9,
            "ontology": 0.85,
            "nlp": 0.75,
            "ml": 0.65,
            "embedding": 0.6,
            "llm": 0.55,
            "human": 0.45,
        }.get(descriptor.method_type, 0.6)
        status_factor = {"implemented": 1.0, "adapter": 0.75, "stub": 0.4}.get(descriptor.status, 0.5)
        return method_bonus * status_factor

    @staticmethod
    def _recommendation(
        final_score: float,
        dimensions: dict[str, float],
        penalties: dict[str, float],
    ) -> str:
        best_dimension = max(dimensions, key=lambda name: dimensions[name])
        worst_penalty = max(penalties, key=lambda name: penalties[name])
        return (
            f"Score {final_score:.3f}; strongest dimension is {best_dimension}="
            f"{dimensions[best_dimension]:.2f}; largest penalty is {worst_penalty}="
            f"{penalties[worst_penalty]:.2f}."
        )


__all__ = [
    "PenaltyWeights",
    "ProductionThresholds",
    "ScoringEngine",
    "ScoringWeights",
    "TechniqueScore",
    "bounded_metric",
    "inverse_unbounded",
]
