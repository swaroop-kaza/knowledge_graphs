"""Dataset-aware technique selection for comparative research."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from kg_construction.certification.registry import (
    PluginRegistry,
    TechniqueDescriptor,
    default_registry,
)
from kg_construction.construction.profiling import DatasetProfile, clamp


class ApplicabilityWeights(BaseModel):
    """Weights for the Applicability(t, d) formula."""

    model_config = ConfigDict(extra="forbid")

    format_match: float = 0.20
    schema_match: float = 0.15
    structure_match: float = 0.18
    text_suitability: float = 0.15
    ontology_suitability: float = 0.10
    scale_suitability: float = 0.10
    cost_suitability: float = 0.12


class ApplicabilityResult(BaseModel):
    """Applicability score for a technique and dataset profile."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    technique_name: str
    category: str
    method_type: str
    status: str
    score: float
    components: dict[str, float] = Field(default_factory=dict)
    applicable: bool
    reason: str
    descriptor: TechniqueDescriptor | None = Field(default=None, exclude=True)


class TechniqueSelector:
    """Select applicable techniques without hard-coding a winner."""

    def __init__(
        self,
        *,
        registry: PluginRegistry | None = None,
        weights: ApplicabilityWeights | None = None,
        threshold: float = 0.40,
    ) -> None:
        self.registry = registry or default_registry()
        self.weights = weights or ApplicabilityWeights()
        self.threshold = threshold

    def score(self, descriptor: TechniqueDescriptor, profile: DatasetProfile) -> ApplicabilityResult:
        components = {
            "format_match": self._format_match(descriptor, profile),
            "schema_match": self._schema_match(descriptor, profile),
            "structure_match": self._structure_match(descriptor, profile),
            "text_suitability": self._text_suitability(descriptor, profile),
            "ontology_suitability": self._ontology_suitability(descriptor, profile),
            "scale_suitability": self._scale_suitability(descriptor, profile),
            "cost_suitability": self._cost_suitability(descriptor),
        }
        score = clamp(
            components["format_match"] * self.weights.format_match
            + components["schema_match"] * self.weights.schema_match
            + components["structure_match"] * self.weights.structure_match
            + components["text_suitability"] * self.weights.text_suitability
            + components["ontology_suitability"] * self.weights.ontology_suitability
            + components["scale_suitability"] * self.weights.scale_suitability
            + components["cost_suitability"] * self.weights.cost_suitability
        )
        return ApplicabilityResult(
            technique_name=descriptor.name,
            category=descriptor.category,
            method_type=descriptor.method_type,
            status=descriptor.status,
            score=score,
            components=components,
            applicable=score >= self.threshold,
            reason=self._reason(descriptor, profile, components),
            descriptor=descriptor,
        )

    def select(self, profile: DatasetProfile, *, force_all: bool = False) -> list[ApplicabilityResult]:
        results = [self.score(descriptor, profile) for descriptor in self.registry.techniques]
        selected = results if force_all else [result for result in results if result.applicable]
        return sorted(selected, key=lambda item: item.score, reverse=True)

    @staticmethod
    def _format_match(descriptor: TechniqueDescriptor, profile: DatasetProfile) -> float:
        if not descriptor.input_types:
            return 0.55
        if profile.file_format in descriptor.input_types or profile.data_type in descriptor.input_types:
            return 1.0
        if profile.file_format == "txt" and "text" in descriptor.input_types:
            return 1.0
        if profile.data_type == "mixed":
            return 0.7
        return 0.1

    @staticmethod
    def _schema_match(descriptor: TechniqueDescriptor, profile: DatasetProfile) -> float:
        if descriptor.method_type == "ontology":
            return clamp(profile.schema_quality_score * 0.6 + profile.ontology_alignment_score * 0.4)
        if descriptor.method_type == "llm":
            return clamp(1.0 - (profile.schema_quality_score * 0.4))
        if descriptor.category == "mapper":
            return profile.schema_quality_score
        return clamp(0.4 + profile.schema_quality_score * 0.4)

    @staticmethod
    def _structure_match(descriptor: TechniqueDescriptor, profile: DatasetProfile) -> float:
        if descriptor.method_type == "deterministic":
            return profile.structuredness_score
        if descriptor.method_type == "rules":
            return clamp(max(profile.structuredness_score * 0.5, profile.entity_density_score))
        if descriptor.method_type in {"ml", "nlp", "llm"}:
            return clamp(profile.text_density * 0.7 + profile.nestedness_score * 0.3)
        return 0.5

    @staticmethod
    def _text_suitability(descriptor: TechniqueDescriptor, profile: DatasetProfile) -> float:
        if "text" in descriptor.input_types or descriptor.method_type in {"nlp", "ml", "llm"}:
            return clamp(profile.text_density * 0.35 + profile.text_complexity_score * 0.35 + profile.entity_density_score * 0.3)
        return clamp(1.0 - profile.text_density)

    @staticmethod
    def _ontology_suitability(descriptor: TechniqueDescriptor, profile: DatasetProfile) -> float:
        if descriptor.method_type == "ontology":
            return clamp(max(profile.ontology_alignment_score, profile.schema_quality_score))
        if descriptor.category == "validation":
            return clamp(0.6 + profile.schema_quality_score * 0.4)
        return 0.4

    @staticmethod
    def _scale_suitability(descriptor: TechniqueDescriptor, profile: DatasetProfile) -> float:
        if descriptor.method_type == "llm":
            return clamp(1.0 - profile.data_volume_score * 0.8)
        if descriptor.method_type in {"deterministic", "rules"}:
            return 0.95
        if descriptor.method_type in {"ml", "embedding"}:
            return 0.7
        return 0.6

    @staticmethod
    def _cost_suitability(descriptor: TechniqueDescriptor) -> float:
        if descriptor.method_type == "deterministic":
            return 1.0
        if descriptor.method_type == "rules":
            return 0.95
        if descriptor.method_type in {"nlp", "ontology"}:
            return 0.80
        if descriptor.method_type in {"ml", "embedding"}:
            return 0.60
        if descriptor.method_type == "llm":
            return 0.35
        if descriptor.method_type == "human":
            return 0.20
        return 0.5

    @staticmethod
    def _reason(
        descriptor: TechniqueDescriptor,
        profile: DatasetProfile,
        components: dict[str, float],
    ) -> str:
        strongest = max(components, key=lambda name: components[name])
        weakest = min(components, key=lambda name: components[name])
        return (
            f"{descriptor.name} scored highest on {strongest}={components[strongest]:.2f} "
            f"and lowest on {weakest}={components[weakest]:.2f} for {profile.data_type} data."
        )


__all__ = ["ApplicabilityResult", "ApplicabilityWeights", "TechniqueSelector"]
