"""Notebook-friendly Ollama label proposal plus GLiNER extraction workflow."""

from __future__ import annotations

from pathlib import Path
from statistics import mean

import yaml
from pydantic import BaseModel, ConfigDict, Field

from kg_construction.domain.models import ConstructionBatch, SourceRecord
from kg_construction.infrastructure.extractors.gazetteer import GazetteerEntityExtractor
from kg_construction.infrastructure.extractors.gliner import (
    GlinerConfig,
    GlinerEntityExtractor,
    GlinerModelProtocol,
)
from kg_construction.infrastructure.extractors.ollama_graph import OllamaGraphExtractor
from kg_construction.infrastructure.extractors.regex import RegexEntityExtractor
from kg_construction.infrastructure.llm.label_proposal import (
    LabelProposalResult,
    LabelProposalSettings,
    OllamaLabelProposalExtractor,
)
from kg_construction.infrastructure.llm.ollama import OllamaConfig, OllamaProvider

DEFAULT_HYBRID_GAZETTEER = {
    "Organization": ["Platform Team", "Research", "Compiler", "Flight", "ML"],
    "Incident": ["INCIDENT-1042"],
    "System": ["Recommendation API", "Feature Store"],
}


class LabelReviewSettings(BaseModel):
    """Manual review overrides for proposed labels."""

    model_config = ConfigDict(extra="forbid")

    approved_labels: list[str] = Field(default_factory=list)
    rejected_labels: list[str] = Field(default_factory=list)


class HybridGlinerPipelineConfig(BaseModel):
    """YAML configuration for Ollama-labels plus GLiNER extraction."""

    model_config = ConfigDict(extra="forbid")

    llm: OllamaConfig = Field(default_factory=OllamaConfig)
    label_proposal: LabelProposalSettings = Field(default_factory=LabelProposalSettings)
    review: LabelReviewSettings = Field(default_factory=LabelReviewSettings)
    gliner: GlinerConfig = Field(default_factory=GlinerConfig)
    compare_regex: bool = True
    compare_gazetteer: bool = True
    compare_ollama_graph: bool = False

    @classmethod
    def from_yaml(cls, path: str | Path) -> HybridGlinerPipelineConfig:
        payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return cls.model_validate(payload)


class ExtractionComparisonRow(BaseModel):
    """Compact comparison row for extraction strategies."""

    model_config = ConfigDict(extra="forbid")

    technique: str
    entity_count: int = 0
    relation_count: int = 0
    labels: list[str] = Field(default_factory=list)
    evidence_coverage: float = 0.0
    average_confidence: float = 0.0
    error: str | None = None


class HybridGlinerExtractionResult(BaseModel):
    """Complete output for notebook display and downstream graph construction."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    source: SourceRecord
    label_proposal: LabelProposalResult
    reviewed_labels: list[str]
    gliner_batch: ConstructionBatch
    comparisons: list[ExtractionComparisonRow] = Field(default_factory=list)


def extract_entities_with_ollama_labels_and_gliner(
    text: str,
    *,
    config: HybridGlinerPipelineConfig | None = None,
    config_path: str | Path | None = None,
    provider: OllamaProvider | None = None,
    gliner_model: GlinerModelProtocol | None = None,
    source_name: str = "memory://ollama-gliner",
) -> HybridGlinerExtractionResult:
    """Run text -> Ollama labels -> reviewed labels -> GLiNER span candidates."""

    pipeline_config = config or (
        HybridGlinerPipelineConfig.from_yaml(config_path)
        if config_path is not None
        else HybridGlinerPipelineConfig()
    )
    source = SourceRecord.from_text(text, source_type="text", source_name=source_name)
    llm_provider = provider or OllamaProvider(pipeline_config.llm)
    label_chain = OllamaLabelProposalExtractor(
        llm_provider,
        settings=pipeline_config.label_proposal,
    )
    proposal = label_chain.propose(text)
    reviewed_labels = apply_manual_label_review(
        proposal.reviewed_label_names,
        approved_labels=pipeline_config.review.approved_labels,
        rejected_labels=pipeline_config.review.rejected_labels,
    )
    gliner_batch = GlinerEntityExtractor(
        pipeline_config.gliner,
        model=gliner_model,
    ).extract(source, labels=reviewed_labels)
    comparisons = compare_entity_extraction_strategies(
        source,
        gliner_batch=gliner_batch,
        reviewed_labels=reviewed_labels,
        provider=llm_provider,
        config=pipeline_config,
    )
    return HybridGlinerExtractionResult(
        source=source,
        label_proposal=proposal,
        reviewed_labels=reviewed_labels,
        gliner_batch=gliner_batch,
        comparisons=comparisons,
    )


def apply_manual_label_review(
    labels: list[str],
    *,
    approved_labels: list[str],
    rejected_labels: list[str],
) -> list[str]:
    """Apply optional notebook/user review overrides to approved labels."""

    rejected = {label.casefold() for label in rejected_labels}
    if approved_labels:
        approved = {label.casefold() for label in approved_labels}
        return [label for label in labels if label.casefold() in approved and label.casefold() not in rejected]
    return [label for label in labels if label.casefold() not in rejected]


def compare_entity_extraction_strategies(
    source: SourceRecord,
    *,
    gliner_batch: ConstructionBatch,
    reviewed_labels: list[str],
    provider: OllamaProvider,
    config: HybridGlinerPipelineConfig,
) -> list[ExtractionComparisonRow]:
    """Compare hybrid GLiNER output against lightweight baselines."""

    rows = [comparison_row("ollama_labels_plus_gliner", gliner_batch)]
    if config.compare_regex:
        rows.append(comparison_row("regex_entity_extractor", RegexEntityExtractor().extract(source)))
    if config.compare_gazetteer:
        rows.append(
            comparison_row(
                "gazetteer_entity_extractor",
                GazetteerEntityExtractor(DEFAULT_HYBRID_GAZETTEER).extract(source),
            )
        )
    if config.compare_ollama_graph:
        try:
            rows.append(
                comparison_row(
                    "ollama_graph_extractor",
                    OllamaGraphExtractor(
                        provider,
                        allowed_node_types=reviewed_labels or ["Person", "Organization", "System", "Concept", "Event"],
                    ).extract(source),
                )
            )
        except Exception as exc:  # pragma: no cover - exercised with live Ollama failures.
            rows.append(
                ExtractionComparisonRow(
                    technique="ollama_graph_extractor",
                    error=str(exc),
                )
            )
    return rows


def comparison_row(technique: str, batch: ConstructionBatch) -> ExtractionComparisonRow:
    """Build one comparable extraction summary row."""

    object_count = len(batch.entities) + len(batch.relations)
    evidence_count = sum(1 for entity in batch.entities if entity.evidence) + sum(
        1 for relation in batch.relations if relation.evidence
    )
    confidences = [entity.confidence for entity in batch.entities] + [
        relation.confidence for relation in batch.relations
    ]
    return ExtractionComparisonRow(
        technique=technique,
        entity_count=len(batch.entities),
        relation_count=len(batch.relations),
        labels=sorted({entity.entity_type for entity in batch.entities}),
        evidence_coverage=evidence_count / max(1, object_count),
        average_confidence=mean(confidences) if confidences else 0.0,
    )


__all__ = [
    "ExtractionComparisonRow",
    "HybridGlinerExtractionResult",
    "HybridGlinerPipelineConfig",
    "LabelReviewSettings",
    "apply_manual_label_review",
    "compare_entity_extraction_strategies",
    "comparison_row",
    "extract_entities_with_ollama_labels_and_gliner",
]
