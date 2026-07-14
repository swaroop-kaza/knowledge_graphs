"""Ollama-driven entity label proposal for GLiNER extraction."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from kg_construction.infrastructure.llm.ollama import OllamaProvider
from kg_construction.infrastructure.llm.prompts import (
    DEFAULT_LABEL_PROPOSAL_TEMPLATE,
    LABEL_PROPOSAL_PROMPT_VERSION,
    build_label_proposal_prompt,
    validate_label_payload,
)

LabelReviewStatus = Literal["proposed", "approved", "rejected"]


class LabelProposalSettings(BaseModel):
    """YAML-configurable label proposal and filtering settings."""

    model_config = ConfigDict(extra="forbid")

    prompt_template: str = DEFAULT_LABEL_PROPOSAL_TEMPLATE
    allowed_label_count: int = Field(default=8, ge=1, le=50)
    seed_labels: list[str] = Field(default_factory=list)
    include_patterns: list[str] = Field(default_factory=list)
    exclude_patterns: list[str] = Field(default_factory=lambda: ["^(thing|object|data|text|other|unknown)$"])
    min_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    auto_approve: bool = True


class ProposedEntityLabel(BaseModel):
    """One LLM-proposed candidate label."""

    model_config = ConfigDict(extra="forbid")

    name: str
    description: str = ""
    rationale: str = ""
    confidence: float = Field(default=0.7, ge=0.0, le=1.0)
    status: LabelReviewStatus = "proposed"


class LabelProposalResult(BaseModel):
    """Label proposal output from Ollama plus review-filtered labels."""

    model_config = ConfigDict(extra="forbid")

    proposed_labels: list[ProposedEntityLabel] = Field(default_factory=list)
    reviewed_labels: list[ProposedEntityLabel] = Field(default_factory=list)
    prompt_version: str = LABEL_PROPOSAL_PROMPT_VERSION
    prompt: str = ""
    provider: str = "ollama"
    model: str = ""

    @property
    def reviewed_label_names(self) -> list[str]:
        """Return labels approved for GLiNER."""

        return [label.name for label in self.reviewed_labels if label.status == "approved"]


class OllamaLabelProposalExtractor:
    """Use Ollama to propose GLiNER labels, then review/filter them."""

    name = "ollama_label_proposal_extractor"
    technique = "llm_ollama_label_proposal"

    def __init__(
        self,
        provider: OllamaProvider,
        *,
        settings: LabelProposalSettings | None = None,
    ) -> None:
        self.provider = provider
        self.settings = settings or LabelProposalSettings()

    @classmethod
    def from_yaml(
        cls,
        path: str | Path,
        *,
        provider: OllamaProvider,
    ) -> OllamaLabelProposalExtractor:
        """Build a label proposal extractor from a YAML file."""

        payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        settings_payload = payload.get("label_proposal", payload)
        return cls(provider, settings=LabelProposalSettings.model_validate(settings_payload))

    def propose(self, text: str) -> LabelProposalResult:
        """Propose and review labels for one text chunk."""

        prompt = build_label_proposal_prompt(
            chunk_text=text,
            allowed_label_count=self.settings.allowed_label_count,
            seed_labels=self.settings.seed_labels,
            prompt_template=self.settings.prompt_template,
        )
        payload = self.provider.complete_json(prompt, validator=validate_label_payload)
        proposed = self._labels_from_payload(payload)
        reviewed = self.review(proposed)
        return LabelProposalResult(
            proposed_labels=proposed,
            reviewed_labels=reviewed,
            prompt=prompt,
            provider=self.provider.config.provider,
            model=self.provider.config.model,
        )

    def review(self, labels: list[ProposedEntityLabel]) -> list[ProposedEntityLabel]:
        """Apply deterministic review rules to proposed labels."""

        reviewed: list[ProposedEntityLabel] = []
        seen: set[str] = set()
        for label in labels:
            normalized = normalize_label_name(label.name)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            status: LabelReviewStatus = "approved" if self.settings.auto_approve else "proposed"
            if not self._passes_filters(normalized, label.confidence):
                status = "rejected"
            reviewed.append(label.model_copy(update={"name": normalized, "status": status}))
            if len([item for item in reviewed if item.status == "approved"]) >= self.settings.allowed_label_count:
                break
        return reviewed

    def _labels_from_payload(self, payload: dict[str, object]) -> list[ProposedEntityLabel]:
        labels: list[ProposedEntityLabel] = []
        raw_labels = payload.get("labels", [])
        if not isinstance(raw_labels, list):
            return labels
        for item in raw_labels:
            if isinstance(item, str):
                labels.append(ProposedEntityLabel(name=item))
                continue
            if not isinstance(item, dict):
                continue
            name = str(item.get("name", "")).strip()
            if not name:
                continue
            confidence = item.get("confidence", 0.7)
            labels.append(
                ProposedEntityLabel(
                    name=name,
                    description=str(item.get("description", "")),
                    rationale=str(item.get("rationale", "")),
                    confidence=float(confidence) if isinstance(confidence, int | float | str) else 0.7,
                )
            )
        return labels

    def _passes_filters(self, label: str, confidence: float) -> bool:
        if confidence < self.settings.min_confidence:
            return False
        include_patterns = [re.compile(pattern, flags=re.IGNORECASE) for pattern in self.settings.include_patterns]
        exclude_patterns = [re.compile(pattern, flags=re.IGNORECASE) for pattern in self.settings.exclude_patterns]
        if include_patterns and not any(pattern.search(label) for pattern in include_patterns):
            return False
        return not any(pattern.search(label) for pattern in exclude_patterns)


def normalize_label_name(label: str) -> str:
    """Normalize labels into stable graph-friendly names."""

    cleaned = re.sub(r"[^A-Za-z0-9]+", " ", label).strip()
    if not cleaned:
        return ""
    parts = cleaned.split()
    return "".join(part[:1].upper() + part[1:] for part in parts)


__all__ = [
    "LabelProposalResult",
    "LabelProposalSettings",
    "LabelReviewStatus",
    "OllamaLabelProposalExtractor",
    "ProposedEntityLabel",
    "normalize_label_name",
]
